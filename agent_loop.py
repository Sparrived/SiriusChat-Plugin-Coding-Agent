from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .api import create_pr, fork_repo, sync_fork, get_default_branch
from .skills import (
    ToolRegistry,
    build_default_registry,
    set_workspace_root,
)
from .stream_writer import StreamWriter

logger = logging.getLogger(__name__)


# ── 日志归档 ──


def _archive_agent_log(stream_file: Path, config: dict) -> None:
    """将 agent 工作流日志归档到 logs/archive/ 目录。
    
    每次工作流完成后调用，将 .stream 文件复制到归档目录并清理旧归档。
    """
    if not config.get("log_archive_enabled", True):
        logger.debug("日志归档已禁用，跳过")
        return
    
    if not stream_file.exists():
        logger.warning("归档失败: stream 文件不存在 %s", stream_file)
        return
    
    archive_dir = stream_file.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_name = f"{stream_file.stem}_{timestamp}{stream_file.suffix}"
    archive_path = archive_dir / archive_name
    
    try:
        shutil.copy2(str(stream_file), str(archive_path))
        logger.info("工作流日志已归档: %s → %s", stream_file.name, archive_path.name)
        
        # 清理旧归档，保留最近 N 个
        max_count = int(config.get("log_archive_max_count", 50))
        _clean_old_archives(archive_dir, max_count)
    except Exception as exc:
        logger.error("工作流日志归档失败: %s", exc, exc_info=True)


def _clean_old_archives(archive_dir: Path, max_count: int) -> None:
    """清理归档目录，保留最近 max_count 个日志文件。"""
    if not archive_dir.exists():
        return
    
    try:
        archives = sorted(
            [f for f in archive_dir.iterdir() if f.suffix == ".stream"],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for old_file in archives[max_count:]:
            old_file.unlink(missing_ok=True)
            logger.debug("已清理旧归档: %s", old_file.name)
    except Exception as exc:
        logger.warning("清理旧归档失败: %s", exc)

_VIEWER_SCRIPT = "console_viewer.py"
_CHANGELOG_RETRIES = 3


def _launch_console_viewer(stream_file: Path, keep_open: bool = False) -> subprocess.Popen | None:
    """在独立 CMD 窗口中启动 console_viewer.py。仅 Windows。"""
    if sys.platform != "win32":
        return None

    viewer_script = Path(__file__).resolve().parent / _VIEWER_SCRIPT
    if not viewer_script.exists():
        return None

    try:
        args = [
            "cmd", "/c", "start", "Sirius GitHub Agent",
            "python", str(viewer_script), str(stream_file),
        ]
        if keep_open:
            args.append("--keep-open")
        proc = subprocess.Popen(args)
        return proc
    except Exception as exc:
        logger.warning("无法启动 console viewer: %s", exc)
        return None


def _rmtree_force(path: Path) -> None:
    """强制删除目录（处理 Windows 只读文件/隐藏 .git）"""
    def _on_error(func, p, exc_info):
        os.chmod(p, stat.S_IWRITE)
        try:
            func(p)
        except Exception:
            pass

    shutil.rmtree(str(path), onerror=_on_error, ignore_errors=True)


async def prepare_workspace(repo_name: str, issue_number: int, config: dict) -> Path:
    """准备本地工作区：Fork → Sync → Clone（每次全新）→ 创建分支。"""
    workspace_root = Path(config["workspace_dir"])
    task_dir = workspace_root / f"task_{issue_number}"
    from .api import _resolve_github_username, _write_token_for_repo
    from git import Repo

    username = _resolve_github_username(repo_name, config)

    # 1. Fork（幂等）
    await fork_repo(repo_name, config)

    # 2. Sync upstream
    await sync_fork(repo_name, config)

    # 3. Clone（每次先强制删除旧目录，确保全新 clone）
    pat = _write_token_for_repo(config, repo_name)
    fork_url = f"https://{pat}@github.com/{username}/{repo_name.split('/')[-1]}.git"

    for clone_attempt in range(1, 4):
        if task_dir.exists():
            _rmtree_force(task_dir)
        # 二次确认：目录必须已消失
        if task_dir.exists():
            logger.warning("rmtree 未能完全删除 %s，尝试 cmd 强制清理", task_dir)
            try:
                proc_clean = await asyncio.create_subprocess_exec(
                    "cmd", "/c", "rmdir", "/s", "/q", str(task_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc_clean.communicate()
            except Exception:
                pass
        task_dir.mkdir(parents=True, exist_ok=True)

        proc = await asyncio.create_subprocess_exec(
            "git", "clone", fork_url, str(task_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            logger.info("git clone 成功: %s → %s", repo_name, task_dir)
            break
        err = stderr.decode()[:500] if stderr else "未知错误"
        if "already exists" in err.lower() and clone_attempt < 3:
            logger.warning("git clone 目标存在残留文件，第%d次重试: %s", clone_attempt + 1, task_dir)
            continue
        raise RuntimeError(f"git clone 失败 (exit={proc.returncode}): {err}")

    # 4. 创建修复分支
    repo = Repo(task_dir)
    fix_branch = f"fix-issue-{issue_number}"
    try:
        repo.git.checkout("-b", fix_branch)
    except Exception:
        logger.info("分支 %s 已存在，切换到该分支", fix_branch)
        repo.git.checkout(fix_branch)

    return task_dir


def _tool_schema_text(tool_registry: ToolRegistry) -> str:
    """将工具注册表转为纯文本描述，嵌入 System Prompt。"""
    lines = []
    for schema in tool_registry.get_schema_list():
        func = schema["function"]
        name = func["name"]
        desc = func["description"]
        params = func["parameters"]
        props = params.get("properties", {})
        required = params.get("required", [])

        lines.append(f"  - {name}: {desc}")
        for p_name, p_info in props.items():
            p_type = p_info.get("type", "any")
            p_desc = p_info.get("description", "")
            req_mark = " [必填]" if p_name in required else ""
            lines.append(f"      参数 {p_name} ({p_type}){req_mark}: {p_desc}")
    return "\n".join(lines)


def build_system_prompt(tool_registry: ToolRegistry, workspace_dir: Path) -> str:
    """构建 Agent 的系统 Prompt。人格属性由 generate_raw(inject_persona=True) 自动注入。"""
    tool_schema = _tool_schema_text(tool_registry)

    return f"""你正在进行资深软件工程设计，通过 tool calling 修复一个 GitHub Issue。请以你的角色身份和沟通风格来完成以下工作，包括思考过程、代码注释、变量命名风格和修复方案的表述都应与你的角色设定一致。

## 运行环境

- **操作系统**：Windows，命令行使用 **PowerShell** 语法
- **工作区路径**：{workspace_dir}
- **重要**：不要使用 Unix 命令（head/cat/wc/grep/sed/awk），它们在本环境不可用

## 可用工具

{tool_schema}

## 工具调用规则

当你需要执行操作时，请输出严格的 JSON 格式工具调用，每行一个。可以一次输出多个工具调用（并行执行）：
```json
{{"tool": "工具名1", "args": {{"参数1": "值1", ...}}}}
{{"tool": "工具名2", "args": {{"参数1": "值1", ...}}}}
```

然后我会执行工具并返回结果给你。

当你完成所有修改、确认 Issue 已被修复后，调用 done 工具：
```json
{{"tool": "done", "args": {{}}}}
```

## 推荐工作流程

1. **定位代码**：用 search_content 搜索关键词，获取文件路径与行号
2. **查看上下文**：用 read_file_chunk 读取 search_content 返回的行号附近的代码
3. **精确修改**：用 search_and_replace_block 替换代码块（确保 old_block 在文件中唯一）
4. **完成**：修改完毕后调用 done 工具

## 重要注意事项

### 文件路径
- 所有文件路径都**相对于工作区根目录** {workspace_dir}
- 可以直接使用文件名（如 `closer.py`），工具会自动在工作区中查找
- 也支持子目录路径（如 `plugins/file.py`）和绝对路径
- 路径分隔符使用正斜杠（如 'sirius_chat/webui/static/style.css'）

### 失败处理策略
- 如果某个工具调用返回错误（特别是 read_file_chunk），**不要重试相同参数**，尝试以下替代方案：
  - 扩大/缩小 search_content 的 directory 参数范围
  - 用 search_content 搜索文件中的其他唯一标记来定位代码
  - 用 run_local_test 执行 `powershell -Command "Get-Content <file> -Head N"` 查看文件前 N 行
- 如果一种方法连续失败 2 次，立刻切换策略，不要反复尝试
- search_content 跳过大文件（>256KB），若目标文件很大请用 read_file_chunk

### run_local_test 使用建议
- 测试命令：直接使用 'pytest' 或 'flake8'
- 查看文件内容：`powershell -Command "Get-Content <file> -Head 50"`
- 执行 Python 脚本：先用 search_and_replace_block 创建脚本文件，再用 'python temp_script.py' 运行
- 避免嵌套引号过深的 python -c 命令"""


async def _streaming_generate(
    engine_proxy: Any,
    prompt: str,
    system_prompt: str,
    messages: list[dict] | None,
    model: str | None,
    stream: StreamWriter | None,
) -> str:
    """流式调用 LLM，实时推送 think/reasoning 到 viewer，返回完整文本。"""
    engine = getattr(engine_proxy, "_engine", None)
    provider = getattr(engine, "provider_async", None) if engine else None
    if provider is None or not hasattr(provider, "generate_stream"):
        # 降级：非流式
        result = await engine_proxy.generate_raw(
            prompt=prompt, system_prompt=system_prompt, messages=messages,
            inject_persona=True, model=model, task_name="plugin_raw",
            return_reasoning=True,
        )
        reasoning_text = ""
        if isinstance(result, tuple):
            reasoning_text, content_text = result
        else:
            content_text = result
        if stream:
            if reasoning_text:
                stream.reasoning(reasoning_text)
            stream.think(content_text)
        return content_text or reasoning_text

    # 人格注入
    persona = getattr(engine, "persona", None)
    if persona:
        persona_lines = []
        name = getattr(persona, "name", "")
        if name:
            persona_lines.append(f"你当前的角色身份是「{name}」。")
        summary = getattr(persona, "persona_summary", "")
        if summary:
            persona_lines.append(f"角色简介：{summary}")
        traits = getattr(persona, "personality_traits", [])
        if traits:
            persona_lines.append(f"性格特征：{'、'.join(traits[:3])}")
        style = getattr(persona, "communication_style", "")
        if style:
            persona_lines.append(f"沟通风格：{style}")
        if persona_lines:
            system_prompt = "\n".join(persona_lines) + ("\n\n" + system_prompt if system_prompt else "")

    resolved_model = model
    if resolved_model is None:
        model_router = getattr(engine, "model_router", None)
        if model_router:
            resolved_model = model_router.resolve("plugin_raw").model_name

    msgs: list[dict[str, object]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    if messages:
        msgs.extend(messages)
    msgs.append({"role": "user", "content": prompt})

    from sirius_pulse.providers.base import GenerationRequest
    request = GenerationRequest(
        model=resolved_model or "",
        system_prompt="",
        messages=msgs,
        temperature=0.7,
        max_tokens=4096,
        timeout_seconds=120.0,
        purpose="plugin_raw",
    )

    full_text = ""
    full_reasoning = ""
    try:
        async for chunk_type, text in provider.generate_stream(request):
            if chunk_type == "reasoning":
                full_reasoning += text
            else:
                full_text += text
            if stream:
                if chunk_type == "reasoning":
                    stream.reasoning(full_reasoning)
                else:
                    stream.think(full_text)
    except Exception:
        logger.exception("流式生成失败，降级到非流式")
        result = await engine_proxy.generate_raw(
            prompt=prompt, system_prompt=system_prompt, messages=messages,
            inject_persona=False, model=model, task_name="plugin_raw",
            return_reasoning=True,
        )
        reasoning_text = ""
        if isinstance(result, tuple):
            reasoning_text, content_text = result
        else:
            content_text = result
        if stream:
            if reasoning_text:
                stream.reasoning(reasoning_text)
            stream.think(content_text)
        return content_text or reasoning_text

    if stream:
        if full_reasoning:
            stream.reasoning(full_reasoning)
        stream.think(full_text)
    return full_text


async def call_llm_with_tools(
    messages: list[dict],
    tool_registry: ToolRegistry,
    engine_proxy: Any,
    workspace_dir: Path,
    stream: StreamWriter | None = None,
    config: dict | None = None,
) -> list[dict]:
    """调用 LLM，执行工具调用循环直到 LLM 调用 done 且产生代码变更。

    - 无工具调用 → 警告并继续（不退）
    - 有工具调用 → 执行 → 返回结果
    - 调用 done → 检查 git diff → 有变更则退出，无变更则警告继续
    - 仅 max_rounds 耗尽或 done+diff 满足时退出
    """
    max_tool_rounds = 200
    model = (config or {}).get("model", "") or None
    done_without_changes = 0

    for _round in range(max_tool_rounds):
        system_prompt = messages[0]["content"] if messages else ""
        existing = messages[1:-1]
        user_prompt = messages[-1]["content"] if messages else ""

        result_text = await _streaming_generate(
            engine_proxy,
            prompt=user_prompt,
            system_prompt=system_prompt,
            messages=existing,
            model=model,
            stream=stream,
        )

        tool_calls = _parse_tool_calls(result_text)
        messages.append({"role": "assistant", "content": result_text})

        if not tool_calls:
            messages.append({
                "role": "user",
                "content": (
                    "你没有调用任何工具。请使用 search_content 搜索相关代码、"
                    "read_file_chunk 查看上下文、search_and_replace_block 进行修改。"
                    "修改完成后调用 done 工具。不要只输出文字说明。"
                ),
            })
            continue

        # 执行本轮所有工具
        tool_results: list[tuple[str, str, dict]] = []
        for t in tool_calls:
            tool_name = t.get("tool", "")
            tool_args = t.get("args", {})
            if stream:
                stream.tool_call(tool_name, tool_args)
            result_str = await tool_registry.call(tool_name, **tool_args)
            if stream:
                stream.tool_result(tool_name, result_str)
            tool_results.append((tool_name, result_str, tool_args))

        done_called = any(name == "done" for name, _, _ in tool_results)

        if not done_called:
            results_feedback = "\n\n".join(
                f"[{name}] 返回：\n{res}" for name, res, _ in tool_results
            )
            messages.append({
                "role": "user",
                "content": (
                    f"工具执行结果：\n\n{results_feedback}\n\n"
                    "继续分析或修改代码。修改完成后调用 done 工具。"
                ),
            })
            continue

        # done 被调用 → 检查代码变更
        diff_proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--stat",
            cwd=str(workspace_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        diff_stdout, _ = await diff_proc.communicate()
        if diff_stdout.decode().strip():
            break

        done_without_changes += 1
        if done_without_changes >= 5:
            logger.warning("LLM 连续 %d 次 done 无变更，强制退出", done_without_changes)
            break

        messages.append({
            "role": "user",
            "content": (
                "你调用了 done 但仓库没有任何代码变更。"
                "请定位到需要修改的代码并用 search_and_replace_block 修改，"
                "完成后再次调用 done 工具。"
            ),
        })

    else:
        logger.warning("工具调用轮数达到上限 %d，强制终止", max_tool_rounds)

    return messages


def _parse_tool_calls(text: str) -> list[dict]:
    """从 LLM 输出中解析所有 JSON 工具调用（支持单次回复多个工具调用）。"""
    results: list[dict] = []
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if "tool" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue
    return results


async def generate_changelog(diff: str, issue_data: dict, engine_proxy: Any) -> str:
    """使用 LLM 根据 git diff 生成人类可读的 Changelog。"""
    if not diff.strip():
        return "无文本变更（可能仅修改了二进制文件）。"
    prompt = (
        f"你正在做技术文档撰写工作。请以你的角色身份和沟通风格，根据以下 git diff 撰写一份简洁的中文 Changelog。\n\n"
        f"Issue: #{issue_data.get('number', '?')} - {issue_data.get('title', '')}\n\n"
        f"要求：\n"
        f"1. 以要点列表形式列出每项变更（3-6 条为宜），每条用你的角色口吻描述修改内容和原因\n"
        f"2. 使用 Markdown 格式（每行以 - 开头）\n"
        f"3. 不需要评价代码质量，只描述做了哪些改动及其影响\n"
        f"4. 禁止输出 JSON，直接输出 Markdown 要点\n\n"
        f"Git Diff:\n{diff[:6000]}"
    )
    last_error = None
    for attempt in range(1, _CHANGELOG_RETRIES + 1):
        try:
            result = await engine_proxy.generate_raw(prompt, inject_persona=True)
            return result.strip()
        except Exception as exc:
            last_error = exc
            if attempt < _CHANGELOG_RETRIES:
                logger.info(
                    "Changelog 生成第 %d/%d 次失败，重试中: %s",
                    attempt, _CHANGELOG_RETRIES, exc,
                )
            else:
                logger.error(
                    "Changelog 生成 %d 次重试全部失败: %s",
                    _CHANGELOG_RETRIES, exc,
                )

    raise RuntimeError(
        f"Issue #{issue_data.get('number', '?')} Changelog 生成失败（{_CHANGELOG_RETRIES}次重试）"
    ) from last_error


async def run_agent_loop(
    task_data: dict,
    config: dict,
    engine_proxy: Any,
    adapter: Any | None = None,
) -> str:
    """完整的 agent 修复管线：工作区 → 代码分析 → 修改 → 测试 → PR。

    Returns:
        状态码: "SUCCESS" | "MAX_RETRIES_EXCEEDED" | "ERROR"
    """
    task_id = task_data["task_id"]
    workspace_root = Path(config.get("workspace_dir", "data/github_workspace"))
    workspace_root.mkdir(parents=True, exist_ok=True)
    stream_file = workspace_root / "logs" / f"agent_{task_id}.stream"
    stream = StreamWriter(stream_file)

    viewer_process = None
    if config.get("console_viewer_enabled", True):
        viewer_process = _launch_console_viewer(stream_file, config.get("console_viewer_keep_open", False))

    try:
        stream.phase("PREPARATION", f"Issue #{task_data['issue_number']}: {task_data['issue_title']}")
        workspace_dir = await prepare_workspace(task_data["repo"], task_data["issue_number"], config)
        set_workspace_root(workspace_dir)

        tool_registry = build_default_registry()

        issue_data = {
            "number": task_data["issue_number"],
            "title": task_data["issue_title"],
            "body": task_data.get("issue_body", ""),
        }
        test_command = config.get("test_command", "pytest")
        system_prompt = build_system_prompt(tool_registry, workspace_dir)
        user_message = f"Issue #{issue_data['number']}: {issue_data['title']}\n\n{issue_data.get('body', '')}"

        stream.phase("ANALYSIS", "开始代码检索与定位...")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        messages = await call_llm_with_tools(messages, tool_registry, engine_proxy, workspace_dir, stream, config)

        # ── 验证阶段：测试，失败则重试 ──
        max_retries = config.get("max_retries", 3)
        tests_passed = False

        for attempt in range(1, max_retries + 1):
            stream.phase("VALIDATION", f"第 {attempt} 轮验证")

            # 可选 lint 命令：仅当配置中给出非 pytest 的 lint 命令时运行
            lint_cmd = config.get("lint_command", "")
            if lint_cmd:
                lint_result = await _run_test_cmd(lint_cmd, workspace_dir)
                stream.test_run(lint_cmd, lint_result["success"], lint_result.get("stdout", ""), lint_result.get("stderr", ""))
                if not lint_result["success"]:
                    if attempt < max_retries:
                        stream.retry(attempt, max_retries, lint_result.get("stderr", ""))
                        lint_failure = (
                            (lint_result.get("stdout", "") + "\n" + lint_result.get("stderr", "")).strip()
                            or "(无输出)"
                        )
                        messages.append({
                            "role": "user",
                            "content": f"静态检查失败（第{attempt}次）: {lint_failure}。请修复代码风格/语法问题。",
                        })
                        messages = await call_llm_with_tools(messages, tool_registry, engine_proxy, workspace_dir, stream, config)
                        continue
                    stream.error("静态检查未通过，已达重试上限")
                    stream.done(success=False, summary=f"{lint_cmd} 检查未通过")
                    return "MAX_RETRIES_EXCEEDED"

            test_result = await _run_test_cmd(test_command, workspace_dir)
            stream.test_run(test_command, test_result["success"], test_result.get("stdout", ""), test_result.get("stderr", ""))

            if test_result["success"]:
                tests_passed = True
                break

            if attempt < max_retries:
                stream.retry(attempt, max_retries, test_result.get("stderr", ""))
                failure_output = (
                    (test_result.get("stdout", "") + "\n" + test_result.get("stderr", "")).strip()
                    or "(无输出)"
                )
                messages.append({
                    "role": "user",
                    "content": (
                        f"测试失败（第{attempt}次），以下是完整输出:\n\n{failure_output}\n\n"
                        "请分析以上测试失败的原因，搜索相关代码，用 search_and_replace_block 修复问题，然后调用 done 工具。"
                    ),
                })
                messages = await call_llm_with_tools(messages, tool_registry, engine_proxy, workspace_dir, stream, config)
            else:
                stream.error(f"达到最大重试次数 ({max_retries})，修复失败")
                stream.done(success=False, summary="测试未通过，已达重试上限")
                return "MAX_RETRIES_EXCEEDED"

        if not tests_passed:
            stream.done(success=False, summary="修复失败")
            return "FAILED"

        # 测试通过 → 提交 PR
        stream.phase("COMMIT", "测试通过，开始提交与 PR...")
        pr_url = await _finalize_and_create_pr(
            workspace_dir=workspace_dir,
            repo_name=task_data["repo"],
            issue_number=task_data["issue_number"],
            config=config,
            engine_proxy=engine_proxy,
            issue_data=issue_data,
            adapter=adapter,
        )
        stream.done(success=True, summary="PR 已创建", pr_url=pr_url)
        return "SUCCESS"

    except Exception as exc:
        logger.exception("Agent loop 异常")
        if "stream" in locals():
            stream.error(str(exc))
            stream.done(success=False, summary=f"异常终止: {exc}")
        return "ERROR"

    finally:
        if "stream" in locals():
            stream.close()
            # 工作流日志归档
            _archive_agent_log(stream_file, config)
        if viewer_process:
            try:
                viewer_process.wait(timeout=2)
            except Exception:
                pass


async def _run_test_cmd(test_command: str, workspace_dir: Path) -> dict:
    """运行测试命令的封装（使用 shell 执行以支持复杂命令）。

    自动查找项目根目录（含有 pyproject.toml 的祖先目录），
    确保 pytest 等命令能从项目根运行并找到正确的测试文件。
    """
    # 向上查找含有 pyproject.toml 的项目根目录
    project_root = workspace_dir
    for parent in [workspace_dir] + list(workspace_dir.parents):
        if (parent / "pyproject.toml").exists():
            project_root = parent
            break
        if len(project_root.parents) > 10:
            break

    try:
        proc = await asyncio.create_subprocess_shell(
            test_command,
            cwd=str(project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            return {"success": False, "stdout": "", "stderr": "测试超时（>120秒）"}
        return {
            "success": proc.returncode == 0,
            "stdout": (stdout.decode("utf-8", errors="replace") or "")[-3000:] if stdout else "",
            "stderr": (stderr.decode("utf-8", errors="replace") or "")[-2000:] if stderr else "",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"命令未找到: {test_command.split()[0] if test_command else ''}",
        }


async def _finalize_and_create_pr(
    workspace_dir: Path,
    repo_name: str,
    issue_number: int,
    config: dict,
    engine_proxy: Any,
    issue_data: dict,
    adapter: Any | None = None,
) -> str:
    """提交代码并创建 Pull Request。返回 PR URL。"""
    from git import Repo
    from .api import _resolve_github_username

    repo = Repo(workspace_dir)
    repo.git.add(".")

    # 安全校验：无变更则不提交
    diff_result = repo.git.diff("--cached", "--stat")
    if not diff_result.strip():
        raise RuntimeError("git diff 为空：Agent 未产生任何代码变更，拒绝创建空 PR")

    # 设置仓库级 git 用户身份，确保 GitHub 将提交归因于正确账户
    username = _resolve_github_username(repo_name, config)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", username)
        email = config.get("github_email", "") or f"{username}@users.noreply.github.com"
        cw.set_value("user", "email", email)

    issue_title = issue_data.get("title", f"Fix issue #{issue_number}")
    repo.index.commit(f"Auto-fix issue #{issue_number}: {issue_title[:60]}")

    # 强制推送：每次 clone 都是全新的，远程可能已有同名分支
    fix_branch = f"fix-issue-{issue_number}"
    repo.git.push("--force", "origin", fix_branch)

    pr_title = f"Fix #{issue_number}: {issue_title[:72]}"
    default_branch = await get_default_branch(repo_name, config)
    diff_full = repo.git.diff(default_branch)
    changelog = await generate_changelog(diff_full[:6000], issue_data, engine_proxy)
    pr_body = f"## 自动修复\n\n### 变更摘要\n{changelog}\n\nCloses #{issue_number}"

    pr_result = await create_pr(
        repo_name,
        pr_title,
        pr_body,
        f"{username}:{fix_branch}",
        default_branch,
        config,
    )
    pr_url = pr_result.get("html_url", "") if pr_result else ""

    admin_id = _resolve_admin_id(adapter)
    if adapter and admin_id:
        await adapter.send_private_message(
            admin_id,
            f"修复完成，PR 已创建：{pr_url}",
        )

    shutil.rmtree(workspace_dir, ignore_errors=True)
    return pr_url


def _resolve_admin_id(adapter: Any | None) -> str:
    """从 adapter 读取 root 用户 ID。"""
    if adapter is None:
        return ""
    plugin_config = getattr(adapter, "plugin_config", None)
    if isinstance(plugin_config, dict):
        return str(plugin_config.get("root", "")).strip()
    return ""