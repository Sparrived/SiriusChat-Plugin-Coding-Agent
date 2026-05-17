from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable


@dataclass
class ToolDef:
    """工具定义。"""
    name: str
    description: str
    parameters: dict  # JSON Schema 格式
    handler: Callable[..., Awaitable[Any]]


class ToolRegistry:
    """Agent Loop 内部工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def get_schema_list(self) -> list[dict]:
        """生成 OpenAI function calling 格式的工具列表。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    async def call(self, name: str, **kwargs: Any) -> str:
        """调用工具并返回结果字符串。"""
        tool = self._tools.get(name)
        if tool is None:
            return f"未知工具: {name}"
        try:
            result = await tool.handler(**kwargs)
            return str(result)
        except Exception as e:
            return f"工具执行失败: {e}"


# ── 工作区路径安全校验 ──

_workspace_root: Path | None = None


def set_workspace_root(root: Path) -> None:
    global _workspace_root
    _workspace_root = root.resolve()


def _validate_path(file_path: str) -> Path:
    """确保文件路径在 workspace 内，防止路径穿越攻击。

    支持两种路径：
    - 绝对路径（如 D:\\path\\to\\file.py）
    - 相对于工作区根的路径（如 closer.py、plugins/file.py）

    Returns:
        解析后的绝对路径。
    """
    if _workspace_root is None:
        raise RuntimeError("工作区未初始化")

    # 1. 尝试从进程 CWD 解析
    resolved = Path(file_path).resolve()
    if str(resolved).startswith(str(_workspace_root)):
        return resolved

    # 2. 尝试从工作区根解析（支持相对路径如 "closer.py"）
    alt_resolved = (_workspace_root / file_path).resolve()
    # 防止路径穿越攻击：检查是否仍在工作区内
    if str(alt_resolved).startswith(str(_workspace_root)):
        return alt_resolved

    raise PermissionError(
        f"禁止访问工作区外的文件: {file_path}\n"
        f"  工作区根目录: {_workspace_root}\n"
        f"  提示：请使用相对于工作区根的路径，如 '{_workspace_root.name}/your_file.py'"
    )


# ── 4 个工具实现 ──


async def search_content(keyword: str, directory: str = ".") -> str:
    """搜索关键词，返回文件路径与行号。使用 Python 原生实现，无外部依赖。
    搜索范围始终限制在工作区根目录内，外部路径会被截断。"""
    base = (Path(directory) if directory != "." else Path(".")).resolve()
    # 限制在工作区内
    if _workspace_root is not None:
        try:
            base = base.relative_to(_workspace_root)
        except ValueError:
            base = Path(".")
        base = (_workspace_root / base).resolve()
    if not base.exists():
        return f"搜索目录不存在: {directory}"
    results: list[str] = []
    noise_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode", ".mypy_cache", ".pytest_cache"}
    for file_path in base.rglob("*"):
        if not file_path.is_file():
            continue
        parts = set(file_path.parts) & noise_dirs
        if parts:
            continue
        try:
            if file_path.suffix and file_path.stat().st_size > 256_000:
                continue
        except OSError:
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for line_no, line in enumerate(content.splitlines(), 1):
            if keyword.lower() in line.lower():
                rel = file_path.relative_to(base)
                results.append(f"{rel}:{line_no}:{line[:200].strip()}")
                if len(results) >= 50:
                    return "\n".join(results) + "\n（结果已截断至 50 条）"
    return "\n".join(results) if results else "未找到匹配结果"


async def read_file_chunk(file_path: str, start_line: int, end_line: int) -> str:
    """按行读取文件片段，防止超大文件撑爆 Token。"""
    try:
        resolved = _validate_path(file_path)
        lines = await asyncio.to_thread(_read_lines, str(resolved), start_line, end_line)
        return "\n".join(lines)
    except Exception as e:
        return f"读取文件失败: {e}"


def _read_lines(file_path: str, start_line: int, end_line: int) -> list[str]:
    result: list[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i > end_line:
                break
            if i >= start_line:
                result.append(f"{i}:{line.rstrip()}")
    return result


async def search_and_replace_block(file_path: str, old_block: str, new_block: str) -> str:
    """精确字符串替换，校验唯一性。"""
    try:
        resolved = _validate_path(file_path)
    except PermissionError as e:
        return str(e)
    content = resolved.read_text(encoding="utf-8")
    count = content.count(old_block)
    if count == 0:
        return f"错误：未在 {file_path} 中找到目标代码块"
    if count > 1:
        positions = []
        idx = 0
        while True:
            pos = content.find(old_block, idx)
            if pos == -1:
                break
            line_no = content[:pos].count("\n") + 1
            positions.append(f"L{line_no}")
            idx = pos + 1
        return f"错误：目标代码块在 {file_path} 中出现 {count} 次（不唯一）。匹配位置：{', '.join(positions)}。请提供更多上下文使匹配唯一。"
    new_content = content.replace(old_block, new_block, 1)
    resolved.write_text(new_content, encoding="utf-8")
    return f"成功替换 {file_path} 中的代码块"


async def run_local_test(test_command: str) -> dict:
    """在沙盒中运行测试命令（Windows 环境，使用 shell 执行以支持管道、重定向、引号）。"""
    try:
        proc = await asyncio.create_subprocess_shell(
            test_command,
            cwd=str(_workspace_root) if _workspace_root else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            proc.kill()
            return {"success": False, "stdout": "", "stderr": "测试超时（>60秒）"}
        return {
            "success": proc.returncode == 0,
            "stdout": stdout.decode("utf-8", errors="replace") if stdout else "",
            "stderr": stderr.decode("utf-8", errors="replace") if stderr else "",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"命令未找到（Windows 环境，请使用 PowerShell 兼容命令）: {test_command.split()[0] if test_command else ''}",
        }


async def tool_done() -> str:
    """声明工作完成，触发验证流程。"""
    return "DONE"


def build_default_registry() -> ToolRegistry:
    """创建并注册 4 个默认工具的 ToolRegistry。"""
    registry = ToolRegistry()

    registry.register(ToolDef(
        name="search_content",
        description=(
            "在指定目录中递归搜索文件内容中的关键词（大小写不敏感），"
            "跳过 .git/node_modules/__pycache__ 等噪音目录，"
            "返回匹配的文件路径、行号和行内容摘要（上限 50 条）。"
            "用于快速定位相关代码的位置。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "搜索关键词（大小写不敏感），如 'slider'、'renderPluginSettingsForm'",
                },
                "directory": {
                    "type": "string",
                    "description": "搜索目录的相对于工作区根的路径，如 '.' 或 'sirius_chat/webui/static'",
                },
            },
            "required": ["keyword"],
        },
        handler=search_content,
    ))

    registry.register(ToolDef(
        name="read_file_chunk",
        description=(
            "按行号范围读取文件内容片段，防止超大文件撑爆 Token。"
            "当 search_content 已定位到目标行号后，用此工具读取上下文代码。"
            "start_line 和 end_line 都必填（均从 1 开始计数）。"
            "file_path 可以是相对于工作区根的文件路径，或是简单的文件名（如 'style.css'）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "文件路径（相对于工作区根，如 'closer.py' 或 'sirius_chat/webui/static/style.css'）",
                },
                "start_line": {
                    "type": "integer",
                    "description": "起始行号（从 1 开始，含该行）",
                },
                "end_line": {
                    "type": "integer",
                    "description": "结束行号（从 1 开始，含该行）",
                },
            },
            "required": ["file_path", "start_line", "end_line"],
        },
        handler=read_file_chunk,
    ))

    registry.register(ToolDef(
        name="search_and_replace_block",
        description=(
            "在文件中精确替换代码块。要求 old_block 在目标文件中仅出现一次，"
            "否则会报错并返回所有匹配位置。请提供足够的上下文行使匹配唯一。"
            "修改成功后文件即刻保存。"
            "file_path 可以是相对于工作区根的文件路径，或是简单的文件名（如 'style.css'）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "文件路径（相对于工作区根，如 'closer.py' 或 'sirius_chat/webui/static/style.css'）",
                },
                "old_block": {
                    "type": "string",
                    "description": "要被替换的原始代码块（必须与文件中内容完全一致，包括空白字符）",
                },
                "new_block": {
                    "type": "string",
                    "description": "替换后的新代码块",
                },
            },
            "required": ["file_path", "old_block", "new_block"],
        },
        handler=search_and_replace_block,
    ))

    registry.register(ToolDef(
        name="run_local_test",
        description=(
            "在工作区沙盒中运行一条 shell 命令（Windows PowerShell 环境）。"
            "支持管道、重定向、引号。用于运行 pytest、flake8 等测试， "
            "或执行一次性 Python 脚本查看文件内容。"
            "注意：不要使用 Unix 命令（如 head/cat/wc/grep），"
            "应使用 PowerShell 命令或 'python -c ...'。"
            "若需执行复杂 Python 逻辑，建议先用 search_and_replace_block "
            "创建/覆盖临时 .py 脚本文件再运行。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "test_command": {
                    "type": "string",
                    "description": "要执行的命令，如 'pytest'、'python temp.py'、'powershell -Command \"Get-Content file.css -Head 20\"'",
                },
            },
            "required": ["test_command"],
        },
        handler=run_local_test,
    ))

    registry.register(ToolDef(
        name="done",
        description="确认所有修改已完成，触发代码验证、提交和 PR 创建流程",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=tool_done,
    ))

    return registry
