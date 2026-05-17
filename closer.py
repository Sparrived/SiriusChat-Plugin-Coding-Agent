"""垃圾 Issue/PR 检测与自动关闭。

用 LLM 分析新提交的 Issue/PR，判断是否属于无意义或垃圾提交，
若判定为垃圾则生成人格化关闭评论并关闭。
"""

from __future__ import annotations

import logging
from typing import Any

from .api import close_issue, close_pr, is_issue_closed

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


# ── 公开 API ──


async def try_close_garbage_issue(
    issue_data: dict[str, Any],
    repo_name: str,
    engine_proxy: Any,
    config: dict[str, Any],
) -> bool:
    """分析 Issue 是否为垃圾内容，若是则生成人格化评论并关闭。

    Returns True 如果已关闭或被关闭，False 表示保留。
    """
    issue_number = issue_data.get("number", 0)
    # 检查是否已被外部关闭
    if await is_issue_closed(repo_name, issue_number, config):
        logger.info("Issue #%d 已被外部关闭，跳过垃圾检测", issue_number)
        return True

    result_text = ""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result_text = await engine_proxy.generate_raw(
                _build_issue_garbage_prompt(issue_data), inject_persona=False, model=config.get("model", ""), json_mode=True,
            )
            result_text = result_text.strip()
            break
        except Exception as exc:
            if attempt < _MAX_RETRIES:
                logger.info("Issue #%d 垃圾检测第 %d/%d 次失败，重试中: %s",
                            issue_data.get("number", "?"), attempt, _MAX_RETRIES, exc)
            else:
                logger.error("Issue #%d 垃圾检测 %d 次全部失败，跳过关闭",
                             issue_data.get("number", "?"), _MAX_RETRIES)

    if not result_text:
        return False

    parsed = _parse_garbage_result(result_text)
    if not parsed["is_garbage"]:
        return False

    close_msg = await _generate_close_comment(issue_data, repo_name, engine_proxy, parsed["reason"])
    await close_issue(repo_name, issue_data["number"], close_msg, config)
    logger.info("Issue #%d 已自动关闭", issue_data["number"])
    return True


async def try_close_garbage_pr(
    pr_data: dict[str, Any],
    repo_name: str,
    engine_proxy: Any,
    config: dict[str, Any],
) -> bool:
    """分析 PR 是否为垃圾内容，若是则生成人格化评论并关闭。

    Returns True 如果已关闭或被关闭，False 表示保留。
    """
    pr_number = pr_data.get("number", 0)
    # 检查是否已被外部关闭（PR 状态通过 issue API 查询）
    if await is_issue_closed(repo_name, pr_number, config):
        logger.info("PR #%d 已被外部关闭，跳过垃圾检测", pr_number)
        return True

    # 获取 PR 完整信息（webhook payload 的 body 可能为空）
    from .api import get_pr, get_pr_diff
    full_pr = await get_pr(repo_name, pr_number, config)
    if full_pr:
        pr_data = full_pr

    # 获取 diff 用于内容判断
    diff_text = ""
    try:
        diff_text = await get_pr_diff(repo_name, pr_number, config)
    except Exception:
        pass

    result_text = ""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result_text = await engine_proxy.generate_raw(
                _build_pr_garbage_prompt(pr_data, diff_text), inject_persona=False, model=config.get("model", ""), json_mode=True,
            )
            result_text = result_text.strip()
            break
        except Exception as exc:
            if attempt < _MAX_RETRIES:
                logger.info("PR #%d 垃圾检测第 %d/%d 次失败，重试中: %s",
                            pr_data.get("number", "?"), attempt, _MAX_RETRIES, exc)
            else:
                logger.error("PR #%d 垃圾检测 %d 次全部失败，跳过关闭",
                             pr_data.get("number", "?"), _MAX_RETRIES)

    if not result_text:
        return False

    parsed = _parse_garbage_result(result_text)
    if not parsed["is_garbage"]:
        return False

    close_msg = await _generate_close_comment(pr_data, repo_name, engine_proxy, parsed["reason"])
    await close_pr(repo_name, pr_data.get("number", 0), close_msg, config)
    logger.info("PR #%d 已自动关闭", pr_data.get("number", 0))
    return True


# ── 人格化关闭评论生成 ──


async def _generate_close_comment(
    submission_data: dict[str, Any],
    repo_name: str,
    engine_proxy: Any,
    reason: str,
) -> str:
    """生成人格化关闭评论（inject_persona=True）。

    仿 commenter.generate_issue_comment 的模式：
    - 感谢提交 ✓
    - 说明不符合项目要求 ✓
    - 关闭通知 ✓
    - 整体语气由 persona 注入决定 ✓
    """
    kind = "Pull Request" if "pull_request" in submission_data else "Issue"
    title = submission_data.get("title", "")
    body = submission_data.get("body", "")[:2000] or "（无内容）"

    prompt = f"""你正在关闭一个不合适的 GitHub {kind}。请以你的角色身份撰写一条公开关闭评论。

{kind} #{submission_data.get('number', '?')}: {title}

提交内容:
{body}

自动判定关闭原因: {reason}

关闭评论要求：
1. 感谢用户提交，语气友好但坚定，展现你的角色个性
2. 简明说明此提交不符合项目范围/质量标准
3. 提示如有异议可联系管理员
4. 长度控制在 80-150 字
5. 输出纯文本（Markdown 格式，不要代码块包裹）

请直接输出关闭评论正文，不要包含任何前缀说明。"""

    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = await engine_proxy.generate_raw(prompt, inject_persona=True)
            return result.strip()
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                logger.info("生成关闭评论第 %d/%d 次失败，重试中: %s", attempt, _MAX_RETRIES, exc)
            else:
                logger.error("生成关闭评论 %d 次全部失败", _MAX_RETRIES)

    return reason or "此提交经自动分析判定为不适用，已自动关闭。"


# ── 垃圾判定 Prompt ──


def _build_issue_garbage_prompt(issue_data: dict[str, Any]) -> str:
    return f"""你正在审核一个 GitHub Issue，判断它是否属于垃圾/无意义提交。

判定为垃圾的标准（满足任意一条即可）：
1. 内容为纯广告、推广链接
2. 内容完全与项目无关（如随机字符、测试、空白内容）
3. 标题和内容均为无意义的占位符（如 "test"、"..."、"asdf"）
4. 明确是恶意或滥用性质的提交

注意：如果 Issue 仅仅描述不够清晰但仍有合理诉求，则判定为正常提交。

Issue #{issue_data.get('number', '?')}: {issue_data.get('title', '')}

Issue 内容:
{issue_data.get('body', '')[:3000] or '（无内容）'}

请以严格的 JSON 格式输出，不要包含其他内容：
{{"is_garbage": true或false, "reason": "若判定为垃圾，给出简短中文关闭理由（不超过80字）；否则留空"}}"""


def _build_pr_garbage_prompt(pr_data: dict[str, Any], diff: str = "") -> str:
    diff_section = ""
    if diff.strip():
        diff_section = f"\n\n代码变更 (diff，前6000字):\n{diff[:6000]}"
    return f"""你正在审核一个 GitHub Pull Request，判断它是否属于垃圾/无意义提交。

判定为垃圾的标准（满足任意一条即可）：
1. 内容为纯广告、推广链接
2. 内容完全与项目无关（如随机字符、测试、空白内容）
3. 标题和内容均为无意义的占位符（如 "test"、"..."、"asdf"）
4. 没有任何实质代码变更或仅包含故意破坏性修改
5. 明确是恶意或滥用性质的提交

注意：如果 PR 包含合理的代码改动哪怕很小，则判定为正常提交。以下情况不是垃圾：自动修复 PR、测试通过后的代码合并、包含实质性 diff 的提交。

PR #{pr_data.get('number', '?')}: {pr_data.get('title', '')}

PR 内容:
{pr_data.get('body', '')[:3000] or '（无描述）'}{diff_section}

请以严格的 JSON 格式输出，不要包含其他内容：
{{"is_garbage": true或false, "reason": "若判定为垃圾，给出简短中文关闭理由（不超过80字）；否则留空"}}"""


def _parse_garbage_result(text: str) -> dict[str, Any]:
    import json as _json
    import re

    text = text.strip()

    # try full-text parse first (covers multi-line JSON)
    try:
        data = _json.loads(text)
        if isinstance(data, dict) and "is_garbage" in data:
            return {
                "is_garbage": bool(data.get("is_garbage", False)),
                "reason": str(data.get("reason", "")).strip()[:200],
            }
    except (_json.JSONDecodeError, ValueError):
        pass

    # try extracting JSON block between { and }
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            data = _json.loads(m.group())
            if isinstance(data, dict) and "is_garbage" in data:
                return {
                    "is_garbage": bool(data.get("is_garbage", False)),
                    "reason": str(data.get("reason", "")).strip()[:200],
                }
        except (_json.JSONDecodeError, ValueError):
            pass

    # strip markdown fences then retry
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    if clean != text:
        try:
            data = _json.loads(clean)
            if isinstance(data, dict) and "is_garbage" in data:
                return {
                    "is_garbage": bool(data.get("is_garbage", False)),
                    "reason": str(data.get("reason", "")).strip()[:200],
                }
        except (_json.JSONDecodeError, ValueError):
            pass

    # fallback: line-by-line
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and "is_garbage" in line:
            try:
                data = _json.loads(line)
                if isinstance(data, dict) and "is_garbage" in data:
                    return {
                        "is_garbage": bool(data.get("is_garbage", False)),
                        "reason": str(data.get("reason", "")).strip()[:200],
                    }
            except (_json.JSONDecodeError, ValueError):
                continue
    logger.warning("垃圾判定 JSON 解析失败，保留: %s", text[:200])
    return {"is_garbage": False, "reason": ""}
