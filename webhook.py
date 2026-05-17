"""Plugin webhook 事件处理器（业务逻辑层）。

Issue opened → 垃圾检测 → 自动标签 → 入队到 IssueTracker（后台自主收集信息）。
PR opened → 垃圾检测 → 自动审阅。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .closer import try_close_garbage_issue, try_close_garbage_pr
from .labeler import apply_labels_to_issue, auto_label_issue
from .review import auto_review_pr, has_existing_review

logger = logging.getLogger(__name__)

_tracker: Any = None


def set_tracker(tracker: Any) -> None:
    global _tracker
    _tracker = tracker


async def handle_issue_opened(
    body: dict[str, Any],
    config: dict[str, Any],
    adapter: Any,
    engine_proxy: Any,
    data_store: Any,
) -> None:
    """处理 Issue 打开事件：垃圾检测 → 自动标签 → 入队后台信息收集。"""
    issue_data = body["issue"]
    repo_name = body["repository"]["full_name"]
    admin_id = _resolve_admin_id(adapter)
    issue_number = issue_data["number"]

    # 0. 垃圾检测
    if config.get("auto_close_garbage", True):
        try:
            closed = await try_close_garbage_issue(issue_data, repo_name, engine_proxy, config)
            if closed:
                if admin_id:
                    await adapter.send_private_message(
                        admin_id,
                        f"Issue #{issue_number}: {issue_data['title']} 已自动关闭（判定为垃圾）\n"
                        f"仓库: {repo_name}",
                    )
                return
        except Exception as exc:
            logger.error("Issue #%d 垃圾检测失败: %s", issue_number, exc, exc_info=True)

    labels: list[str] = []

    # 1. 自动标签
    if config.get("auto_label", True):
        try:
            labels = await auto_label_issue(issue_data, repo_name, config, engine_proxy)
            ok = await apply_labels_to_issue(repo_name, issue_number, labels, config)
            if ok:
                logger.info("Issue #%d 自动标签: %s", issue_number, labels)
            else:
                logger.warning("Issue #%d 自动标签 API 调用失败（HTTP非200）: %s", issue_number, labels)
        except Exception as exc:
            logger.error("Issue #%d 自动标签失败: %s", issue_number, exc, exc_info=True)

    # 2. 入队到 IssueTracker → 后台自主收集信息
    if _tracker is not None:
        task_id = _tracker.enqueue(
            issue_number=issue_number,
            repo=repo_name,
            title=issue_data["title"],
            body=issue_data.get("body", ""),
            labels=labels,
        )
        label_str = " ".join(f"[{l}]" for l in labels) if labels else "（未自动标签）"
        if admin_id:
            await adapter.send_private_message(
                admin_id,
                f"新 Issue #{issue_number}: {issue_data['title']}\n"
                f"标签: {label_str}\n"
                f"仓库: {repo_name}\n"
                f"状态: 后台信息收集中\n"
                f"回复 /gh {task_id} status 查看进度",
            )
    else:
        logger.warning("IssueTracker 未初始化，Issue #%d 仅标签未入队", issue_number)


async def handle_pr_event(
    body: dict[str, Any],
    config: dict[str, Any],
    adapter: Any,
    engine_proxy: Any,
) -> None:
    """处理 PR 事件：垃圾检测 → 自动代码审阅。"""
    pr_data = body["pull_request"]
    repo_name = body["repository"]["full_name"]
    pr_number = pr_data.get("number", 0)
    pr_title = pr_data.get("title", "") or f"PR #{pr_number}"
    action = body.get("action", "")

    # 0. 垃圾检测
    if config.get("auto_close_garbage", True):
        try:
            closed = await try_close_garbage_pr(pr_data, repo_name, engine_proxy, config)
            if closed:
                admin_id = _resolve_admin_id(adapter)
                if admin_id:
                    await adapter.send_private_message(
                        admin_id,
                        f"PR #{pr_number}: {pr_title} 已自动关闭（判定为垃圾）\n"
                        f"仓库: {repo_name}",
                    )
                return
        except Exception as exc:
            logger.error("PR #%d 垃圾检测失败: %s", pr_number, exc, exc_info=True)

    if action == "synchronize":
        already_reviewed = await has_existing_review(repo_name, pr_number, config)
        review_mode = "incremental" if already_reviewed else "quick"
    else:
        review_mode = "quick"

    try:
        result = await auto_review_pr(pr_data, repo_name, engine_proxy, config, review_mode)
        if "error" in result:
            logger.error("PR #%d 审阅失败: %s", pr_number, result["error"])
            return

        admin_id = _resolve_admin_id(adapter)
        if admin_id:
            pr_url = pr_data["html_url"]
            verdict_emoji = {"approve": "OK", "comment": "COMMENT", "request_changes": "CHANGES"}
            emoji = verdict_emoji.get(result.get("verdict", ""), "BOT")
            await adapter.send_private_message(
                admin_id,
                f"[{emoji}] PR #{pr_number} 自动审阅完成\n"
                f"标题: {pr_title}\n"
                f"结论: {result.get('verdict', 'N/A')}（{result.get('issues_count', 0)} 个问题）\n"
                f"摘要: {result.get('summary', '')}\n"
                f"链接: {pr_url}",
            )
    except Exception as exc:
        logger.error("PR #%d 审阅后台任务异常: %s", pr_number, exc, exc_info=True)


def _resolve_admin_id(adapter: Any) -> str:
    if adapter is None:
        return ""
    plugin_config = getattr(adapter, "plugin_config", None)
    if isinstance(plugin_config, dict):
        return str(plugin_config.get("root", "")).strip()
    return ""
