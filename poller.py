"""GitHub API 轮询模块 —— 替代 Webhook，通过定时 API 调用检测新 Issue 和 PR。

不需要公网 IP / ngrok / frp，适合无法接收外部 HTTP 请求的环境。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .api import list_repo_issues, list_repo_pulls
from .commenter import generate_issue_comment, post_comment
from .labeler import apply_labels_to_issue, auto_label_issue
from .review import auto_review_pr, has_existing_review

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 60  # 默认轮询间隔（秒）


async def start_polling_loop(
    config: dict[str, Any],
    adapter: Any,
    engine_proxy: Any,
    data_store: Any,
    *,
    on_stop: asyncio.Event | None = None,
) -> asyncio.Task:
    """启动后台轮询任务。返回 Task 对象，供 on_unload 取消。"""
    interval = int(config.get("poll_interval_seconds", _DEFAULT_INTERVAL))
    interval = max(interval, 30)  # 最少 30 秒，避免触发 GitHub 限流

    async def _poll():
        logger.info("轮询模式已启动（间隔 %d 秒），不再依赖 Webhook", interval)
        while True:
            if on_stop and on_stop.is_set():
                logger.info("轮询任务收到停止信号，退出")
                return
            try:
                await _poll_once(config, adapter, engine_proxy, data_store)
            except Exception as exc:
                logger.error("轮询周期异常: %s", exc, exc_info=True)
            await asyncio.sleep(interval)

    return asyncio.create_task(_poll())


async def _poll_once(
    config: dict[str, Any],
    adapter: Any,
    engine_proxy: Any,
    data_store: Any,
) -> None:
    """执行一次完整的轮询周期。"""
    repos = config.get("repos", [])
    if not repos:
        return

    # 去重状态：已处理过的 Issue/PR 编号（按仓库存储）
    seen_issues_key = "_poll_seen_issues"
    seen_prs_key = "_poll_seen_prs"
    seen_issues: dict[str, set[int]] = _load_seen(data_store, seen_issues_key)
    seen_prs: dict[str, set[int]] = _load_seen(data_store, seen_prs_key)

    for repo in repos:
        if repo not in seen_issues:
            seen_issues[repo] = set()
        if repo not in seen_prs:
            seen_prs[repo] = set()

        # ── 检测新 Issue ──
        if config.get("auto_label", True) or config.get("auto_comment", True):
            try:
                issues = await list_repo_issues(repo, config, state="open", per_page=10)
                for issue_data in issues:
                    issue_number = issue_data["number"]
                    if issue_number in seen_issues[repo]:
                        continue
                    seen_issues[repo].add(issue_number)
                    logger.info("轮询发现新 Issue #%d（仓库 %s）", issue_number, repo)
                    await _process_new_issue(issue_data, repo, config, adapter, engine_proxy, data_store)
            except Exception as exc:
                logger.error("轮询 Issue 时出错 %s: %s", repo, exc, exc_info=True)

        # ── 检测需要审阅的 PR ──
        if config.get("auto_review", True):
            try:
                pulls = await list_repo_pulls(repo, config, state="open", per_page=10)
                for pr_data in pulls:
                    pr_number = pr_data.get("number", 0)
                    updated_at = pr_data.get("updated_at", "")
                    pr_key = f"{pr_number}:{updated_at}"
                    if pr_key in seen_prs[repo]:
                        continue
                    seen_prs[repo].add(pr_key)

                    # 检查是否已有审阅
                    already = await has_existing_review(repo, pr_number, config)
                    if already:
                        logger.debug("PR #%d 已有审阅，跳过", pr_number)
                        continue

                    logger.info("轮询发现需审阅 PR #%d（仓库 %s）", pr_number, repo)
                    await _process_pr(repo, pr_number, config, adapter, engine_proxy)
            except Exception as exc:
                logger.error("轮询 PR 时出错 %s: %s", repo, exc, exc_info=True)

    # 持久化去重状态
    _save_seen(data_store, seen_issues_key, seen_issues)
    _save_seen(data_store, seen_prs_key, seen_prs)


async def _process_new_issue(
    issue_data: dict,
    repo_name: str,
    config: dict,
    adapter: Any,
    engine_proxy: Any,
    data_store: Any,
) -> None:
    """处理新发现的 Issue：标签 → 回复 → 直接进入信息收集（跳过管理员审批）。"""
    issue_number = issue_data["number"]
    labels: list[str] = []

    if config.get("auto_label", True):
        try:
            labels = await auto_label_issue(issue_data, repo_name, config, engine_proxy)
            await apply_labels_to_issue(repo_name, issue_number, labels, config)
            logger.info("Issue #%d 自动标签: %s", issue_number, labels)
        except Exception as exc:
            logger.error("Issue #%d 自动标签失败: %s", issue_number, exc, exc_info=True)

    if config.get("auto_comment", True):
        try:
            comment = await generate_issue_comment(issue_data, labels, repo_name, engine_proxy, config)
            await post_comment(repo_name, issue_number, comment, config)
        except Exception as exc:
            logger.error("Issue #%d 智能回复失败: %s", issue_number, exc, exc_info=True)

    # 直接创建 tracker 任务状态，开始信息收集（跳过管理员审批步骤）
    import uuid
    from .tracker import IssueTracker, IssueState, _PREFIX

    task_id = uuid.uuid4().hex[:12]
    state = IssueState(
        issue_number=issue_number,
        repo=repo_name,
        title=issue_data.get("title", ""),
        body=issue_data.get("body", ""),
        labels=labels,
        status="GATHERING_INFO",
        task_id=task_id,
    )
    data_store.set(f"{_PREFIX}{task_id}", state)

    logger.info("Issue #%d 已创建 tracker 任务 %s，跳过审批直接进入信息收集", issue_number, task_id)

    admin_id = _resolve_admin_id(adapter)
    if admin_id:
        label_str = " ".join(f"[{l}]" for l in labels) if labels else "（未自动标签）"
        await adapter.send_private_message(
            admin_id,
            f"📥 新 Issue #{issue_number}: {issue_data.get('title', '')}\n"
            f"🏷 标签: {label_str}\n"
            f"📋 仓库: {repo_name}\n"
            f"🔄 已跳过审批，直接进入信息收集阶段喵~",
        )


async def _process_pr(
    repo_name: str,
    pr_number: int,
    config: dict,
    adapter: Any,
    engine_proxy: Any,
) -> None:
    """对 PR 执行自动审阅。"""
    try:
        result = await auto_review_pr(
            {"number": pr_number, "title": "", "body": ""},
            repo_name, engine_proxy, config, "quick",
        )
        if "error" in result:
            logger.error("PR #%d 审阅失败: %s", pr_number, result["error"])
            return

        admin_id = _resolve_admin_id(adapter)
        if admin_id:
            verdict_emoji = {"approve": "✅", "comment": "💬", "request_changes": "🔴"}
            emoji = verdict_emoji.get(result.get("verdict", ""), "🤖")
            await adapter.send_private_message(
                admin_id,
                f"{emoji} PR #{pr_number} 自动审阅完成\n"
                f"📋 仓库: {repo_name}\n"
                f"📊 结论: {result.get('verdict', 'N/A')}（{result.get('issues_count', 0)} 个问题）\n"
                f"📝 摘要: {result.get('summary', '')}",
            )
    except Exception as exc:
        logger.error("PR #%d 审阅异常: %s", pr_number, exc, exc_info=True)


def _load_seen(data_store: Any, key: str) -> dict[str, set[int]]:
    """从 data_store 加载去重集合。"""
    raw = None
    if data_store:
        raw = data_store.get(key)
    if isinstance(raw, dict):
        return {k: set(v) for k, v in raw.items()}
    return {}


def _save_seen(data_store: Any, key: str, seen: dict[str, set[int]]) -> None:
    """持久化去重集合。"""
    if not data_store:
        return
    serializable = {k: list(v) for k, v in seen.items()}
    data_store.set(key, serializable)


def _resolve_admin_id(adapter: Any) -> str:
    if adapter is None:
        return ""
    plugin_config = getattr(adapter, "plugin_config", None)
    if isinstance(plugin_config, dict):
        return str(plugin_config.get("root", "")).strip()
    return ""
