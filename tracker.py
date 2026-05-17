"""Issue 信息队列 & 状态机。

每个活跃 Issue 维护一份 IssueState，通过 DataStore 持久化。
后台轮询检查状态并驱动信息收集循环。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .api import get_file_content, get_issue_comments
from .gatherer import analyze_and_gather

logger = logging.getLogger(__name__)

_TRACKER_TICK = 60
_PREFIX = "tracker_"


class IssueState:
    __slots__ = (
        "issue_number", "repo", "title", "body", "labels",
        "status", "conversation", "gathered_summary", "last_comment_fetched_at",
        "questions_asked", "last_activity", "task_id", "repo_context",
    )

    def __init__(
        self,
        issue_number: int,
        repo: str,
        title: str = "",
        body: str = "",
        labels: list[str] | None = None,
        status: str = "GATHERING_INFO",
        conversation: list[dict] | None = None,
        gathered_summary: str = "",
        last_comment_fetched_at: float = 0.0,
        questions_asked: int = 0,
        last_activity: float | None = None,
        task_id: str = "",
        repo_context: str = "",
    ) -> None:
        self.issue_number = issue_number
        self.repo = repo
        self.title = title
        self.body = body
        self.labels = labels or []
        self.status = status
        self.conversation = conversation or []
        self.gathered_summary = gathered_summary
        self.last_comment_fetched_at = last_comment_fetched_at
        self.questions_asked = questions_asked
        self.last_activity = last_activity or time.time()
        self.task_id = task_id
        self.repo_context = repo_context

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_number": self.issue_number,
            "repo": self.repo,
            "title": self.title,
            "body": self.body,
            "labels": self.labels,
            "status": self.status,
            "conversation": self.conversation,
            "gathered_summary": self.gathered_summary,
            "last_comment_fetched_at": self.last_comment_fetched_at,
            "questions_asked": self.questions_asked,
            "last_activity": self.last_activity,
            "task_id": self.task_id,
            "repo_context": self.repo_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IssueState:
        return cls(
            issue_number=int(data.get("issue_number", 0)),
            repo=str(data.get("repo", "")),
            title=str(data.get("title", "")),
            body=str(data.get("body", "")),
            labels=list(data.get("labels", [])),
            status=str(data.get("status", "GATHERING_INFO")),
            conversation=list(data.get("conversation", [])),
            gathered_summary=str(data.get("gathered_summary", "")),
            last_comment_fetched_at=float(data.get("last_comment_fetched_at", 0)),
            questions_asked=int(data.get("questions_asked", 0)),
            last_activity=float(data.get("last_activity", 0)) or time.time(),
            task_id=str(data.get("task_id", "")),
            repo_context=str(data.get("repo_context", "")),
        )

    def add_conversation(self, role: str, content: str) -> None:
        self.conversation.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        self.last_activity = time.time()


class IssueTracker:
    """维护所有活跃 Issue 的状态，驱动信息收集循环。"""

    def __init__(
        self,
        data_store: Any,
        config: dict[str, Any],
        engine_proxy: Any,
        plugin_ctx: Any,
    ) -> None:
        self._store = data_store
        self._config = config
        self._engine_proxy = engine_proxy
        self._plugin_ctx = plugin_ctx
        self._task: asyncio.Task | None = None
        self._running = False
        self._engine_unbound_logged = False

    def _engine_ready(self) -> bool:
        engine = getattr(self._engine_proxy, "get_engine", lambda: None)()
        return engine is not None

    @property
    def _adapter(self) -> Any:
        """动态获取 adapter（NapCat 连接后才注入到 ctx）。"""
        return getattr(self._plugin_ctx, "adapter", None)

    def enqueue(self, issue_number: int, repo: str, title: str, body: str, labels: list[str]) -> str:
        import uuid
        task_id = uuid.uuid4().hex[:12]
        state = IssueState(
            issue_number=issue_number, repo=repo, title=title, body=body,
            labels=labels, task_id=task_id,
        )
        state.add_conversation("user", f"Issue #{issue_number}: {title}\n\n{body}")
        self._store.set(f"{_PREFIX}{task_id}", state.to_dict())
        logger.info("Tracker: Issue #%d (%s) 入队, task_id=%s, status=%s, labels=%s",
                     issue_number, repo, task_id, state.status, state.labels)
        return task_id

    def get_state(self, task_id: str) -> IssueState | None:
        raw = self._store.get(f"{_PREFIX}{task_id}")
        if raw is None:
            return None
        return IssueState.from_dict(raw if isinstance(raw, dict) else {})

    def _save(self, state: IssueState) -> None:
        self._store.set(f"{_PREFIX}{state.task_id}", state.to_dict())

    def list_active(self) -> list[IssueState]:
        result: list[IssueState] = []
        all_data = self._store.all() if hasattr(self._store, "all") else {}
        for key, raw in all_data.items():
            if not key.startswith(_PREFIX):
                continue
            data = raw if isinstance(raw, dict) else {}
            state = IssueState.from_dict(data)
            if state.status in ("GATHERING_INFO", "AWAITING_RESPONSE", "APPROVED"):
                result.append(state)
        return result

    # ── 后台循环 ──

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        engine_ready = self._engine_ready()
        logger.info("Tracker: 后台循环启动 (间隔%ds, 引擎=%s, adapter=%s, active_repos=%s)",
                     _TRACKER_TICK, "已绑定" if engine_ready else "待绑定",
                     "已注入" if self._adapter else "待注入",
                     self._config.get("active_repos", []))

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Tracker: 后台循环已停止")

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except Exception:
                logger.exception("Tracker: tick 异常")
            await asyncio.sleep(_TRACKER_TICK)

    async def _tick(self) -> None:
        if not self._engine_ready():
            if not self._engine_unbound_logged:
                logger.info("Tracker: 引擎未绑定，等待首个命令触发绑定后开始工作")
                self._engine_unbound_logged = True
            return

        self._engine_unbound_logged = False
        active = self.list_active()
        if active:
            logger.debug("Tracker: tick - %d 个活跃 Issue", len(active))
        for state in active:
            try:
                await self._process(state)
            except Exception:
                logger.exception("Tracker: 处理 Issue #%d 异常", state.issue_number)

    async def _process(self, state: IssueState) -> None:
        logger.debug("Tracker: 处理 Issue #%d status=%s q=%d conv=%d",
                     state.issue_number, state.status, state.questions_asked, len(state.conversation))

        # 终态：不做任何处理（防御性检查，list_active 已过滤但以防 data_store 被外部修改）
        if state.status in ("CLOSED", "FIXING", "DONE", "ABORTED"):
            return

        # 首次处理时拉取仓库上下文（README + 顶层目录结构）
        if not state.repo_context:
            await self._load_repo_context(state)

        # 1. 拉取新评论
        await self._fetch_new_comments(state)

        # 2. GATHERING_INFO → 分析是否就绪
        if state.status == "GATHERING_INFO":
            await self._try_gather(state)

        # 3. AWAITING_RESPONSE → 等待用户回复后重新分析
        elif state.status == "AWAITING_RESPONSE":
            since_last = time.time() - state.last_activity
            if since_last > 120:
                logger.warning("Tracker: Issue #%d 等待回复超时 %.0fs，重新分析", state.issue_number, since_last)
                await self._try_gather(state)
            else:
                logger.debug("Tracker: Issue #%d 等待回复中 (%.0fs/120s)",
                             state.issue_number, since_last)

    async def _load_repo_context(self, state: IssueState) -> None:
        """拉取仓库 README 和顶层目录结构，注入到 state.repo_context。"""
        logger.info("Tracker: Issue #%d 首次拉取仓库 %s 上下文", state.issue_number, state.repo)
        try:
            readme_text = ""
            tree_items: list[dict[str, Any]] = []
            from .api import get_readme, get_repo_file_tree
            readme_text, tree_items = await asyncio.gather(
                get_readme(state.repo, self._config),
                get_repo_file_tree(state.repo, self._config),
            )
            lines: list[str] = []
            if readme_text:
                lines.append(f"README 摘要（前2000字）:\n{readme_text[:2000]}")
            if tree_items:
                file_list = []
                for it in tree_items:
                    itype = "dir/" if it.get("type") == "dir" else ""
                    file_list.append(f"  {itype}{it.get('name', '?')}")
                lines.append(f"顶层目录结构:\n" + "\n".join(file_list[:40]))
            state.repo_context = "\n\n".join(lines)
            self._save(state)
            logger.info("Tracker: Issue #%d 仓库上下文已加载 (%d 字符)",
                         state.issue_number, len(state.repo_context))
        except Exception:
            state.repo_context = "（仓库上下文加载失败）"
            self._save(state)
            logger.exception("Tracker: Issue #%d 加载仓库上下文失败", state.issue_number)

    async def _fetch_new_comments(self, state: IssueState) -> None:
        import datetime
        since = None
        if state.last_comment_fetched_at > 0:
            since = datetime.datetime.fromtimestamp(
                state.last_comment_fetched_at, tz=datetime.timezone.utc
            ).isoformat()
        comments = await get_issue_comments(state.repo, state.issue_number, self._config, since=since)
        state.last_comment_fetched_at = time.time()
        had_new = False

        for c in comments:
            user_login = c.get("user", {}).get("login", "unknown")
            body = c.get("body", "")
            if not body:
                continue
            existing_bodies = {m["content"] for m in state.conversation if m["role"] == "assistant"}
            if body in existing_bodies:
                continue
            state.add_conversation("user", f"@{user_login}: {body}")
            had_new = True
            logger.info("Tracker: Issue #%d 新评论 @%s (共%d条对话)",
                         state.issue_number, user_login, len(state.conversation))

        # 有新评论时尝试调整标签
        if had_new and self._config.get("auto_label", True):
            logger.debug("Tracker: Issue #%d 有新评论，尝试调整标签", state.issue_number)
            try:
                from .labeler import adjust_labels_for_issue
                new_labels = await adjust_labels_for_issue(
                    state.issue_number, state.repo, state.title, state.conversation,
                    state.labels, self._config, self._engine_proxy,
                )
                if new_labels is not None and set(new_labels) != set(state.labels):
                    from .api import add_labels_to_issue, remove_label_from_issue
                    to_add = [l for l in new_labels if l not in state.labels]
                    to_remove = [l for l in state.labels if l not in new_labels]
                    remove_ok = True
                    for label in to_remove:
                        ok = await remove_label_from_issue(state.repo, state.issue_number, label, self._config)
                        if not ok:
                            logger.warning("Tracker: Issue #%d 移除标签 %s 失败（HTTP非200）", state.issue_number, label)
                            remove_ok = False
                        else:
                            logger.debug("Tracker: Issue #%d 移除标签 %s", state.issue_number, label)
                    add_ok = True
                    if to_add:
                        add_ok = await add_labels_to_issue(state.repo, state.issue_number, to_add, self._config)
                        if not add_ok:
                            logger.warning("Tracker: Issue #%d 添加标签 %s 失败（HTTP非200）", state.issue_number, to_add)
                    if remove_ok and add_ok:
                        state.labels = new_labels
                    logger.info("Tracker: Issue #%d 标签变更: +%s -%s (移除=%s 添加=%s)",
                                state.issue_number, to_add, to_remove,
                                "ok" if remove_ok else "失败",
                                "ok" if add_ok else "失败")
            except Exception as exc:
                logger.debug("Tracker: Issue #%d 标签调整失败: %s", state.issue_number, exc)

        self._save(state)

    async def _try_gather(self, state: IssueState) -> None:
        max_q = self._config.get("max_questions", 12)
        code_context: dict[str, str] = {}
        fetched_files: set[str] = set()

        logger.info("Tracker: Issue #%d 开始信息收集 (q=%d/%d, conv=%d)",
                     state.issue_number, state.questions_asked, max_q, len(state.conversation))

        # 多轮信息收集：最多 2 轮代码查看 + 1 轮最终决策
        for round_num in range(1, 4):
            result = await analyze_and_gather(state, self._engine_proxy, code_context or None, model=self._config.get("model", ""))

            # 请求查看文件 → 获取后重新分析
            look_at = result.get("look_at_files", [])
            if look_at and round_num < 3:
                new_files = [f for f in look_at if f not in fetched_files]
                if new_files:
                    logger.info("Tracker: Issue #%d 请求查看文件: %s", state.issue_number, new_files)
                    for file_path in new_files[:3]:
                        content = await get_file_content(state.repo, file_path, config=self._config)
                        if content:
                            code_context[file_path] = content
                            fetched_files.add(file_path)
                            logger.info("Tracker: Issue #%d 已获取 %s (%d 字符)",
                                         state.issue_number, file_path, len(content))
                    continue  # 重新分析（带新代码上下文）

            action = result.get("action", "ask")
            logger.info("Tracker: Issue #%d 决策 action=%s round=%d understanding=%s",
                         state.issue_number, action, round_num,
                         result.get("understanding", "")[:80])

            # 关单
            if action == "close":
                close_reason = result.get("close_reason", "经分析此 Issue 无需继续跟进")
                logger.info("Tracker: Issue #%d 判定为应关闭: %s", state.issue_number, close_reason)
                await self._close_issue(state, close_reason)
                return

            # 就绪
            if action == "ready" or state.questions_asked >= max_q:
                state.status = "READY"
                state.gathered_summary = result.get("understanding", state.title)
                self._save(state)
                logger.info("Tracker: Issue #%d 信息就绪 (action=%s), 发送总结并通知 developer",
                             state.issue_number, action)
                await self._post_summary_and_notify(state, result)
                return

            # 追问 — 人格化后再发布（先检查 Issue 是否已被外部关闭）
            question = result.get("question", "")
            if question:
                logger.debug("Tracker: Issue #%d 准备追问 (q%d): %s",
                             state.issue_number, state.questions_asked + 1, question[:80])
                from .api import is_issue_closed, post_issue_comment
                if await is_issue_closed(state.repo, state.issue_number, self._config):
                    logger.info("Tracker: Issue #%d 已被外部关闭，跳过追问", state.issue_number)
                    state.status = "CLOSED"
                    self._save(state)
                    return
                persona_question = await self._persona_question(state, question)
                await post_issue_comment(state.repo, state.issue_number, persona_question, self._config)
                state.add_conversation("assistant", persona_question)
                state.questions_asked += 1
                state.status = "AWAITING_RESPONSE"
                self._save(state)
                logger.info("Tracker: Issue #%d 已追问 (q%d): %s",
                            state.issue_number, state.questions_asked, persona_question[:80])
            else:
                logger.warning("Tracker: Issue #%d action=ask 但无追问内容，跳过", state.issue_number)
            return

    async def _close_issue(self, state: IssueState, reason: str) -> None:
        from .api import close_issue as api_close_issue, is_issue_closed
        from .closer import _generate_close_comment

        # 检查是否已被外部关闭
        if await is_issue_closed(state.repo, state.issue_number, self._config):
            logger.info("Tracker: Issue #%d 已被外部关闭，跳过重复关闭", state.issue_number)
            state.status = "CLOSED"
            self._save(state)
            return

        logger.debug("Tracker: Issue #%d 生成关闭评论...", state.issue_number)
        close_msg = await _generate_close_comment(
            {"number": state.issue_number, "title": state.title, "body": state.body},
            state.repo, self._engine_proxy, reason,
        )
        await api_close_issue(state.repo, state.issue_number, close_msg, self._config)
        state.status = "CLOSED"
        self._save(state)
        logger.info("Tracker: Issue #%d 已关闭 (reason=%s)", state.issue_number, reason)

        admin_id = self._config.get("admin_user_id", "")
        if not admin_id:
            from .webhook import _resolve_admin_id
            admin_id = _resolve_admin_id(self._adapter)
        if admin_id and self._adapter:
            try:
                prompt = f"""以你的角色身份通知管理员一个 Issue 已自动关闭。

Issue #{state.issue_number}: {state.title}
原因: {reason}

用角色口吻简述，1句话。"""
                persona_msg = await self._engine_proxy.generate_raw(prompt, inject_persona=True, model=self._config.get("model", ""))
                msg = persona_msg.strip() or f"Issue #{state.issue_number} 已自动关闭: {reason}"
            except Exception:
                msg = f"Issue #{state.issue_number}: {state.title} 已自动关闭\n仓库: {state.repo}\n原因: {reason}"
            await self._adapter.send_private_message(admin_id, msg)

    async def _persona_question(self, state: IssueState, base_question: str) -> str:
        """将功能性追问转为角色化表达（保留核心问题，避免答非所问）。"""
        try:
            prompt = f"""你正在 GitHub Issue 下以你的角色身份追问用户一个问题。你必须确保最终输出中明确包含核心问题的实质内容。

Issue #{state.issue_number}: {state.title}

核心问题（必须回答）: {base_question}

要求：
1. 用角色口吻开场（1句轻快问候即可）
2. 然后自然地引出核心问题，不得改变问题的实质含义
3. 整体保持友好、自然，50-120字
4. 禁止仅输出角色扮演内容而不包含核心问题"""
            result = await self._engine_proxy.generate_raw(prompt, inject_persona=True, model=self._config.get("model", ""))
            return result.strip() or base_question
        except Exception:
            return base_question

    async def _post_summary_and_notify(self, state: IssueState, result: dict[str, Any]) -> None:
        """信息就绪后：在 Issue 下发一条总结评论，然后直接启动自动修复（跳过管理员审批）。"""
        from .api import is_issue_closed, post_issue_comment

        # 检查 Issue 是否已被外部关闭
        if await is_issue_closed(state.repo, state.issue_number, self._config):
            logger.info("Tracker: Issue #%d 已被外部关闭，跳过总结", state.issue_number)
            state.status = "CLOSED"
            self._save(state)
            return

        understanding = result.get("understanding", state.title)
        approach = result.get("approach", "待分析")

        # 1. 在 Issue 下发总结评论（人格化）
        try:
            summary_prompt = f"""你刚完成了对一个 Issue 的信息收集，现在需要在 Issue 下发一条总结评论。

Issue #{state.issue_number}: {state.title}
你对问题的理解: {understanding}
提议的修复方案: {approach}
总共追问了 {state.questions_asked} 轮

请用你的角色口吻写一条信息收集完成的总结，包含：
1. 对问题核心的简要重述
2. 对提供信息者的感谢
3. 告知将直接启动自动修复（无需等待审批）
控制在 80-120 字。只输出正文。"""
            summary_text = await self._engine_proxy.generate_raw(
                summary_prompt, inject_persona=True,
                model=self._config.get("model", ""),
            )
        except Exception:
            summary_text = f"信息收集已完成，直接启动自动修复喵~"

        await post_issue_comment(state.repo, state.issue_number, summary_text.strip(), self._config)
        state.add_conversation("assistant", summary_text.strip())
        self._save(state)
        logger.info("Tracker: Issue #%d 总结已发布到 Issue", state.issue_number)

        # 2. 直接启动自动修复（跳过管理员审批）
        state.status = "FIXING"
        self._save(state)
        logger.info("Tracker: Issue #%d 跳过审批，直接启动自动修复", state.issue_number)

        from .agent_loop import run_agent_loop
        task_data = {
            "task_id": state.task_id,
            "repo": state.repo,
            "issue_number": state.issue_number,
            "issue_title": state.title,
            "issue_body": state.body,
            "labels": state.labels,
            "status": "APPROVED",
        }
        asyncio.create_task(
            run_agent_loop(
                task_data=task_data,
                config=self._config,
                engine_proxy=self._engine_proxy,
                adapter=self._adapter,
            )
        )

        # 3. 通知管理员已自动启动修复
        admin_id = self._config.get("admin_user_id", "")
        if not admin_id:
            from .webhook import _resolve_admin_id
            admin_id = _resolve_admin_id(self._adapter)
        if admin_id and self._adapter:
            try:
                prompt = f"""以你的角色身份向项目管理员发送通知。

Issue #{state.issue_number}: {state.title}
仓库: {state.repo}
理解: {understanding}
修复方案: {approach}
任务ID: {state.task_id}
已追问: {state.questions_asked} 轮

用你的角色口吻告诉管理员:
1. 信息收集已完成
2. 总结已发布在 Issue 下
3. 已跳过审批，直接启动了自动修复
控制在 1-2 句话。"""
                persona_msg = await self._engine_proxy.generate_raw(
                    prompt, inject_persona=True,
                    model=self._config.get("model", ""),
                )
                msg = persona_msg.strip() or f"#{state.issue_number} 信息已就绪，已自动启动修复喵~"
            except Exception:
                msg = f"#{state.issue_number}: {state.title}\n仓库: {state.repo}\n已跳过审批，自动启动修复喵~"

            await self._adapter.send_private_message(admin_id, msg)
            logger.info("Tracker: Issue #%d 已通知管理员自动修复已启动", state.issue_number)

    async def _notify_developer(self, state: IssueState, result: dict[str, Any]) -> None:
        """信息就绪后直接启动自动修复（跳过管理员审批）。"""
        state.status = "FIXING"
        self._save(state)
        logger.info("Tracker: Issue #%d 跳过审批，直接启动自动修复 (from _notify_developer)", state.issue_number)

        from .agent_loop import run_agent_loop
        task_data = {
            "task_id": state.task_id,
            "repo": state.repo,
            "issue_number": state.issue_number,
            "issue_title": state.title,
            "issue_body": state.body,
            "labels": state.labels,
            "status": "APPROVED",
        }
        asyncio.create_task(
            run_agent_loop(
                task_data=task_data,
                config=self._config,
                engine_proxy=self._engine_proxy,
                adapter=self._adapter,
            )
        )

        from .webhook import _resolve_admin_id
        admin_id = _resolve_admin_id(self._adapter)
        if not admin_id or not self._adapter:
            logger.warning("Tracker: Issue #%d 已自动启动修复，但无法通知管理员", state.issue_number)
            return
        approach = result.get("approach", "待分析")
        try:
            prompt = f"""以你的角色身份向项目管理员发送一条简短通知。

Issue #{state.issue_number}: {state.title}
理解: {state.gathered_summary}
修复方案: {approach}
任务ID: {state.task_id}

用你的角色口吻告诉管理员这个 Issue 已就绪，已经自动启动修复了喵~ 1-2句话。"""
            persona_msg = await self._engine_proxy.generate_raw(prompt, inject_persona=True, model=self._config.get("model", ""))
            msg = persona_msg.strip() or f"[READY] Issue #{state.issue_number} 信息已就绪，已自动启动修复喵~"
        except Exception:
            msg = f"[READY] Issue #{state.issue_number}: {state.title}\n仓库: {state.repo}\n已跳过审批，自动启动修复喵~"
        await self._adapter.send_private_message(admin_id, msg)
        logger.info("Tracker: Issue #%d 已通知管理员自动修复已启动 (admin=%s)", state.issue_number, admin_id)
