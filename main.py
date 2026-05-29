from __future__ import annotations

import asyncio
import io
import logging
import sys
import traceback
from typing import Any

from sirius_pulse.config.config_builder import ConfigBuilder
from sirius_pulse.github.event_bridge import (
    register_comment_handler,
    register_issue_handler,
    register_pr_handler,
    set_coding_bot_login,
    set_issue_repos,
)
from sirius_pulse.core.brain import Brain, ChatRequest, ChatResult
from sirius_pulse.plugins.api import PluginBase, command, PluginResponse

from .commands import handle_gh_command
from .config import GithubAgentConfig
from .monitor_config import MonitorConfig
from .tracker import IssueTracker
from .webhook import handle_issue_opened, handle_pr_event, set_tracker

logger = logging.getLogger(__name__)

_plugin_dependencies = ["httpx", "GitPython", "pytest", "pytest-asyncio", "pytest-cov", "pytest-xdist"]

# 使用 ConfigBuilder 定义插件参数
_config = ConfigBuilder()
_config.group("GitHub 认证").add(
    "github_write_token",
    type="password",
    description="GitHub PAT（fork/PR/标签/评论），留空复用 monitor token",
)
_config.group("GitHub 认证").add(
    "github_username",
    type="str",
    description="GitHub 用户名（git 提交者身份，留空=仓库 owner）",
)
_config.group("GitHub 认证").add(
    "github_email",
    type="str",
    description="GitHub 邮箱（git 提交者 email，留空=username@users.noreply.github.com）",
)
_config.group("仓库设置").add(
    "active_repos",
    type="list",
    description="生效仓库（owner/repo，留空=monitor全部）",
)
_config.group("Agent 设置").add(
    "model",
    type="model",
    description="自定义 LLM 模型名",
)
_config.group("Agent 设置").add(
    "max_retries",
    type="int",
    description="最大重试次数",
    default=3,
)
_config.group("Agent 设置").add(
    "max_questions",
    type="int",
    description="信息收集最大追问次数",
    default=12,
)
_config.group("Agent 设置").add(
    "test_command",
    type="str",
    description="测试命令",
    default="pytest",
)
_config.group("Agent 设置").add(
    "lint_command",
    type="str",
    description="静态检查命令（留空跳过，如 flake8 .）",
    default="",
)
_config.group("功能开关").add(
    "auto_label",
    type="boolean",
    description="启用 Issue 自动标签",
    default=True,
)
_config.group("功能开关").add(
    "auto_review",
    type="boolean",
    description="启用 PR 自动审阅",
    default=True,
)
_config.group("功能开关").add(
    "auto_close_garbage",
    type="boolean",
    description="自动关闭垃圾 Issue/PR",
    default=True,
)
_config.group("功能开关").add(
    "review_mode",
    type="str",
    description="PR 审阅深度: quick|deep",
    default="quick",
    choices=["quick", "deep"],
)
_config.group("控制台").add(
    "console_viewer_enabled",
    type="boolean",
    description="弹出实时控制台窗口",
    default=True,
)
_config.group("控制台").add(
    "console_viewer_keep_open",
    type="boolean",
    description="修复完成后保持窗口打开",
    default=False,
)
_config.group("日志归档").add(
    "log_archive_enabled",
    type="boolean",
    description="启用工作流日志归档",
    default=True,
)
_config.group("日志归档").add(
    "log_archive_max_count",
    type="int",
    description="归档日志最大保留数",
    default=50,
)


class CodingAgentPlugin(PluginBase):
    _plugin_name = "coding_agent"
    _plugin_display_name = "编码助手"
    _plugin_description = "GitHub Issue/PR 自动化管理 + Python 代码执行（事件由 github_monitor 驱动，Issue 信息后台自主收集）"
    _plugin_version = "2.2.0"
    _plugin_author = "Sparrived/Sirius"

    _plugin_parameters = _config.build()
    _plugin_permissions = {
        "developer_only": True,
        "hidden_from_intent": True,
    }
    _plugin_prompt_inject = (
        "编码助手：我可以在后台管理 GitHub Issue 和 PR，包括代码修复、"
        "自动化测试、PR 审阅等，也支持直接执行 Python 代码片段"
    )

    def __init__(self) -> None:
        super().__init__()
        self._gh_config: GithubAgentConfig | None = None
        self._monitor: MonitorConfig = MonitorConfig()
        self._effective_repos: list[str] = []
        self._tracker: IssueTracker | None = None
        self._brain_hooks_registered: bool = False

    async def on_load(self) -> None:
        self._gh_config = GithubAgentConfig.from_dict(self.ctx.config)

        if self.ctx.data_store:
            self._monitor = MonitorConfig.load(self.ctx.data_store)

        if not self._monitor.repo_names:
            logger.info("github_monitor 中未配置任何仓库，等待配置后重载")
            return

        active = self._gh_config.active_repos
        if active:
            active_set = set(active)
            self._effective_repos = [r for r in self._monitor.repo_names if r in active_set]
            logger.info("生效仓库过滤: %d/%d (%s)", len(self._effective_repos),
                        len(self._monitor.repo_names), ", ".join(self._effective_repos) if self._effective_repos else "无")
        else:
            self._effective_repos = list(self._monitor.repo_names)

        if not self._effective_repos:
            logger.info("active_repos 过滤后无生效仓库，跳过")
            return

        set_issue_repos(set(self._effective_repos))
        if self._gh_config.github_username:
            set_coding_bot_login(self._gh_config.github_username)
            logger.info("coding bot login 已设置: %s", self._gh_config.github_username)

        config_dict = self._build_config_dict()

        # 初始化 IssueTracker（后台信息收集循环）
        self._tracker = IssueTracker(
            data_store=self.ctx.data_store,
            config=config_dict,
            engine_proxy=self.ctx.engine,
            plugin_ctx=self.ctx,
        )
        set_tracker(self._tracker)
        await self._tracker.start()

        # 注册到 event_bridge
        async def _on_issue_opened(body: dict[str, Any], repo_name: str) -> None:
            if repo_name not in self._effective_repos:
                return
            await handle_issue_opened(body, config_dict, self.ctx.adapter,
                                       self.ctx.engine, self.ctx.data_store)

        async def _on_pr_event(body: dict[str, Any], repo_name: str, action: str) -> None:
            if repo_name not in self._effective_repos:
                return
            asyncio.create_task(
                handle_pr_event(body, config_dict, self.ctx.adapter, self.ctx.engine)
            )

        register_issue_handler(_on_issue_opened)
        register_pr_handler(_on_pr_event)

        # Issue 评论处理器：已有 tracker 则回灌评论，无 tracker 则新建
        # 注意：GitHub issue_comment 事件对 Issue 和 PR 都触发，需过滤 PR 评论
        async def _on_issue_comment(body: dict[str, Any], repo_name: str) -> None:
            if repo_name not in self._effective_repos:
                return
            issue = body.get("issue", {})
            comment = body.get("comment", {})
            if not issue or not comment:
                return
            # 跳过 PR 评论（PR 有独立的 review/comment 流程）
            if issue.get("pull_request"):
                return
            issue_number = issue.get("number", 0)
            if not issue_number:
                return
            comment_body = comment.get("body", "")
            if not comment_body:
                return

            # 查找已有 tracker
            all_data = self.ctx.data_store.all() if hasattr(self.ctx.data_store, "all") else {}
            from .tracker import _PREFIX
            for key, raw in all_data.items():
                if not key.startswith(_PREFIX):
                    continue
                data = raw if isinstance(raw, dict) else {}
                if data.get("repo") == repo_name and data.get("issue_number") == issue_number:
                    status = data.get("status", "")
                    # 已终态：不再处理
                    if status in ("CLOSED", "FIXING", "DONE", "ABORTED"):
                        logger.debug("回灌评论跳过: Issue #%d 已处于终态 %s", issue_number, status)
                        return
                    # 跳过 AI 自己的评论（已在 conversation 中存在）
                    assistant_bodies = {
                        m["content"] for m in data.get("conversation", [])
                        if m.get("role") == "assistant"
                    }
                    if comment_body.strip() in assistant_bodies:
                        logger.debug("回灌评论跳过: Issue #%d 这条是 AI 自己的评论", issue_number)
                        return
                    # 注入用户评论
                    user_login = comment.get("user", {}).get("login", "unknown")
                    data.setdefault("conversation", []).append({
                        "role": "user", "content": f"@{user_login}: {comment_body}",
                        "timestamp": __import__("time").time(),
                    })
                    data["last_activity"] = __import__("time").time()
                    # 仅当当前在等待回复时才切到信息收集状态
                    if status == "AWAITING_RESPONSE":
                        data["status"] = "GATHERING_INFO"
                    self.ctx.data_store.set(key, data)
                    logger.info("回灌评论到 tracker: Issue #%d @%s", issue_number, user_login)
                    return
            # 无 tracker → 新建
            self._tracker.enqueue(
                issue_number=issue_number,
                repo=repo_name,
                title=issue.get("title", "无标题"),
                body=issue.get("body", ""),
                labels=[l.get("name", "") for l in (issue.get("labels", []) or [])],
            )
            logger.info("为已有 Issue #%d 新建 tracker（来自评论事件）", issue_number)

        register_comment_handler(_on_issue_comment)

        self._register_brain_hooks()

        logger.info("coding_agent v2.2 启动完成 (monitor_repos=%d, effective=%d, tracker=on, "
                    "auto_label=%s, auto_review=%s, auto_close=%s, max_q=%d)",
                    len(self._monitor.repo_names), len(self._effective_repos),
                    self._gh_config.auto_label, self._gh_config.auto_review,
                    self._gh_config.auto_close_garbage, self._gh_config.max_questions)

    async def on_unload(self) -> None:
        if self._tracker:
            await self._tracker.stop()

    def _build_config_dict(self) -> dict[str, Any]:
        if self._gh_config is None:
            return {}
        return {
            "repos": self._effective_repos,
            "_monitor": self._monitor,
            "active_repos": self._effective_repos,
            "github_write_token": self._gh_config.github_write_token,
            "github_username": self._gh_config.github_username,
            "github_email": self._gh_config.github_email,
            "admin_user_id": self._resolve_admin_id(),
            "model": self._gh_config.model,
            "lint_command": self._gh_config.lint_command,
            "max_questions": self._gh_config.max_questions,
            "webhook_secret": self._monitor.webhook_secret,
            "auto_label": self._gh_config.auto_label,
            "auto_review": self._gh_config.auto_review,
            "auto_close_garbage": self._gh_config.auto_close_garbage,
            "review_mode": self._gh_config.review_mode,
            "workspace_dir": str(self._gh_config.workspace_dir),
            "console_viewer_enabled": self._gh_config.console_viewer_enabled,
            "console_viewer_keep_open": self._gh_config.console_viewer_keep_open,
        }

    def _resolve_admin_id(self) -> str:
        adapter = getattr(self.ctx, "adapter", None)
        if adapter is None:
            return ""
        plugin_config = getattr(adapter, "plugin_config", None)
        if isinstance(plugin_config, dict):
            return str(plugin_config.get("root", "")).strip()
        return ""

    # ── Brain Hooks ─────────────────────────────────────────────────

    def _register_brain_hooks(self) -> None:
        """注册 Brain PreHook / PostHook。

        PreHook：在 LLM 生成前注入编码助手能力描述与当前活跃任务状态。
        PostHook：在 LLM 生成后监控 AI 回复中的 Issue/PR 引用。
        """
        if self._brain_hooks_registered:
            return

        engine = self.ctx.engine.get_engine()
        brain: Brain | None = getattr(engine, "brain", None)
        if brain is None:
            logger.warning("Brain 不可用，跳过 PreHook/PostHook 注册")
            return

        plugin_ref = self

        def _pre_hook(_brain: Brain, request: ChatRequest, _ctx: dict) -> None:
            """PreHook：注入当前活跃的 Issue 任务状态。"""
            active_context = plugin_ref._get_active_task_context()
            if active_context:
                request.system_prompt = request.system_prompt.rstrip() + "\n\n" + active_context

        def _post_hook(_brain: Brain, _request: ChatRequest, result: ChatResult, _ctx: dict) -> None:
            """PostHook：监控 AI 回复中是否涉及 Issue/PR。"""
            plugin_ref._on_ai_response(result)

        brain.register_pre_hook(_pre_hook, priority=0)
        brain.register_post_hook(_post_hook, priority=100)
        self._brain_hooks_registered = True

        logger.info("编码助手 Brain PreHook/PostHook 已注册")

    def _get_active_task_context(self) -> str:
        """获取当前活跃的 Issue 任务状态文本，供 PreHook 注入到 LLM 上下文。"""
        if not self._tracker:
            return ""
        try:
            active = self._tracker.list_active()
        except Exception:
            return ""
        if not active:
            return ""
        lines = ["【当前活跃的 GitHub 任务】"]
        for state in active[:3]:
            lines.append(f"- Issue #{state.issue_number} ({state.repo}): {state.title[:50]} [{state.status}]")
        return "\n".join(lines)

    def _on_ai_response(self, result: ChatResult) -> None:
        """PostHook 回调：在 LLM 生成后检查回复中是否涉及 Issue/PR 引用。"""
        import re
        issues_found = re.findall(r'#(\d+)', result.clean_text)
        if issues_found:
            logger.debug("AI 回复中提及 Issue/PR: %s", issues_found)

    @command(
        "py",
        prefix="/",
        patterns=["py", "python", "python3"],
        pattern_type="keyword",
        render_mode="direct",
        description="执行一行 Python 代码并返回结果",
        examples=["/py print('Hello World')", "/py 1+1"],
        hidden_from_intent=True,
    )
    async def execute_python(self, code: str) -> PluginResponse:
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()
        try:
            exec(code)
            output = captured.getvalue().strip()
            return PluginResponse.ok(text=output or "代码执行完成（无输出）")
        except Exception:
            error = traceback.format_exc(limit=0).strip()
            return PluginResponse.fail(f"执行出错:\n{error}")
        finally:
            sys.stdout = old_stdout

    @command(
        "gh",
        prefix="/",
        patterns=["/gh"],
        render_mode="direct",
        description="GitHub Agent 指令：管理 Issue 修复、PR 审阅",
        examples=["/gh <task_id> auto", "/gh review <repo_index> <pr_number> [quick|deep]"],
        hidden_from_intent=True,
    )
    async def github_agent(self, command_args: str = "") -> PluginResponse:
        if not self._monitor.repo_names:
            return PluginResponse.fail("未在 github_monitor 中配置仓库，请在 WebUI 的 SKILL 设置中配置")

        config_dict = {
            **self._build_config_dict(),
            "max_retries": self._gh_config.max_retries if self._gh_config else 3,
            "max_questions": self._gh_config.max_questions if self._gh_config else 12,
            "test_command": self._gh_config.test_command if self._gh_config else "pytest",
            "lint_command": self._gh_config.lint_command if self._gh_config else "",
        }

        result = await handle_gh_command(
            ctx=self.ctx,
            command_args=command_args,
            config=config_dict,
            engine_proxy=self.ctx.engine,
            data_store=self.ctx.data_store,
        )

        if result.startswith("权限不足"):
            return PluginResponse.fail(result)
        return PluginResponse.ok(text=result)
