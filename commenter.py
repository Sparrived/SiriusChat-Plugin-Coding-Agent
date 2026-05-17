from __future__ import annotations

import logging
from typing import Any

from .api import post_issue_comment

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


async def generate_issue_comment(
    issue_data: dict,
    labels: list[str],
    repo_full_name: str,
    engine_proxy: Any,
    config: dict | None = None,
) -> str:
    """使用 LLM 生成 Issue 智能回复评论。

    失败时自动重试（最多 3 次），不降级到模板回复。
    人格属性由 generate_raw(inject_persona=True) 自动注入。
    """
    label_display = " ".join(f"`{l}`" for l in labels) if labels else "（待人工分类）"

    prompt = f"""你正在回复一个 GitHub Issue。回复内容将在 Issue 下公开发表，面向 Issue 提交者。请以你的角色身份和沟通风格撰写回复。

Issue #{issue_data.get('number', '?')}: {issue_data.get('title', '')}

Issue 内容:
{issue_data.get('body', '')[:3000]}

已自动分析并应用的标签: {label_display}

回复要求：
1. 开头感谢用户提交 Issue，语气温馨友好，展现角色个性
2. 简要复述你理解的问题
3. 如果 issue 描述不够清晰，提出 1-2 个追问帮助澄清
4. 如果 issue 包含了复现步骤/错误日志，肯定用户的详细描述
5. 结尾告知后续流程：标签已自动分析，管理员将评估是否启动自动修复
6. 长度控制在 100-200 字，不要过长
7. 输出纯文本（Markdown 格式，但不要代码块包裹）

请直接输出评论正文，不要包含任何前缀说明。"""

    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = await engine_proxy.generate_raw(prompt, inject_persona=True)
            return result.strip()
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                logger.info(
                    "Issue #%d 智能回复第 %d/%d 次失败，重试中: %s",
                    issue_data.get("number", "?"), attempt, _MAX_RETRIES, exc,
                )
            else:
                logger.error(
                    "Issue #%d 智能回复 %d 次重试全部失败: %s",
                    issue_data.get("number", "?"), _MAX_RETRIES, exc,
                )

    raise RuntimeError(
        f"Issue #{issue_data.get('number', '?')} 智能回复生成失败（{_MAX_RETRIES}次重试）"
    ) from last_error


async def post_comment(
    repo_full_name: str,
    issue_number: int,
    comment_body: str,
    config: dict,
) -> bool:
    """在 Issue 下发表评论。"""
    result = await post_issue_comment(repo_full_name, issue_number, comment_body, config)
    if result:
        logger.info("Issue #%d 智能回复已发表", issue_number)
    else:
        logger.error("Issue #%d 智能回复发表失败", issue_number)
    return result
