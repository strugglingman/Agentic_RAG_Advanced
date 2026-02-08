"""Slack Response Formatter - Formats bot responses into Slack Block Kit format."""

import json
import logging
from src.adapters.base_bot_adapter import BotResponse

logger = logging.getLogger(__name__)


class SlackFormatter:
    """Formats BotResponse into Slack Block Kit format."""

    def format_response(self, response: BotResponse) -> dict:
        """Format BotResponse to Slack blocks. Files are uploaded separately by adapter."""
        if response.error:
            return self.format_error(response.error)

        answer_text = response.text or ""
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": answer_text},
            }
        ]

        # Add citations if contexts exist
        if response.contexts:
            context_elements = self.format_contexts_as_citations(response.contexts)
            if context_elements:
                blocks.append({"type": "divider"})
                blocks.append({"type": "context", "elements": context_elements})

        return {
            "blocks": blocks,
            "text": answer_text[:150],  # Fallback for notifications
        }

    def format_error(self, error_message: str) -> dict:
        """Format error message with warning style."""
        return {
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":x: *Error:* {error_message}",
                    },
                }
            ],
            "text": f"Error: {error_message}",
        }

    def format_contexts_as_citations(self, contexts: list[dict]) -> list[dict]:
        """Convert RAG contexts to citation elements. Limit to top 5 sources."""
        if not contexts:
            return []

        # Extract unique filenames from contexts
        filenames = []
        for ctx in contexts[:5]:  # Limit to 5
            filename = ctx.get("filename") or ctx.get("source") or ctx.get("file_name")
            if filename and filename not in filenames:
                filenames.append(filename)

        if not filenames:
            return []

        sources_text = ", ".join(filenames)
        return [
            {"type": "mrkdwn", "text": f":page_facing_up: *Sources:* {sources_text}"}
        ]

    def format_typing_indicator(self) -> dict:
        """Return thinking message to show while waiting for LLM."""
        return {
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": ":hourglass_flowing_sand: Thinking...",
                    },
                }
            ],
            "text": "Thinking...",
        }

    def format_hitl_confirmation(
        self,
        partial_answer: str,
        hitl_data: dict,
        button_value: str,
    ) -> dict:
        """Format HITL confirmation message with Confirm/Cancel buttons.

        Args:
            partial_answer: The partial answer text from completed steps
            hitl_data: HITL interrupt data (action, details, previous_steps)
            button_value: JSON string with state for button callbacks
        """
        action = hitl_data.get("action", "unknown")
        details = hitl_data.get("details", {})

        blocks = []

        # Partial answer
        if partial_answer:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": partial_answer},
            })
            blocks.append({"type": "divider"})

        # Action header
        action_labels = {
            "send_email": ":email: The assistant wants to send an email.",
            "download_file": ":arrow_down: The assistant wants to download a file.",
            "create_document": ":page_facing_up: The assistant wants to create a document.",
        }
        action_text = action_labels.get(action, f":warning: Action: *{action}*")
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Action Confirmation Required*\n{action_text}",
            },
        })

        # Action details
        detail_lines = []
        if details.get("recipient"):
            detail_lines.append(f"*To:* {details['recipient']}")
        if details.get("task"):
            detail_lines.append(f"*Task:* {details['task']}")
        if details.get("available_attachments"):
            atts = ", ".join(details["available_attachments"])
            detail_lines.append(f"*Attachments:* {atts}")
        if detail_lines:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(detail_lines)},
            })

        # Confirm / Cancel buttons
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Confirm"},
                    "style": "primary",
                    "action_id": "hitl_confirm",
                    "value": button_value,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Cancel"},
                    "style": "danger",
                    "action_id": "hitl_cancel",
                    "value": button_value,
                },
            ],
        })

        return {
            "blocks": blocks,
            "text": f"Action confirmation required: {action}",
        }

    def format_hitl_resolved(self, action: str, confirmed: bool) -> dict:
        """Format message to replace buttons after user clicks Confirm/Cancel."""
        if confirmed:
            text = f":white_check_mark: *{action}* confirmed. Processing..."
        else:
            text = f":no_entry_sign: *{action}* cancelled."
        return {
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": text}}
            ],
            "text": text,
        }

    def split_long_message(self, text: str, max_length: int = 3000) -> list[str]:
        """Split long text into chunks, breaking at paragraph boundaries."""
        if len(text) <= max_length:
            return [text]

        chunks = []
        remaining = text

        while remaining:
            if len(remaining) <= max_length:
                chunks.append(remaining)
                break

            # Find a good break point (paragraph or sentence)
            chunk = remaining[:max_length]
            break_point = chunk.rfind("\n\n")  # Paragraph
            if break_point == -1:
                break_point = chunk.rfind(". ")  # Sentence
            if break_point == -1:
                break_point = chunk.rfind(" ")  # Word
            if break_point == -1:
                break_point = max_length  # Force break

            chunks.append(remaining[: break_point + 1].strip())
            remaining = remaining[break_point + 1 :].strip()

        return chunks
