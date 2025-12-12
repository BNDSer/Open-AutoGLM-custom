"""Model client for AI inference using OpenAI-compatible API."""

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


@dataclass
class ModelConfig:
    """Configuration for the AI model."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model_name: str = "autoglm-phone-9b"
    max_tokens: int = 3000
    temperature: float = 0.0
    top_p: float = 0.85
    frequency_penalty: float = 0.2
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Response from the AI model."""

    thinking: str
    action: str
    raw_content: str


class ModelClient:
    """
    Client for interacting with OpenAI-compatible vision-language models.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def request(self, messages: list[dict[str, Any]]) -> ModelResponse:
        """
        Send a request to the model.

        Args:
            messages: List of message dictionaries in OpenAI format.

        Returns:
            ModelResponse containing thinking and action.

        Raises:
            ValueError: If the response cannot be parsed.
        """
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            extra_body=self.config.extra_body,
            stream=False,
        )

        raw_content = response.choices[0].message.content

        # Parse thinking and action from response
        thinking, action = self._parse_response(raw_content)
        
        # Debug: If thinking is empty but raw_content has content before action markers,
        # try to extract thinking from raw_content
        if not thinking and raw_content:
            # Find the first action marker
            action_markers = ["do(action=", "do(", "finish(message=", "finish(", "<answer>"]
            action_pos = len(raw_content)
            for marker in action_markers:
                pos = raw_content.find(marker)
                if pos != -1 and pos < action_pos:
                    action_pos = pos
            
            # If there's content before the action marker, use it as thinking
            if action_pos > 0 and action_pos < len(raw_content):
                thinking_candidate = raw_content[:action_pos].strip()
                # Clean up XML tags but preserve content
                thinking_candidate = thinking_candidate.replace("<think>", "").replace("</think>", "")
                thinking_candidate = thinking_candidate.replace("<think>", "").replace("</think>", "")
                thinking_candidate = thinking_candidate.replace("<think>", "").replace("</think>", "")
                thinking_candidate = thinking_candidate.strip()
                if thinking_candidate:
                    thinking = thinking_candidate

        return ModelResponse(thinking=thinking, action=action, raw_content=raw_content)

    def _parse_response(self, content: str) -> tuple[str, str]:
        """
        Parse the model response into thinking and action parts.

        Parsing rules (in order of priority):
        1. If content contains '<answer>' tag, extract thinking from <think> 
           and action from <answer> tag.
        2. If content contains 'finish(message=', everything before is thinking,
           everything from 'finish(message=' onwards is action.
        3. If content contains 'do(action=', everything before is thinking,
           everything from 'do(action=' onwards is action.
        4. Otherwise, return empty thinking and full content as action.

        Args:
            content: Raw response content.

        Returns:
            Tuple of (thinking, action).
        """
        if not content:
            return "", ""
        
        # Rule 1: Check for XML tag format (<answer> or <think>)
        # This is the most reliable format, so check it first
        if "<answer>" in content:
            parts = content.split("<answer>", 1)
            # Extract thinking from the part before <answer>
            thinking_part = parts[0]
            
            # Remove <think> or <think> tags if present, but preserve the content
            # Check for <think> first (used in system prompts)
            if "<think>" in thinking_part:
                # Extract content between <think> and </think>
                start_tag = "<think>"
                end_tag = "</think>"
                start_idx = thinking_part.find(start_tag)
                if start_idx != -1:
                    start_idx += len(start_tag)
                    end_idx = thinking_part.find(end_tag, start_idx)
                    if end_idx != -1:
                        thinking = thinking_part[start_idx:end_idx].strip()
                    else:
                        # Tag not closed, take everything after the opening tag
                        thinking = thinking_part[start_idx:].strip()
                else:
                    # Opening tag not found, try to clean up
                    thinking = thinking_part.replace("</think>", "").strip()
            elif "<think>" in thinking_part:
                # Also check for <think> tag (alternative format)
                start_tag = "<think>"
                end_tag = "</think>"
                start_idx = thinking_part.find(start_tag)
                if start_idx != -1:
                    start_idx += len(start_tag)
                    end_idx = thinking_part.find(end_tag, start_idx)
                    if end_idx != -1:
                        thinking = thinking_part[start_idx:end_idx].strip()
                    else:
                        thinking = thinking_part[start_idx:].strip()
                else:
                    thinking = thinking_part.replace("</think>", "").strip()
            else:
                # No tags, use the whole part before <answer> as thinking
                # This preserves all text content before the action
                thinking = thinking_part.strip()
            
            # Extract action from <answer> tag
            action_part = parts[1]
            if "</answer>" in action_part:
                action = action_part.split("</answer>", 1)[0].strip()
            else:
                # Tag not closed, use everything after <answer>
                action = action_part.strip()
            
            # Clean up action: remove any remaining XML tags or special markers
            action = action.replace("<|FunctionCallBegin|>", "").replace("<|FunctionCallEnd|>", "").strip()
            
            # If action is empty, try to find do( or finish( in the original content
            if not action:
                if "do(" in content or "do(action=" in content:
                    do_pos = content.find("do(")
                    if do_pos == -1:
                        do_pos = content.find("do(action=")
                    if do_pos != -1:
                        # Extract from do( to the end or to </answer>
                        remaining = content[do_pos:]
                        if "</answer>" in remaining:
                            action = remaining.split("</answer>", 1)[0].strip()
                        else:
                            action = remaining.strip()
                elif "finish(" in content or "finish(message=" in content:
                    finish_pos = content.find("finish(")
                    if finish_pos == -1:
                        finish_pos = content.find("finish(message=")
                    if finish_pos != -1:
                        remaining = content[finish_pos:]
                        if "</answer>" in remaining:
                            action = remaining.split("</answer>", 1)[0].strip()
                        else:
                            action = remaining.strip()
            
            return thinking, action
        
        # Rule 2: Check for finish(message= (before do(action= to prioritize finish)
        if "finish(message=" in content:
            parts = content.split("finish(message=", 1)
            thinking = parts[0].strip()
            # Clean up thinking: remove XML tags but preserve content
            # Try to extract from <think> or <think> tags if present
            if "<think>" in thinking:
                start_tag = "<think>"
                end_tag = "</think>"
                start_idx = thinking.find(start_tag)
                if start_idx != -1:
                    start_idx += len(start_tag)
                    end_idx = thinking.find(end_tag, start_idx)
                    if end_idx != -1:
                        thinking = thinking[start_idx:end_idx].strip()
                    else:
                        thinking = thinking[start_idx:].strip()
                else:
                    thinking = thinking.replace("</think>", "").strip()
            elif "<think>" in thinking:
                start_tag = "<think>"
                end_tag = "</think>"
                start_idx = thinking.find(start_tag)
                if start_idx != -1:
                    start_idx += len(start_tag)
                    end_idx = thinking.find(end_tag, start_idx)
                    if end_idx != -1:
                        thinking = thinking[start_idx:end_idx].strip()
                    else:
                        thinking = thinking[start_idx:].strip()
                else:
                    thinking = thinking.replace("</think>", "").strip()
            # If no tags, keep the original content (it's the thinking part)
            action = "finish(message=" + parts[1]
            return thinking, action

        # Rule 3: Check for do(action=
        if "do(action=" in content:
            parts = content.split("do(action=", 1)
            thinking = parts[0].strip()
            # Clean up thinking: remove XML tags but preserve content
            # Try to extract from <think> tags if present
            if "<think>" in thinking:
                start_tag = "<think>"
                end_tag = "</think>"
                start_idx = thinking.find(start_tag)
                if start_idx != -1:
                    start_idx += len(start_tag)
                    end_idx = thinking.find(end_tag, start_idx)
                    if end_idx != -1:
                        thinking = thinking[start_idx:end_idx].strip()
                    else:
                        # Tag not closed, take everything after the opening tag
                        thinking = thinking[start_idx:].strip()
                else:
                    # Only closing tag found, remove it
                    thinking = thinking.replace("</think>", "").strip()
            # If no tags, keep the original content (it's the thinking part)
            # This preserves all text content before do(action=
            # IMPORTANT: If thinking is empty after cleaning, it means the content
            # started directly with do(action=, so there's no thinking part
            action = "do(action=" + parts[1]
            return thinking, action

        # Rule 4: Check for do( (without action=)
        if "do(" in content and "do(action=" not in content:
            parts = content.split("do(", 1)
            thinking = parts[0].strip()
            # Clean up thinking: remove XML tags but preserve content
            if "<think>" in thinking:
                start_tag = "<think>"
                end_tag = "</think>"
                start_idx = thinking.find(start_tag)
                if start_idx != -1:
                    start_idx += len(start_tag)
                    end_idx = thinking.find(end_tag, start_idx)
                    if end_idx != -1:
                        thinking = thinking[start_idx:end_idx].strip()
                    else:
                        thinking = thinking[start_idx:].strip()
                else:
                    thinking = thinking.replace("</think>", "").strip()
            elif "<think>" in thinking:
                start_tag = "<think>"
                end_tag = "</think>"
                start_idx = thinking.find(start_tag)
                if start_idx != -1:
                    start_idx += len(start_tag)
                    end_idx = thinking.find(end_tag, start_idx)
                    if end_idx != -1:
                        thinking = thinking[start_idx:end_idx].strip()
                    else:
                        thinking = thinking[start_idx:].strip()
                else:
                    thinking = thinking.replace("</think>", "").strip()
            # If no tags, keep the original content
            action = "do(" + parts[1]
            return thinking, action

        # Rule 5: Check for finish( (without message=)
        if "finish(" in content and "finish(message=" not in content:
            parts = content.split("finish(", 1)
            thinking = parts[0].strip()
            # Clean up thinking: remove XML tags but preserve content
            if "<think>" in thinking:
                start_tag = "<think>"
                end_tag = "</think>"
                start_idx = thinking.find(start_tag)
                if start_idx != -1:
                    start_idx += len(start_tag)
                    end_idx = thinking.find(end_tag, start_idx)
                    if end_idx != -1:
                        thinking = thinking[start_idx:end_idx].strip()
                    else:
                        thinking = thinking[start_idx:].strip()
                else:
                    thinking = thinking.replace("</think>", "").strip()
            elif "<think>" in thinking:
                start_tag = "<think>"
                end_tag = "</think>"
                start_idx = thinking.find(start_tag)
                if start_idx != -1:
                    start_idx += len(start_tag)
                    end_idx = thinking.find(end_tag, start_idx)
                    if end_idx != -1:
                        thinking = thinking[start_idx:end_idx].strip()
                    else:
                        thinking = thinking[start_idx:].strip()
                else:
                    thinking = thinking.replace("</think>", "").strip()
            # If no tags, keep the original content
            action = "finish(" + parts[1]
            return thinking, action

        # Rule 6: No markers found, try to extract thinking from <think> or <think> if present
        if "<think>" in content:
            start_tag = "<think>"
            end_tag = "</think>"
            start_idx = content.find(start_tag)
            if start_idx != -1:
                start_idx += len(start_tag)
                end_idx = content.find(end_tag, start_idx)
                if end_idx != -1:
                    thinking = content[start_idx:end_idx].strip()
                    # Use the rest as action
                    action = content[end_idx + len(end_tag):].strip()
                    if not action:
                        action = content
                    return thinking, action
        
        # Also check for <think> tag (without <answer>)
        if "<think>" in content and "<answer>" not in content:
            start_tag = "<think>"
            end_tag = "</think>"
            start_idx = content.find(start_tag)
            if start_idx != -1:
                start_idx += len(start_tag)
                end_idx = content.find(end_tag, start_idx)
                if end_idx != -1:
                    thinking = content[start_idx:end_idx].strip()
                    # Use the rest as action
                    action = content[end_idx + len(end_tag):].strip()
                    if not action:
                        action = content
                    return thinking, action

        # Rule 7: No markers found, return empty thinking and full content as action
        return "", content


class MessageBuilder:
    """Helper class for building conversation messages."""

    @staticmethod
    def create_system_message(content: str) -> dict[str, Any]:
        """Create a system message."""
        return {"role": "system", "content": content}

    @staticmethod
    def create_user_message(
        text: str, image_base64: str | None = None
    ) -> dict[str, Any]:
        """
        Create a user message with optional image.

        Args:
            text: Text content.
            image_base64: Optional base64-encoded image.

        Returns:
            Message dictionary.
        """
        content = []

        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )

        content.append({"type": "text", "text": text})

        return {"role": "user", "content": content}

    @staticmethod
    def create_assistant_message(content: str) -> dict[str, Any]:
        """Create an assistant message."""
        return {"role": "assistant", "content": content}

    @staticmethod
    def remove_images_from_message(message: dict[str, Any]) -> dict[str, Any]:
        """
        Remove image content from a message to save context space.

        Args:
            message: Message dictionary.

        Returns:
            Message with images removed.
        """
        if isinstance(message.get("content"), list):
            message["content"] = [
                item for item in message["content"] if item.get("type") == "text"
            ]
        return message

    @staticmethod
    def build_screen_info(current_app: str, **extra_info) -> str:
        """
        Build screen info string for the model.

        Args:
            current_app: Current app name.
            **extra_info: Additional info to include.

        Returns:
            JSON string with screen info.
        """
        info = {"current_app": current_app, **extra_info}
        return json.dumps(info, ensure_ascii=False)
