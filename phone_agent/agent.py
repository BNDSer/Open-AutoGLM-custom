"""Main PhoneAgent class for orchestrating phone automation."""

import json
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from phone_agent.actions import ActionHandler
from phone_agent.actions.handler import do, finish, parse_action
from phone_agent.adb import get_current_app, get_screenshot
from phone_agent.config import get_messages, get_system_prompt
from phone_agent.model import ModelClient, ModelConfig
from phone_agent.model.client import MessageBuilder


@dataclass
class AgentConfig:
    """Configuration for the PhoneAgent."""

    max_steps: int = 100
    device_id: str | None = None
    lang: str = "cn"
    system_prompt: str | None = None
    verbose: bool = True

    def __post_init__(self):
        if self.system_prompt is None:
            self.system_prompt = get_system_prompt(self.lang)


@dataclass
class StepResult:
    """Result of a single agent step."""

    success: bool
    finished: bool
    action: dict[str, Any] | None
    thinking: str
    message: str | None = None


class PhoneAgent:
    """
    AI-powered agent for automating Android phone interactions.

    The agent uses a vision-language model to understand screen content
    and decide on actions to complete user tasks.

    Args:
        model_config: Configuration for the AI model.
        agent_config: Configuration for the agent behavior.
        confirmation_callback: Optional callback for sensitive action confirmation.
        takeover_callback: Optional callback for takeover requests.

    Example:
        >>> from phone_agent import PhoneAgent
        >>> from phone_agent.model import ModelConfig
        >>>
        >>> model_config = ModelConfig(base_url="http://localhost:8000/v1")
        >>> agent = PhoneAgent(model_config)
        >>> agent.run("Open WeChat and send a message to John")
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        agent_config: AgentConfig | None = None,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], None] | None = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.agent_config = agent_config or AgentConfig()

        self.model_client = ModelClient(self.model_config)
        self.action_handler = ActionHandler(
            device_id=self.agent_config.device_id,
            confirmation_callback=confirmation_callback,
            takeover_callback=takeover_callback,
        )

        self._context: list[dict[str, Any]] = []
        self._step_count = 0
        self._interrupted = False
        self._interrupt_lock = threading.Lock()

    def run(self, task: str) -> str:
        """
        Run the agent to complete a task.

        Args:
            task: Natural language description of the task.

        Returns:
            Final message from the agent.
        """
        # Reset interrupted flag
        with self._interrupt_lock:
            self._interrupted = False
        
        self._context = []
        self._step_count = 0

        # First step with user prompt
        result = self._execute_step(task, is_first=True)

        if result.finished:
            return result.message or "Task completed"
        
        # Check if interrupted
        with self._interrupt_lock:
            if self._interrupted:
                return "Task interrupted by user"

        # Continue until finished or max steps reached
        while self._step_count < self.agent_config.max_steps:
            result = self._execute_step(is_first=False)

            # Immediately return if finished to avoid repeated thinking
            if result.finished:
                return result.message or "Task completed"
            
            # Check if interrupted
            with self._interrupt_lock:
                if self._interrupted:
                    return "Task interrupted by user"

        return "Max steps reached"

    def step(self, task: str | None = None) -> StepResult:
        """
        Execute a single step of the agent.

        Useful for manual control or debugging.

        Args:
            task: Task description (only needed for first step).

        Returns:
            StepResult with step details.
        """
        is_first = len(self._context) == 0

        if is_first and not task:
            raise ValueError("Task is required for the first step")

        return self._execute_step(task, is_first)

    def reset(self) -> None:
        """Reset the agent state for a new task."""
        self._context = []
        self._step_count = 0
        with self._interrupt_lock:
            self._interrupted = False
    
    def interrupt(self) -> None:
        """Interrupt the current task execution."""
        with self._interrupt_lock:
            self._interrupted = True
    
    def is_interrupted(self) -> bool:
        """Check if the agent has been interrupted."""
        with self._interrupt_lock:
            return self._interrupted

    def _execute_step(
        self, user_prompt: str | None = None, is_first: bool = False
    ) -> StepResult:
        """Execute a single step of the agent loop."""
        # Check if interrupted at the start of each step
        with self._interrupt_lock:
            if self._interrupted:
                return StepResult(
                    success=False,
                    finished=True,
                    action=None,
                    thinking="",
                    message="Task interrupted by user",
                )
        
        self._step_count += 1

        # Check if interrupted before capturing screen
        with self._interrupt_lock:
            if self._interrupted:
                return StepResult(
                    success=False,
                    finished=True,
                    action=None,
                    thinking="",
                    message="Task interrupted by user",
                )
        
        # Capture current screen state
        screenshot = get_screenshot(self.agent_config.device_id)
        current_app = get_current_app(self.agent_config.device_id)
        
        # Check if interrupted after capturing screen
        with self._interrupt_lock:
            if self._interrupted:
                return StepResult(
                    success=False,
                    finished=True,
                    action=None,
                    thinking="",
                    message="Task interrupted by user",
                )

        # Build messages
        if is_first:
            self._context.append(
                MessageBuilder.create_system_message(self.agent_config.system_prompt)
            )

            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"{user_prompt}\n\n{screen_info}"

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )
        else:
            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"** Screen Info **\n\n{screen_info}"

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )

        # Check if interrupted before model request
        with self._interrupt_lock:
            if self._interrupted:
                return StepResult(
                    success=False,
                    finished=True,
                    action=None,
                    thinking="",
                    message="Task interrupted by user",
                )
        
        # Get model response
        try:
            response = self.model_client.request(self._context)
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            return StepResult(
                success=False,
                finished=True,
                action=None,
                thinking="",
                message=f"Model error: {e}",
            )
        
        # Check if interrupted after model request
        with self._interrupt_lock:
            if self._interrupted:
                return StepResult(
                    success=False,
                    finished=True,
                    action=None,
                    thinking="",
                    message="Task interrupted by user",
                )

        # Parse action from response
        try:
            # Check if action is empty before parsing
            if not response.action or not response.action.strip():
                if self.agent_config.verbose:
                    print(f"\n⚠️  Warning: Empty action response from model")
                    print(f"Raw content: {response.raw_content[:200] if response.raw_content else 'None'}")
                # Create a note action to allow retry
                action = do(action="Note", message="Model returned empty action. Please retry with a valid action.")
            else:
                action = parse_action(response.action)
        except ValueError as e:
            if self.agent_config.verbose:
                print(f"\n⚠️  Warning: Failed to parse action: {e}")
                print(f"Raw action string: {response.action[:200] if response.action else '(empty)'}")
                print(f"Raw content: {response.raw_content[:200] if response.raw_content else 'None'}")
                traceback.print_exc()
            # Create a note action instead of finish to allow retry
            # This prevents the task from being incorrectly marked as completed
            action = do(action="Note", message=f"Failed to parse action: {str(e)}. Raw response: {response.action[:100] if response.action else '(empty)'}")

        if self.agent_config.verbose:
            # Print thinking process
            msgs = get_messages(self.agent_config.lang)
            print("\n" + "=" * 50)
            print(f"💭 {msgs['thinking']}:")
            print("-" * 50)
            # Ensure complete thinking output, handle empty or None cases
            thinking_output = response.thinking if response.thinking else ""
            # Remove any remaining XML tags that might interfere with display
            thinking_output = thinking_output.replace("<think>", "").replace("</think>", "")
            thinking_output = thinking_output.replace("<think>", "").replace("</think>", "")
            thinking_output = thinking_output.strip()
            
            # If thinking is empty, try to extract from raw content
            if not thinking_output and response.raw_content:
                # Try to extract thinking from raw content before action markers
                raw_content = response.raw_content
                # Find the position of action markers
                action_markers = ["do(action=", "do(", "finish(message=", "finish(", "<answer>"]
                action_pos = len(raw_content)
                for marker in action_markers:
                    pos = raw_content.find(marker)
                    if pos != -1 and pos < action_pos:
                        action_pos = pos
                
                # Extract thinking part before action
                if action_pos < len(raw_content) and action_pos > 0:
                    thinking_part = raw_content[:action_pos].strip()
                    # Clean up XML tags but preserve the actual content
                    # Remove opening and closing tags
                    thinking_part = thinking_part.replace("<think>", "").replace("</think>", "")
                    thinking_part = thinking_part.replace("<think>", "").replace("</think>", "")
                    thinking_part = thinking_part.replace("<think>", "").replace("</think>", "")
                    # Remove any leading/trailing whitespace and newlines
                    thinking_part = thinking_part.strip()
                    if thinking_part:
                        thinking_output = thinking_part
            
            # If still empty, show a message
            if not thinking_output:
                thinking_output = "(思考内容为空或无法解析)"
            
            print(thinking_output)
            print("-" * 50)
            print(f"🎯 {msgs['action']}:")
            print(json.dumps(action, ensure_ascii=False, indent=2))
            print("=" * 50 + "\n")

        # Remove image from context to save space
        self._context[-1] = MessageBuilder.remove_images_from_message(self._context[-1])

        # Execute action
        try:
            result = self.action_handler.execute(
                action, screenshot.width, screenshot.height
            )
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            result = self.action_handler.execute(
                finish(message=str(e)), screenshot.width, screenshot.height
            )

        # Check if finished
        finished = action.get("_metadata") == "finish" or result.should_finish

        # If finished, don't add assistant response to context to avoid repeated thinking
        # Only add to context if not finished
        if not finished:
            # Add assistant response to context
            # Use <think> and <answer> tags for consistency with parsing logic
            self._context.append(
                MessageBuilder.create_assistant_message(
                    f"<think>{response.thinking}</think><answer>{response.action}</answer>"
                )
            )
        else:
            # Task is finished, just log the final response without adding to context
            if self.agent_config.verbose:
                msgs = get_messages(self.agent_config.lang)
                print("\n" + "🎉 " + "=" * 48)
                print(
                    f"✅ {msgs['task_completed']}: {result.message or action.get('message', msgs['done'])}"
                )
                print("=" * 50 + "\n")

        return StepResult(
            success=result.success,
            finished=finished,
            action=action,
            thinking=response.thinking,
            message=result.message or action.get("message"),
        )

    @property
    def context(self) -> list[dict[str, Any]]:
        """Get the current conversation context."""
        return self._context.copy()

    @property
    def step_count(self) -> int:
        """Get the current step count."""
        return self._step_count
