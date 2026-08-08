"""Main BDI (Belief-Desire-Intention) agent implementation.

This module contains the main BDI agent class that orchestrates all BDI components:
beliefs, desires, intentions, planning, execution, and monitoring.
"""

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any, Generic, List, Optional, TypeVar, overload

from pydantic_ai import Agent, models, usage as _usage
from pydantic_ai.agent import (
    AgentMetadata,
    EventStreamHandler,
    Instructions,
    RunOutputDataT,
)
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.output import OutputSpec
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import BuiltinToolFunc, DeferredToolResults
from pydantic_ai.toolsets import AbstractToolset

from voluntas._utils import bcolors
from voluntas.schemas import (
    BeliefExtractionResult,
    BeliefSet,
    Desire,
    Intention,
    generate_desire_id,
)
from voluntas.belief_updates import update_beliefs_from_desire_extraction
from voluntas.errors import is_validation_output_error
from voluntas.logging import (
    build_structured_run_log_entry,
    configure_terminal_output_mirror,
    log_states,
)
from voluntas.usage import BDIUsageTracker
from voluntas.prompts import build_initial_belief_extraction_prompt
from voluntas.planning import generate_intentions_from_desires
from voluntas.execution import execute_intentions
from voluntas.cycle import bdi_cycle

T = TypeVar("T")


class _MaterializedStreamedRunResult(Generic[T]):
    """Expose a completed streamed run through the regular run-result API."""

    def __init__(self, streamed_result: Any, output: T) -> None:
        self._streamed_result = streamed_result
        self.output = output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._streamed_result, name)


class BDI(Agent, Generic[T]):
    """BDI (Belief-Desire-Intention) agent implementation.

    Extends the Pydantic AI Agent with a BDI architecture.
    Relies on the base Agent's capabilities for tool execution,
    including integration with MCP servers for external tools.
    """

    def __init__(
        self,
        *args,
        desires: Optional[List[str]] = None,
        intentions: Optional[List[str]] = None,
        verbose: bool = False,
        log_file_path: Optional[str] = None,
        structured_log_file_path: Optional[str] = None,
        usage_tracker: Optional[BDIUsageTracker] = None,
        emit_run_events_to_stdout: bool = False,
        stream_model_requests: bool = False,
        output_retries: int = 3,  # Higher default for structured output retries
        **kwargs,
    ):
        # Pass output_retries to the parent Agent class
        super().__init__(*args, output_retries=output_retries, **kwargs)
        self.beliefs = BeliefSet()
        self.desires: List[Desire] = []
        self.active_intention: Intention | None = None
        self.initial_intention_guidance: List[str] = intentions or []
        self._initial_intention_guidance_consumed = False
        self.verbose = verbose
        self.log_file_path = log_file_path
        self.structured_log_file_path = structured_log_file_path
        self.usage_tracker = usage_tracker
        self.emit_run_events_to_stdout = emit_run_events_to_stdout
        self.stream_model_requests = stream_model_requests
        self._structured_log_entries: list[dict[str, Any]] = []
        self.cycle_count = 0

        if self.log_file_path:
            self._initialize_log_file()

        if self.structured_log_file_path:
            self._initialize_structured_log_file()

        self._initialize_string_desires(desires)

    def _initialize_string_desires(self, desire_strings: Optional[List[str]]) -> None:
        """Initialize desires from string descriptions.

        Args:
            desire_strings: List of desire description strings
        """
        for desire_string in desire_strings or []:
            desire = Desire(
                id=generate_desire_id(desire_string),
                description=desire_string,
                priority=0.5,
            )
            self.desires.append(desire)
        log_states(self, ["desires"])

    async def extract_beliefs_from_desires(self) -> None:
        """Extract factual information from desire descriptions and add to beliefs.

        This method analyzes desire descriptions to extract concrete facts that should
        be recorded as initial beliefs, avoiding the need to rediscover this information
        during execution.
        """
        if not self.desires:
            return

        desires_text = "\n".join([f"- {d.description}" for d in self.desires])

        belief_extraction_prompt = build_initial_belief_extraction_prompt(desires_text)

        try:
            extraction_result = await self.run(
                belief_extraction_prompt, output_type=BeliefExtractionResult
            )

            if (
                extraction_result
                and extraction_result.output
                and extraction_result.output.beliefs
            ):
                update_stats = await update_beliefs_from_desire_extraction(
                    self, extraction_result.output.beliefs
                )

                if self.verbose:
                    print(
                        f"{bcolors.BELIEF}Extracted beliefs from desires: +{update_stats['created']} ~{update_stats['updated']} ={update_stats['unchanged']}.{bcolors.ENDC}"
                    )
                    log_states(self, ["beliefs"])

        except Exception as e:
            # Belief extraction is non-critical - log briefly and continue
            if is_validation_output_error(e):
                if self.verbose:
                    print(
                        f"{bcolors.WARNING}Initial belief extraction skipped: LLM output format issue.{bcolors.ENDC}"
                    )
            else:
                print(
                    f"{bcolors.WARNING}Could not extract beliefs from desires: {e}{bcolors.ENDC}"
                )

    def _initialize_log_file(self) -> None:
        """Initialize terminal-mirrored log file."""
        if not self.log_file_path:
            return

        log_path = self.log_file_path

        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            # Truncate file and mirror terminal output into it.
            with open(log_path, "w", encoding="utf-8"):
                pass

            configure_terminal_output_mirror(log_path)

            if self.verbose:
                print(
                    f"{bcolors.SYSTEM}Log file initialized at: {self.log_file_path}{bcolors.ENDC}"
                )
        except Exception as e:
            print(
                f"{bcolors.FAIL}Failed to initialize log file at {self.log_file_path}: {e}{bcolors.ENDC}"
            )
            self.log_file_path = None

    def _initialize_structured_log_file(self) -> None:
        """Initialize the structured JSON run log file."""
        if not self.structured_log_file_path:
            return

        try:
            Path(self.structured_log_file_path).parent.mkdir(
                parents=True, exist_ok=True
            )
            self._structured_log_entries = []
            with open(self.structured_log_file_path, "w", encoding="utf-8") as f:
                json.dump(self._structured_log_entries, f, ensure_ascii=False, indent=2)

            if self.verbose:
                print(
                    f"{bcolors.SYSTEM}Structured log file initialized at: {self.structured_log_file_path}{bcolors.ENDC}"
                )
        except Exception as e:
            print(
                f"{bcolors.FAIL}Failed to initialize structured log file at {self.structured_log_file_path}: {e}{bcolors.ENDC}"
            )
            self.structured_log_file_path = None

    def _persist_structured_log_entries(self) -> None:
        """Rewrite the structured run log JSON file."""
        if not self.structured_log_file_path:
            return

        try:
            with open(self.structured_log_file_path, "w", encoding="utf-8") as f:
                json.dump(self._structured_log_entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(
                f"{bcolors.WARNING}Failed to persist structured log at {self.structured_log_file_path}: {e}{bcolors.ENDC}"
            )
            self.structured_log_file_path = None

    def _record_structured_run(
        self,
        user_prompt: str | Sequence[UserContent] | None,
        result: AgentRunResult[Any],
    ) -> None:
        """Capture a single `run(...)` invocation as structured run metadata."""
        model_name = self.usage_tracker.model_name if self.usage_tracker else None
        entry = build_structured_run_log_entry(
            user_prompt,
            result,
            model_name=model_name,
        )
        self._structured_log_entries.append(entry)
        self._persist_structured_log_entries()
        if self.emit_run_events_to_stdout:
            self._emit_stdout_run_event(entry)

    def _emit_stdout_run_event(self, entry: dict[str, Any]) -> None:
        """Emit one JSON Lines event for SBench stdout capture."""
        event = {
            "type": "voluntas.agent.run.completed",
            "run_index": len(self._structured_log_entries),
            **entry,
        }
        print(json.dumps(event, ensure_ascii=True))

    def _usage_attributes(self) -> dict[str, Any]:
        desire_statuses: dict[str, int] = {}
        for desire in self.desires:
            status = desire.status.value
            desire_statuses[status] = desire_statuses.get(status, 0) + 1

        return {
            "bdi_cycle_count": self.cycle_count,
            "bdi_beliefs": len(self.beliefs.beliefs),
            "bdi_desires": len(self.desires),
            "bdi_desire_statuses": desire_statuses,
            "bdi_intentions": int(self.active_intention is not None),
        }

    def _record_usage(self, result: AgentRunResult[Any]) -> None:
        if self.usage_tracker is None:
            return

        self.usage_tracker.record_result(result, attributes=self._usage_attributes())

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[Any] = None,
        deps: Any = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[Any] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[Any]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[Any]]
        | None = None,
        event_stream_handler: EventStreamHandler[Any] | None = None,
    ) -> AgentRunResult[T]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[Any] = None,
        deps: Any = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[Any] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[Any]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[Any]]
        | None = None,
        event_stream_handler: EventStreamHandler[Any] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[Any] = None,
        deps: Any = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[Any] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[Any]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[Any]]
        | None = None,
        event_stream_handler: EventStreamHandler[Any] | None = None,
    ) -> AgentRunResult[Any]:
        """Run the underlying Pydantic AI agent and capture a structured log entry."""
        if self.stream_model_requests:
            async with super().run_stream(
                user_prompt=user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                instructions=instructions,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                metadata=metadata,
                infer_name=infer_name,
                toolsets=toolsets,
                builtin_tools=builtin_tools,
                event_stream_handler=event_stream_handler,
            ) as streamed_result:
                output = await streamed_result.get_output()
            result = _MaterializedStreamedRunResult(streamed_result, output)
        else:
            result = await super().run(
                user_prompt=user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                instructions=instructions,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                metadata=metadata,
                infer_name=infer_name,
                toolsets=toolsets,
                builtin_tools=builtin_tools,
                event_stream_handler=event_stream_handler,
            )

        try:
            self._record_usage(result)
        except Exception as e:
            print(
                f"{bcolors.WARNING}Failed to capture usage metadata: {e}{bcolors.ENDC}"
            )

        try:
            self._record_structured_run(user_prompt, result)
        except Exception as e:
            print(
                f"{bcolors.WARNING}Failed to capture structured run log entry: {e}{bcolors.ENDC}"
            )

        return result

    async def generate_intentions_from_desires(self) -> None:
        """Generate high-level intentions from desires."""
        await generate_intentions_from_desires(self)

    async def execute_intentions(self):
        """Execute one step of the current intention."""
        return await execute_intentions(self)

    async def bdi_cycle(self) -> str:
        """Run one BDI reasoning cycle.

        Returns:
            Status string indicating cycle outcome:
            - "executed": Normal cycle with work done
            - "terminal": All known desires are terminal and no intentions remain
            - "stopped": No pending work remains
        """
        return await bdi_cycle(self)

    def log_states(self, types: list, message: str | None = None):
        """Log agent state to terminal output."""
        log_states(self, types, message)


__all__ = ["BDI"]
