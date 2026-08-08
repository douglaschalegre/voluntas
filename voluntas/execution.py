"""Intention execution and belief extraction for the BDI agent.

This module handles executing individual steps of intentions, analyzing outcomes,
and extracting beliefs from execution results.
"""

from typing import TYPE_CHECKING, Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import traceback
import json
from types import SimpleNamespace

from voluntas._utils import bcolors
from voluntas.schemas import (
    PlanStep,
    BeliefExtractionResult,
    StepAssessmentResult,
)
from voluntas.errors import is_validation_output_error
from voluntas.belief_updates import update_beliefs_from_step_extraction
from voluntas.logging import log_states, format_beliefs_for_context
from voluntas.monitoring import generate_history_context
from voluntas.prompts import (
    build_descriptive_execution_prompt,
    build_step_assessment_prompt,
    build_step_belief_extraction_prompt,
    build_tool_execution_prompt,
)
from voluntas.state_transitions import complete_intention_and_update_desire

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRunResult
    from voluntas.agent import BDI

# Fixed retry configuration - opinionated approach
MAX_STEP_RETRIES = 2  # Total of 3 attempts per step


class ExecutionOutcomeKind(str, Enum):
    NO_INTENTION = "no_intention"
    STEP_SUCCEEDED = "step_succeeded"
    STEP_FAILED = "step_failed"
    PLAN_COMPLETED = "plan_completed"
    EXCEPTION = "exception"


@dataclass(frozen=True)
class ExecutionOutcome:
    kind: ExecutionOutcomeKind

    @property
    def should_reconsider(self) -> bool:
        return self.kind in {
            ExecutionOutcomeKind.STEP_FAILED,
            ExecutionOutcomeKind.EXCEPTION,
            ExecutionOutcomeKind.PLAN_COMPLETED,
        }


@dataclass
class StepRetryContext:
    """Transient context for tracking retry attempts during step execution.

    This is NOT persisted in schemas - it's execution-time only.
    """

    attempt_number: int = 0  # 0 = first attempt, 1+ = retries
    failure_history: List[Dict[str, Any]] = field(default_factory=list)

    def record_failure(
        self, result_output: str, beliefs_extracted: List[Dict[str, Any]]
    ):
        """Record a failed attempt with its beliefs."""
        self.failure_history.append(
            {
                "attempt": self.attempt_number,
                "result": result_output,
                "beliefs": beliefs_extracted,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def should_retry(self) -> bool:
        """Determine if we should retry based on attempt count."""
        return self.attempt_number < MAX_STEP_RETRIES

    def is_retry(self) -> bool:
        """Check if current execution is a retry."""
        return self.attempt_number > 0


async def extract_relevant_beliefs_from_result(
    agent: "BDI",
    step: PlanStep,
    result: Optional["AgentRunResult"],
    step_success: bool,
) -> List[Dict[str, Any]]:
    """Extract beliefs from a step result without updating the belief set.

    This is used during retry attempts to gather failure information.

    Args:
        agent: The BDI agent instance
        step: The PlanStep that was executed
        result: The result returned by the agent's run method
        step_success: Whether the step was assessed as successful

    Returns:
        List of belief dictionaries with keys: name, value, certainty
    """
    if agent.verbose:
        print(f"{bcolors.SYSTEM}  Extracting beliefs from step result...{bcolors.ENDC}")

    belief_extraction_prompt = build_step_belief_extraction_prompt(
        step.description,
        result.output if result else "No result",
        step_success,
        format_beliefs_for_context(agent),
    )

    extracted_beliefs = []

    try:
        extraction_result = await agent.run(
            belief_extraction_prompt, output_type=BeliefExtractionResult
        )

        if (
            extraction_result
            and extraction_result.output
            and extraction_result.output.beliefs
        ):
            for belief in extraction_result.output.beliefs:
                extracted_beliefs.append(
                    {
                        "name": belief.name,
                        "value": belief.value,
                        "certainty": belief.certainty,
                    }
                )

            if agent.verbose:
                print(
                    f"{bcolors.BELIEF}  Extracted {len(extracted_beliefs)} belief(s).{bcolors.ENDC}"
                )

    except Exception as extract_e:
        # Belief extraction is non-critical - log a warning but don't spam with tracebacks
        if is_validation_output_error(extract_e):
            # Common case: LLM returned wrong format - just note it briefly
            if agent.verbose:
                print(
                    f"{bcolors.WARNING}  Belief extraction skipped: LLM output format issue.{bcolors.ENDC}"
                )
        else:
            # Unexpected error - show more detail
            print(
                f"{bcolors.WARNING}  Belief extraction failed: {extract_e}{bcolors.ENDC}"
            )
            if agent.verbose:
                traceback.print_exc()

    return extracted_beliefs


async def analyze_step_outcome_and_update_beliefs(
    agent: "BDI",
    step: PlanStep,
    result: Optional["AgentRunResult"],
    extracted_beliefs_out: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Analyze the outcome of an executed step, updates beliefs, and determines success.

    Args:
        agent: The BDI agent instance
        step: The PlanStep that was executed
        result: The result returned by the base Agent's run method

    Returns:
        True if the step is considered successful, False otherwise
    """
    if agent.verbose:
        print(f"{bcolors.SYSTEM}  Analyzing step outcome...{bcolors.ENDC}")
    if not result:
        print(
            f"{bcolors.WARNING}  Analysis: Step failed - No result returned.{bcolors.ENDC}"
        )
        return False

    if agent.verbose:
        print(
            f"{bcolors.SYSTEM}  (Belief update check based on result: {result.output}){bcolors.ENDC}"
        )

    # Get history context
    intention = agent.active_intention
    if intention is None:
        return False
    history_context = generate_history_context(intention)

    # --- Success Assessment (using LLM) ---
    step_type = (
        f"Tool call: {step.tool_name}"
        if step.is_tool_call and step.tool_name
        else "Descriptive step"
    )
    assessment_prompt = build_step_assessment_prompt(
        step.description,
        result.output,
        step_type,
        history_context,
    )

    step_success = False
    try:
        assessment_result = await agent.run(
            assessment_prompt, output_type=StepAssessmentResult
        )
        if (
            assessment_result
            and assessment_result.output
            and assessment_result.output.success
        ):
            if agent.verbose:
                print(
                    f"{bcolors.SYSTEM}  LLM Assessment: Step SUCCEEDED.{bcolors.ENDC}"
                )
                if assessment_result.output.reason:
                    print(
                        f"{bcolors.SYSTEM}  Reason: {assessment_result.output.reason}{bcolors.ENDC}"
                    )
            step_success = True
        else:
            reason = (
                assessment_result.output.reason
                if (
                    assessment_result
                    and assessment_result.output
                    and assessment_result.output.reason
                )
                else "No assessment result or negative assessment"
            )
            print(
                f"{bcolors.WARNING}  LLM Assessment: Step FAILED. Reason: {reason}{bcolors.ENDC}"
            )
            step_success = False

    except Exception as assess_e:
        print(
            f"{bcolors.FAIL}  Error during LLM success assessment: {assess_e}{bcolors.ENDC}"
        )

        # INTELLIGENT FALLBACK: Don't default to failure
        # For tool calls, check if the result contains error indicators
        if step.is_tool_call and result and result.output:
            error_indicators = [
                "error",
                "exception",
                "failed to",
                "could not",
                "not found",
                "does not exist",
            ]
            result_lower = result.output.lower()

            has_error = any(indicator in result_lower for indicator in error_indicators)
            has_substantial_output = len(result.output) > 50

            if not has_error and has_substantial_output:
                print(
                    f"{bcolors.SYSTEM}  Fallback: Tool call returned substantial data without errors - marking as SUCCESS.{bcolors.ENDC}"
                )
                step_success = True
            else:
                print(
                    f"{bcolors.WARNING}  Fallback: Tool call appears to have failed - marking as FAILURE.{bcolors.ENDC}"
                )
                step_success = False
        else:
            # For non-tool calls, default to failure since we can't assess
            print(
                f"{bcolors.WARNING}  Fallback: Cannot assess non-tool step - marking as FAILURE.{bcolors.ENDC}"
            )
            step_success = False

    # --- Belief Extraction ---
    # Extract beliefs from the step result regardless of success/failure
    extracted_beliefs = await extract_relevant_beliefs_from_result(
        agent, step, result, step_success
    )
    if extracted_beliefs_out is not None:
        extracted_beliefs_out.extend(extracted_beliefs)

    # Update the belief set with extracted beliefs
    if extracted_beliefs:
        plan = intention.active_plan
        print(
            f"{bcolors.BELIEF}  Processing {len(extracted_beliefs)} extracted belief(s).{bcolors.ENDC}"
        )
        update_stats = await update_beliefs_from_step_extraction(
            agent,
            extracted_beliefs,
            source=f"plan_step_{plan.current_step_index + 1}_{step.description[:30]}",
        )
        if agent.verbose:
            print(
                f"{bcolors.BELIEF}  Belief changes: +{update_stats['created']} ~{update_stats['updated']} ={update_stats['unchanged']}{bcolors.ENDC}"
            )
    else:
        if agent.verbose:
            print(f"{bcolors.SYSTEM}  No beliefs extracted.{bcolors.ENDC}")

    return step_success


def _snapshot_beliefs(agent: "BDI") -> Dict[str, Dict[str, Any]]:
    return {
        name: {"value": b.value, "source": b.source, "certainty": b.certainty}
        for name, b in agent.beliefs.beliefs.items()
    }


def _build_retry_context(retry_ctx: StepRetryContext) -> str:
    if not retry_ctx.failure_history:
        return ""

    lines = ["", "", "PREVIOUS ATTEMPT FAILURES:"]
    for failure in retry_ctx.failure_history:
        lines.append(f"- Attempt {failure['attempt'] + 1}: {failure['result']}")
        if failure["beliefs"]:
            lines.append(f"  Beliefs learned: {json.dumps(failure['beliefs'])}")
    lines.append("")
    lines.append("Please consider these failures and adjust your approach accordingly.")
    return "\n".join(lines) + "\n"


def _log_tool_debug_messages(agent: "BDI", step_result: "AgentRunResult") -> None:
    if not agent.verbose or not hasattr(step_result, "all_messages"):
        return

    all_messages = step_result.all_messages()
    if not all_messages:
        return

    print(f"{bcolors.SYSTEM}  === DEBUG: Tool Call Details ==={bcolors.ENDC}")
    for msg in all_messages:
        parts = getattr(msg, "parts", None)
        if not parts:
            continue

        for part in parts:
            tool_name = getattr(part, "tool_name", None)
            if tool_name:
                print(f"{bcolors.SYSTEM}    Tool: {tool_name}{bcolors.ENDC}")
                args = getattr(part, "args", None)
                if args is not None:
                    print(f"{bcolors.SYSTEM}    Args: {args}{bcolors.ENDC}")
                continue

            if getattr(part, "tool_call_id", None):
                content = getattr(part, "content", "")
                content_preview = (
                    "<binary>"
                    if isinstance(content, (bytes, bytearray))
                    else str(content)[:200]
                )
                print(f"{bcolors.SYSTEM}    Result: {content_preview}...{bcolors.ENDC}")

    print(f"{bcolors.SYSTEM}  === End Tool Call Details ==={bcolors.ENDC}")


def _extract_latest_tool_result_content(step_result: "AgentRunResult") -> Optional[str]:
    """Extract the latest raw tool result payload from agent messages.

    For tool-execution steps we prefer the direct tool output over the model's
    synthesized final narrative response.
    """
    if not hasattr(step_result, "all_messages"):
        return None

    all_messages = step_result.all_messages()
    if not all_messages:
        return None

    latest_content: Optional[str] = None
    for msg in all_messages:
        parts = getattr(msg, "parts", None)
        if not parts:
            continue

        for part in parts:
            if not getattr(part, "tool_call_id", None):
                continue

            content = getattr(part, "content", None)
            if content is None:
                continue

            if isinstance(content, (bytes, bytearray)):
                latest_content = content.decode(errors="replace")
            else:
                latest_content = str(content)

    return latest_content


async def _run_step_attempt(
    agent: "BDI",
    current_step: PlanStep,
    retry_ctx: StepRetryContext,
) -> "AgentRunResult":
    is_retry = retry_ctx.is_retry()
    beliefs_context = format_beliefs_for_context(agent)
    retry_context = _build_retry_context(retry_ctx) if is_retry else ""

    if current_step.is_tool_call and current_step.tool_name:
        if agent.verbose:
            print(
                f"{bcolors.SYSTEM}  Attempting tool call via self.run: {current_step.tool_name}({current_step.tool_params}){bcolors.ENDC}"
            )

        tool_prompt = build_tool_execution_prompt(
            beliefs_context,
            retry_context,
            current_step.tool_name,
            current_step.tool_params or {},
            is_retry,
        )
        step_result = await agent.run(tool_prompt)
        extracted_tool_output = _extract_latest_tool_result_content(step_result)
        effective_output = extracted_tool_output or step_result.output
        print(
            f"{bcolors.SYSTEM}  Tool '{current_step.tool_name}' result: {effective_output}{bcolors.ENDC}"
        )
        _log_tool_debug_messages(agent, step_result)
        return SimpleNamespace(output=effective_output)

    if agent.verbose:
        print(
            f"{bcolors.SYSTEM}  Executing descriptive step via self.run: {current_step.description}{bcolors.ENDC}"
        )

    enhanced_prompt = build_descriptive_execution_prompt(
        beliefs_context,
        retry_context,
        current_step.description,
        is_retry,
    )
    step_result = await agent.run(enhanced_prompt)
    if agent.verbose:
        print(f"{bcolors.SYSTEM}  Step result: {step_result.output}{bcolors.ENDC}")

    return step_result


def _handle_step_execution_exception(
    agent: "BDI",
    intention,
    current_step: PlanStep,
    error: Exception,
) -> None:
    print(f"{bcolors.FAIL}Exception during step execution: {error}{bcolors.ENDC}")
    traceback.print_exc()

    plan = intention.active_plan
    plan.record_failure(
        current_step,
        f"Exception: {str(error)}",
        _snapshot_beliefs(agent),
    )

    print(
        f"{bcolors.WARNING}  Plan marked failed after exception; plan-level reconsideration will decide whether to continue, repair, replace, or fail the Desire.{bcolors.ENDC}"
    )


async def _run_step_with_retries(
    agent: "BDI",
    intention,
    current_step: PlanStep,
) -> tuple[
    Optional["AgentRunResult"],
    bool,
    StepRetryContext,
    Optional[ExecutionOutcome],
]:
    retry_ctx = StepRetryContext(attempt_number=0)
    step_result: Optional["AgentRunResult"] = None
    step_succeeded = False
    plan = intention.active_plan

    while True:
        if retry_ctx.is_retry():
            attempt_label = (
                f"(Attempt {retry_ctx.attempt_number + 1}/{MAX_STEP_RETRIES + 1})"
            )
            print(
                f"{bcolors.WARNING}  Retrying Plan Step {plan.current_step_index + 1} {attempt_label}{bcolors.ENDC}"
            )

        try:
            step_result = await _run_step_attempt(agent, current_step, retry_ctx)
            extracted_beliefs: List[Dict[str, Any]] = []
            step_succeeded = await analyze_step_outcome_and_update_beliefs(
                agent,
                current_step,
                step_result,
                extracted_beliefs_out=extracted_beliefs,
            )

            if step_succeeded:
                print(
                    f"{bcolors.INTENTION}  Plan Step {plan.current_step_index + 1} successful{' after retry' if retry_ctx.is_retry() else ''}.{bcolors.ENDC}"
                )
                break

            retry_ctx.record_failure(
                result_output=step_result.output if step_result else "No result",
                beliefs_extracted=extracted_beliefs,
            )
            retry_ctx.attempt_number += 1

            if retry_ctx.should_retry():
                print(
                    f"{bcolors.WARNING}  Plan Step {plan.current_step_index + 1} failed. Auto-retry enabled (attempt {retry_ctx.attempt_number + 1}/{MAX_STEP_RETRIES + 1}).{bcolors.ENDC}"
                )
                continue

            print(
                f"{bcolors.WARNING}  Plan Step {plan.current_step_index + 1} failed after {retry_ctx.attempt_number} retries. Escalating...{bcolors.ENDC}"
            )
            break

        except Exception as error:
            _handle_step_execution_exception(agent, intention, current_step, error)
            return None, False, retry_ctx, ExecutionOutcome(ExecutionOutcomeKind.EXCEPTION)

    return step_result, step_succeeded, retry_ctx, None


async def _handle_successful_step(
    agent: "BDI", intention, current_step: PlanStep, result: str
) -> ExecutionOutcome:
    plan = intention.active_plan
    completed = plan.record_outcome_and_advance(
        current_step, result, _snapshot_beliefs(agent)
    )

    if completed:
        print(
            f"{bcolors.INTENTION}Completed final step. Intention for desire '{intention.desire_id}' finished.{bcolors.ENDC}"
        )
        await complete_intention_and_update_desire(agent, intention)
        return ExecutionOutcome(ExecutionOutcomeKind.PLAN_COMPLETED)
    log_states(
        agent,
        types=["intentions"],
        message=f"Plan for desire '{intention.desire_id}' advanced to Plan Step {plan.current_step_index + 1}",
    )
    return ExecutionOutcome(ExecutionOutcomeKind.STEP_SUCCEEDED)


async def _handle_failed_step(
    agent: "BDI",
    intention,
    current_step: PlanStep,
    step_result: Optional["AgentRunResult"],
    retry_ctx: StepRetryContext,
) -> ExecutionOutcome:
    plan = intention.active_plan
    print(
        f"{bcolors.WARNING}  Plan Step {plan.current_step_index + 1} failed analysis after {retry_ctx.attempt_number} attempt(s). Intention progress paused.{bcolors.ENDC}"
    )

    log_states(agent, ["beliefs"])
    plan.record_failure(
        current_step,
        step_result.output if step_result else "No result",
        _snapshot_beliefs(agent),
    )
    print(
        f"{bcolors.WARNING}  Plan marked failed; plan-level reconsideration will decide whether to continue, repair, replace, or fail the Desire.{bcolors.ENDC}"
    )
    return ExecutionOutcome(ExecutionOutcomeKind.STEP_FAILED)


async def execute_intentions(agent: "BDI") -> ExecutionOutcome:
    """Execute one step of the current intention, analyze outcome, and handle success/failure.

    Does NOT proceed to the next step if the current step fails analysis.

    Args:
        agent: The BDI agent instance

    Returns:
        The outcome of executing the current plan step.
    """
    if agent.active_intention is None:
        print(f"{bcolors.SYSTEM}No intentions to execute.{bcolors.ENDC}")
        return ExecutionOutcome(ExecutionOutcomeKind.NO_INTENTION)

    intention = agent.active_intention
    plan = intention.active_plan
    if plan.is_complete():
        plan.mark_completed()
        print(
            f"{bcolors.INTENTION}Intention for desire '{intention.desire_id}' already completed (found in execute_intentions).{bcolors.ENDC}"
        )
        await complete_intention_and_update_desire(agent, intention)
        return ExecutionOutcome(ExecutionOutcomeKind.PLAN_COMPLETED)

    current_step = plan.current_step()
    if current_step is None:
        raise RuntimeError("Active Plan has no current step")
    print(
        f"{bcolors.INTENTION}Executing Desire '{intention.desire_id}' | Intention '{intention.description or '(no description)'}' | Plan status '{plan.status.value}' | Plan Step {plan.current_step_index + 1}/{len(plan.steps)}: {current_step.description}{bcolors.ENDC}"
    )

    (
        step_result,
        step_succeeded,
        retry_ctx,
        early_return,
    ) = await _run_step_with_retries(agent, intention, current_step)
    if early_return is not None:
        return early_return

    if step_succeeded:
        return await _handle_successful_step(
            agent,
            intention,
            current_step,
            step_result.output if step_result else "No result",
        )

    return await _handle_failed_step(
        agent,
        intention,
        current_step,
        step_result,
        retry_ctx,
    )


__all__ = [
    "analyze_step_outcome_and_update_beliefs",
    "ExecutionOutcome",
    "ExecutionOutcomeKind",
    "execute_intentions",
]
