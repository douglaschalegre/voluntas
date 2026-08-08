"""Prompt builders for BDI modules.

Centralizes long prompt text to keep operational code paths easier to read.
"""

from typing import Any
import json
from textwrap import dedent


def build_initial_belief_extraction_prompt(desires_text: str) -> str:
    return dedent(
        f"""
        Analyze the following desire descriptions and extract any factual information that should be recorded as beliefs.

        Desire Descriptions:
        {desires_text}

        Extract ONLY concrete, factual information explicitly stated in the desires, such as:
        - File paths or directory paths (e.g., "repository path is /path/to/repo")
        - Names or identifiers (e.g., "the project is called X")
        - URLs or endpoints
        - Specific values or configurations mentioned
        - Any other concrete facts that would be useful context

        Do NOT extract:
        - The goals or objectives themselves (these are desires, not beliefs)
        - Inferred or assumed information not explicitly stated
        - Vague or subjective statements

        IMPORTANT: Each belief MUST have exactly these three fields:
        - "name": A concise identifier string (e.g., "repo_path", "project_name", "target_url")
        - "value": The actual value as a string (e.g., "/path/to/repo", "my-project", "https://api.example.com")
        - "certainty": A float between 0.0 and 1.0 (use 1.0 for explicitly stated facts)

        Example of CORRECT format:
        {{
          "beliefs": [
            {{"name": "repo_path", "value": "/Users/douglas/code/masters/pydantic-ai-voluntas", "certainty": 1.0}},
            {{"name": "repo_name", "value": "pydantic-ai-voluntas", "certainty": 1.0}}
          ],
          "explanation": "Extracted repository path and name from desire description."
        }}

        Example of INCORRECT format (DO NOT USE):
        {{
          "beliefs": [{{"repo_path": "/path", "repo_name": "project"}}]
        }}

        If no factual information can be extracted, return an empty beliefs list with an explanation.
        """
    )


def build_planning_stage1_prompt(
    desires_text: str,
    beliefs_text: str,
    intention_guidance_text: str | None = None,
) -> str:
    guidance_section = (
        f"""
        Initial Intention Guidance:
        {intention_guidance_text}

        Treat this as contextual guidance from the original user-provided workflow, not as a mandatory queue to recreate.
        """
        if intention_guidance_text
        else ""
    )
    return dedent(
        f"""
        Given the following pending desires and current beliefs, select exactly one desire to pursue now and identify exactly one high-level intention for it.
        The intention should represent a distinct goal or task achievable *by you, the AI agent*.

        Focus ONLY on WHAT needs to be done at a high level, but ensure these goals are achievable through information processing, analysis, or using the available tools.
        Do *not* propose intentions that you do not possess the hability to do.

        Pending Desires:
        {desires_text}

        Current Beliefs:
        {beliefs_text}

        {guidance_section}

        Available Tools:
        (The underlying Pydantic AI agent will provide the available tools, including those from MCP, to the LLM.)

        Respond with exactly one high-level intention using the required format. Associate it with the selected desire ID.
        Do not propose intentions for unselected desires.
        """
    )


def build_step_belief_extraction_prompt(
    step_description: str,
    step_result: str,
    step_success: bool,
    current_beliefs: str,
) -> str:
    return dedent(
        f"""
        Analyze the following step execution and extract any factual information that should be recorded as beliefs.

        Step Objective: "{step_description}"
        Step Result: "{step_result}"
        Step Success: {step_success}

        Current Known Beliefs:
        {current_beliefs}

        Extract beliefs about:
        - Factual information discovered (e.g., file paths, status values, API responses)
        - Error causes or constraints (e.g., "path does not exist", "network unavailable")
        - State changes or conditions revealed (e.g., "repository is empty", "file contains X")
        - Tool availability or limitations learned (e.g., "tool requires parameter Y")

        For FAILED steps, focus on extracting information about WHY it failed - these constraints are valuable.
        For SUCCESSFUL steps, extract the positive information discovered.

        CRITICAL DEDUPLICATION RULES:
        - Do NOT re-emit facts that are already present in Current Known Beliefs unless the value changed.
        - Do NOT emit synonyms for existing belief names (reuse the same belief name when possible).
        - If the step only repeats existing beliefs, return an empty beliefs list.
        - Avoid operational/meta beliefs like step_success unless they represent genuinely new state.

        IMPORTANT: Each belief MUST have exactly these three fields:
        - "name": A concise identifier string (e.g., "repo_path", "commit_count", "error_type")
        - "value": The actual value as a string (e.g., "/path/to/repo", "42", "permission_denied")
        - "certainty": A float between 0.0 and 1.0 indicating confidence

        Example of CORRECT format:
        {{
          "beliefs": [
            {{"name": "repo_path", "value": "/Users/douglas/code/project", "certainty": 1.0}},
            {{"name": "has_commits", "value": "true", "certainty": 0.9}}
          ],
          "explanation": "Extracted repository path and confirmed commits exist."
        }}

        Example of INCORRECT format (DO NOT USE):
        {{
          "beliefs": [{{"repo_path": "/path", "has_commits": true}}]
        }}

        If no meaningful beliefs can be extracted, return an empty beliefs list with an explanation.
        """
    )


def build_step_assessment_prompt(
    step_description: str,
    result_output: str,
    step_type: str,
    history_context: str,
) -> str:
    return dedent(
        f"""
        Original objective for the step: "{step_description}"
        Result obtained: "{result_output}"
        Step type: {step_type}

        Recent step history:
        {history_context}

        Evaluate if the step successfully achieved its original objective.

        Assessment Guidelines:

        FOR TOOL CALL STEPS:
        - If the tool executed and returned data (not an error message), mark as SUCCESS
        - The result may include analysis or discussion of the data - this is normal and doesn't indicate failure
        - Only mark as FAILED if the tool returned an error, exception, or "not found" type message

        FOR CHECK/VERIFY STEPS:
        - If the result provides a definitive answer (yes/no, found/not found, true/false), mark as SUCCESS

        FOR DESCRIPTIVE STEPS (no tool call):
        - If the step produced a concrete outcome or information, mark as SUCCESS
        - If the step only discussed what should be done without doing it, mark as FAILED

        FOR ALL STEPS:
        - Ignore verbose explanations or analysis in the result - focus on whether the objective was met
        - If the result explicitly states an error occurred, mark as FAILED
        - If uncertain, prefer SUCCESS over FAILURE (be lenient)

        Provide your assessment:
        - success: true if the step achieved its objective, false if it clearly failed
        - reason: brief explanation (especially important if failed)
        """
    )


def build_desire_satisfaction_prompt(
    desire_id: str,
    desire_description: str,
    completed_intention_description: str,
    completed_intention_history: str,
    current_beliefs: str,
    remaining_intentions_text: str,
) -> str:
    return dedent(
        f"""
        Assess whether the Desire is satisfied after a completed Intention.

        Desire:
        - ID: {desire_id}
        - Description: {desire_description}

        Completed Intention:
        {completed_intention_description}

        Completed Intention History:
        {completed_intention_history}

        Current Beliefs:
        {current_beliefs}

        Remaining Intentions for this Desire:
        {remaining_intentions_text}

        Decide only whether the Desire itself is now satisfied.

        Decision rules:
        1. Return satisfied=true only when the Desire description has been fulfilled by the completed Intention history and current beliefs.
        2. Return satisfied=false when useful work remains, when the outcome is partial, or when the evidence is unclear.
        3. Do not mark the Desire satisfied merely because the Intention completed.
        4. Remaining Intentions are context, not proof that the Desire is unsatisfied.

        Provide your assessment as:
        - satisfied: true if the Desire is fulfilled, false otherwise
        - reason: concise explanation for the lifecycle decision
        """
    )


def build_tool_execution_prompt(
    beliefs_context: str,
    retry_context: str,
    tool_name: str,
    tool_params: dict[str, Any],
    is_retry: bool,
) -> str:
    retry_warning = (
        "IMPORTANT: Previous attempts failed. Review the failure information above and modify your approach."
        if is_retry
        else ""
    )
    return dedent(
        f"""
        Current known information (beliefs):
        {beliefs_context}
        {retry_context}
        Execute the tool '{tool_name}' with the suggested parameters: {tool_params}

        You may adjust parameters if current beliefs suggest better values or if conditions have changed.
        {retry_warning}
        You MUST call the specified tool for this step and base your response on the tool's returned output.
        Do not claim inability/capability constraints unless the tool call explicitly fails with an error.
        Perform this action now.
        """
    )


def build_descriptive_execution_prompt(
    beliefs_context: str,
    retry_context: str,
    task_description: str,
    is_retry: bool,
) -> str:
    retry_warning = (
        "IMPORTANT: Previous attempts failed. Review the failure information above and modify your approach."
        if is_retry
        else ""
    )
    return dedent(
        f"""
        Current known information (beliefs):
        {beliefs_context}
        {retry_context}
        Task: {task_description}

        Consider the current beliefs when executing this task.
        {retry_warning}
        """
    )


def build_reconsideration_prompt(
    beliefs_text: str,
    completed_steps_text: str,
    desire_id: str,
    remaining_steps_text: str,
    failure_history_text: str,
) -> str:
    return dedent(
        f"""
        Current Agent Beliefs:
        {beliefs_text}

        Completed Plan Steps:
        {completed_steps_text}

        Remaining Plan Steps (for Desire ID '{desire_id}'):
        {remaining_steps_text}

        Relevant Failure History:
        {failure_history_text}

        Evaluate the active Plan as a whole, not whether one Plan Step is sufficient by itself.

        Provide your assessment as:
        - action: one of "continue", "repair_plan", "replace_plan", or "fail_desire"
        - reason: brief explanation for the selected action
        - plan_steps: required for "repair_plan" and "replace_plan"; omit or return null otherwise

        Consider:
        1. Does the full Plan remain likely to achieve the original desire '{desire_id}'?
        2. Did completed Plan Steps make valid progress toward the Desire?
        3. Do failures, contradictions, stale assumptions, or current Beliefs require repair or replacement?
        4. Use "continue" only when the existing Plan should keep running as-is.
        5. Use "repair_plan" when the current Plan is mostly sound but needs local changes.
        6. Use "replace_plan" when the Plan strategy is no longer suitable and should be replanned.
        7. Use "fail_desire" only when the Desire should be abandoned as infeasible or invalid.
        8. For "repair_plan", plan_steps should replace the remaining Plan Steps from the current failed/stale step onward.
        9. For "replace_plan", plan_steps should be a full replacement Plan for the committed Intention.
        """
    )


def build_belief_update_resolution_prompt(
    belief_name: str,
    existing_value: Any,
    existing_certainty: float,
    incoming_value: Any,
    incoming_certainty: float,
    incoming_source: str,
) -> str:
    return dedent(
        f"""
        Decide whether an incoming belief should update an existing belief.

        Belief name: {belief_name}
        Existing belief value: {json.dumps(existing_value)}
        Existing certainty: {existing_certainty}

        Incoming belief value: {json.dumps(incoming_value)}
        Incoming certainty: {incoming_certainty}
        Incoming source: {incoming_source}

        Rules:
        1. If incoming value is semantically equivalent to existing value, do not update.
        2. If incoming value adds meaningful correction/detail, update.
        3. If values conflict, prefer the more reliable and specific value.
        4. Certainty should be between 0.0 and 1.0.

        Return structured output with:
        - should_update: true/false
        - normalized_value: string value to store if updating (or existing value if not)
        - certainty: certainty to store
        - rationale: short justification
        """
    )


def build_belief_name_resolution_prompt(
    incoming_name: str,
    incoming_value: Any,
    existing_beliefs: dict[str, Any],
) -> str:
    return dedent(
        f"""
        Resolve whether an incoming belief should reuse an existing belief name.

        Incoming belief name: {incoming_name}
        Incoming belief value: {json.dumps(incoming_value)}

        Existing beliefs (name -> value):
        {json.dumps(existing_beliefs)}

        Rules:
        1. If an existing belief name already represents the same concept, return that exact existing name.
        2. If no existing belief name matches semantically, keep the incoming name.
        3. Prefer stable naming and avoid creating synonyms.

        Return structured output with:
        - resolved_name: final belief name to use
        - rationale: short explanation
        """
    )


__all__ = [
    "build_descriptive_execution_prompt",
    "build_initial_belief_extraction_prompt",
    "build_planning_stage1_prompt",
    "build_desire_satisfaction_prompt",
    "build_reconsideration_prompt",
    "build_belief_update_resolution_prompt",
    "build_belief_name_resolution_prompt",
    "build_step_assessment_prompt",
    "build_step_belief_extraction_prompt",
    "build_tool_execution_prompt",
]
