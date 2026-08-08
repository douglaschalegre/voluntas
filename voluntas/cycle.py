"""BDI reasoning cycle orchestration.

This module implements the main BDI reasoning cycle that coordinates belief updates,
deliberation, intention generation, execution, and plan monitoring.
"""

from typing import TYPE_CHECKING

from voluntas._utils import bcolors
from voluntas.schemas import DesireStatus
from voluntas.logging import log_states
from voluntas.planning import generate_intentions_from_desires
from voluntas.execution import ExecutionOutcome, ExecutionOutcomeKind, execute_intentions
from voluntas.monitoring import reconsider_current_intention
from voluntas.state_transitions import all_desires_terminal

if TYPE_CHECKING:
    from voluntas.agent import BDI


FINAL_CYCLE_STATUSES = frozenset({"terminal", "stopped"})


def is_final_cycle_status(status: str) -> bool:
    """Return True when a BDI cycle status should stop its caller loop."""
    return status in FINAL_CYCLE_STATUSES


async def bdi_cycle(agent: "BDI") -> str:
    """Run one BDI reasoning cycle including reconsideration.

    The cycle follows the standard BDI architecture phases:
    1. Belief Update (via action outcomes in execution)
    2. Deliberation (desire status check)
    3. Intention Generation (if needed)
    4. Intention Execution (one step)
    5. Reconsideration (plan validity monitoring)

    If the agent is idle (no intentions and no active desires), the cycle
    stops without requesting additional input.

    Args:
        agent: The BDI agent instance

    Returns:
        Status string indicating cycle outcome:
        - "executed": Normal cycle with work done
        - "terminal": All known desires are terminal and no intentions remain
        - "stopped": No pending work remains
    """
    agent.cycle_count += 1

    if agent.active_intention is None and all_desires_terminal(agent):
        print(
            f"{bcolors.SYSTEM}All known desires are terminal and no intentions remain. Stopping cleanly.{bcolors.ENDC}"
        )
        log_states(
            agent,
            types=["beliefs", "desires", "intentions"],
            message="States after BDI cycle (terminal)",
        )
        return "terminal"

    # Extract beliefs from desires on first cycle (before any execution)
    # This ensures factual information in desire descriptions is available as beliefs
    if agent.cycle_count == 1 and not agent.beliefs.beliefs and agent.desires:
        print(
            f"{bcolors.SYSTEM}First cycle: Extracting initial beliefs from desire descriptions...{bcolors.ENDC}"
        )
        await agent.extract_beliefs_from_desires()

    log_states(
        agent,
        types=["beliefs", "desires", "intentions"],
        message="States before starting BDI cycle",
    )
    print(f"{bcolors.SYSTEM}\n--- BDI Cycle Start ---{bcolors.ENDC}")

    # 1. Belief Update (Triggered by Action Outcomes)
    # Beliefs are updated within analyze_step_outcome_and_update_beliefs
    # after an action is taken in execute_intentions.
    if agent.verbose:
        print(f"{bcolors.BELIEF}Current Beliefs:{bcolors.ENDC}")
        log_states(agent, ["beliefs"])

    # 2. Deliberation / Desire Status Check
    # Check for active/pending desires.
    if agent.verbose:
        print(f"{bcolors.DESIRE}Current Desires:{bcolors.ENDC}")
        log_states(agent, ["desires"])
    active_desires = [
        d
        for d in agent.desires
        if d.status in [DesireStatus.PENDING, DesireStatus.ACTIVE]
    ]

    # 3. Intention Generation (if needed)
    # If we have active/pending desires but no intentions queued, generate them.
    if active_desires and agent.active_intention is None:
        print(
            f"{bcolors.SYSTEM}No current intentions, but active/pending desires exist. Generating intentions...{bcolors.ENDC}"
        )
        await generate_intentions_from_desires(agent)
    else:
        if agent.verbose:
            print(f"{bcolors.INTENTION}Current Intentions:{bcolors.ENDC}")
            log_states(agent, ["intentions"])
        if agent.active_intention is None:
            print(
                f"{bcolors.SYSTEM}No intentions pending and no active desires require new ones.{bcolors.ENDC}"
            )

            print(
                f"{bcolors.SYSTEM}Agent is idle with no pending work. Stopping.{bcolors.ENDC}"
            )
            return "stopped"

    # 4. Intention Execution (One Step)
    outcome = ExecutionOutcome(ExecutionOutcomeKind.NO_INTENTION)
    if agent.active_intention is not None:
        outcome = await execute_intentions(agent)
    else:
        print(f"{bcolors.SYSTEM}No intentions to execute this cycle.{bcolors.ENDC}")

    # 5. Reconsideration (Plan Monitoring)
    # After executing a step (successfully or not), reconsider the current plan.
    if agent.active_intention is not None:
        if outcome.kind is ExecutionOutcomeKind.STEP_SUCCEEDED:
            print(
                f"{bcolors.SYSTEM}  Skipping reconsideration: Plan Step succeeded and Plan progress should continue.{bcolors.ENDC}"
            )
        # Only reconsider if the intention wasn't just completed/removed by execute_intentions
        elif outcome.should_reconsider:
            await reconsider_current_intention(agent)
        else:
            print(
                f"{bcolors.SYSTEM}  Skipping reconsideration: Current intention just completed, removed, or has no steps.{bcolors.ENDC}"
            )
    else:
        print(
            f"{bcolors.SYSTEM}  Skipping reconsideration: No intentions remaining.{bcolors.ENDC}"
        )

    if agent.active_intention is None and all_desires_terminal(agent):
        print(f"{bcolors.SYSTEM}--- BDI Cycle End (terminal) ---{bcolors.ENDC}")
        log_states(
            agent,
            types=["beliefs", "desires", "intentions"],
            message="States after BDI cycle (terminal)",
        )
        return "terminal"

    print(f"{bcolors.SYSTEM}--- BDI Cycle End ---{bcolors.ENDC}")
    log_states(
        agent,
        types=["beliefs", "desires", "intentions"],
        message="States after BDI cycle",
    )

    return "executed"


__all__ = [
    "FINAL_CYCLE_STATUSES",
    "bdi_cycle",
    "is_final_cycle_status",
]
