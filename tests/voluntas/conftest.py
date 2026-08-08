from collections import deque
from types import SimpleNamespace

import pytest

from voluntas.schemas import BeliefSet, Desire, DesireStatus, Intention, Plan, PlanStep


class StubBDIAgent:
    def __init__(self):
        self.beliefs = BeliefSet()
        self.desires = []
        self.active_intention = None
        self.initial_intention_guidance = []
        self.verbose = False
        self.tool_configs = {}
        self._queued_run_outputs = deque()
        self.run_calls = []
        self.cycle_count = 0
        self.extracted_beliefs_from_desires = False

    def add_desire(
        self,
        *,
        desire_id: str,
        description: str,
        status: DesireStatus = DesireStatus.PENDING,
        priority: float = 0.5,
    ) -> Desire:
        desire = Desire(
            id=desire_id,
            description=description,
            priority=priority,
            status=status,
        )
        self.desires.append(desire)
        return desire

    def set_current_intention(
        self,
        *,
        desire_id: str,
        step_descriptions: list[str],
        current_step: int = 0,
    ) -> Intention:
        intention = Intention(
            desire_id=desire_id,
            description=f"Intention for {desire_id}",
            active_plan=Plan(
                steps=[PlanStep(description=s) for s in step_descriptions],
                current_step_index=current_step,
            ),
        )
        self.active_intention = intention
        return intention

    def queue_run_output(self, output) -> None:
        self._queued_run_outputs.append(SimpleNamespace(output=output))

    async def run(self, *_args, **_kwargs):
        prompt = _args[0] if _args else _kwargs.get("prompt", "")
        self.run_calls.append(
            {
                "prompt": prompt,
                "output_type": _kwargs.get("output_type"),
            }
        )
        if not self._queued_run_outputs:
            raise AssertionError("StubBDIAgent.run called without queued output")
        return self._queued_run_outputs.popleft()

    async def extract_beliefs_from_desires(self) -> None:
        self.extracted_beliefs_from_desires = True


@pytest.fixture
def stub_agent() -> StubBDIAgent:
    return StubBDIAgent()
