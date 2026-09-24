import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

import voluntas.agent as agent_module
from voluntas import usage as usage_module
from voluntas.agent import BDI
from voluntas._utils import bcolors
from voluntas.logging import build_structured_run_log_entry


def test_colors_are_omitted_from_captured_stdout(capsys) -> None:
    print(f"{bcolors.SYSTEM}BDI cycle{bcolors.ENDC}")
    assert capsys.readouterr().out == "BDI cycle\n"


def _build_result(
    *,
    messages,
    output,
    response=None,
    output_tool_name=None,
    usage=None,
):
    if response is None:
        response = next(
            (message for message in reversed(messages) if isinstance(message, ModelResponse)),
            None,
        )

    result = SimpleNamespace(
        output=output,
        response=response,
        new_messages=lambda: list(messages),
        _output_tool_name=output_tool_name,
    )
    if usage is not None:
        result.usage = lambda: usage
    return result


def test_build_structured_run_log_entry_for_text_response() -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart("Summarize the latest changes.")]),
        ModelResponse(parts=[TextPart(content="The latest changes update the planner.")]),
    ]
    result = _build_result(
        messages=messages,
        output="The latest changes update the planner.",
    )

    entry = build_structured_run_log_entry("Summarize the latest changes.", result)

    assert entry == {
        "user": "Summarize the latest changes.",
        "assistant": "The latest changes update the planner.",
        "tool_calls": [],
    }


def test_build_structured_run_log_entry_for_structured_output_filters_output_tool() -> None:
    class StructuredAnswer(BaseModel):
        summary: str

    messages = [
        ModelRequest(parts=[UserPromptPart("Return a structured summary.")]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={"summary": "Planner updated"},
                    tool_call_id="output_1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="final_result",
                    content={"summary": "Planner updated"},
                    tool_call_id="output_1",
                )
            ]
        ),
        ModelResponse(parts=[]),
    ]
    result = _build_result(
        messages=messages,
        output=StructuredAnswer(summary="Planner updated"),
        response=messages[-1],
        output_tool_name="final_result",
    )

    entry = build_structured_run_log_entry(None, result)

    assert entry == {
        "user": "Return a structured summary.",
        "assistant": '{"summary": "Planner updated"}',
        "tool_calls": [],
    }


def test_build_structured_run_log_entry_for_tool_calls_preserves_order() -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart("Use the tools and report back.")]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="execute_code",
                    args='{"code": "print(1)"}',
                    tool_call_id="call_1",
                ),
                ToolCallPart(
                    tool_name="browse",
                    args="https://example.com",
                    tool_call_id="call_2",
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="execute_code",
                    content={"status": "ok"},
                    tool_call_id="call_1",
                ),
                ToolReturnPart(
                    tool_name="browse",
                    content="Loaded example.com",
                    tool_call_id="call_2",
                ),
            ]
        ),
        ModelResponse(parts=[TextPart(content="Finished.")]),
    ]
    result = _build_result(messages=messages, output="Finished.")

    entry = build_structured_run_log_entry(None, result)

    assert entry == {
        "user": "Use the tools and report back.",
        "assistant": "Finished.",
        "tool_calls": [
            {
                "func_name": "execute_code",
                "args": {"code": "print(1)"},
                "result": '{"status": "ok"}',
            },
            {
                "func_name": "browse",
                "args": {"input": "https://example.com"},
                "result": "Loaded example.com",
            },
        ],
    }


def test_build_structured_run_log_entry_includes_usage_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        usage_module,
        "_estimate_cost_usd",
        lambda _usage, _model_name: 0.123,
    )
    messages = [
        ModelRequest(parts=[UserPromptPart("Use tracked tokens.")]),
        ModelResponse(
            parts=[TextPart(content="Tracked.")],
            model_name="gpt-test",
        ),
    ]
    result = _build_result(
        messages=messages,
        output="Tracked.",
        usage=SimpleNamespace(
            requests=1,
            tool_calls=2,
            input_tokens=100,
            cache_read_tokens=40,
            cache_write_tokens=5,
            output_tokens=20,
            details={"reasoning_tokens": 7},
        ),
    )

    entry = build_structured_run_log_entry("Use tracked tokens.", result)

    assert entry == {
        "user": "Use tracked tokens.",
        "assistant": "Tracked.",
        "tool_calls": [],
        "usage": {
            "requests": 1,
            "tool_calls": 2,
            "input_tokens": 100,
            "cached_input_tokens": 40,
            "cache_read_tokens": 40,
            "cache_write_tokens": 5,
            "input_audio_tokens": 0,
            "cache_audio_read_tokens": 0,
            "output_tokens": 20,
            "output_audio_tokens": 0,
            "total_tokens": 120,
            "details": {"reasoning_tokens": 7},
        },
        "cost": {"usd": 0.123, "estimated": True},
        "model": "gpt-test",
    }


@pytest.mark.asyncio
async def test_bdi_run_records_usage_tracker(monkeypatch) -> None:
    async def fake_run(self, user_prompt=None, **_kwargs):
        assistant_text = f"assistant:{user_prompt}"
        messages = [
            ModelRequest(parts=[UserPromptPart(str(user_prompt))]),
            ModelResponse(parts=[TextPart(content=assistant_text)]),
        ]
        return _build_result(messages=messages, output=assistant_text)

    class FakeUsageTracker:
        def __init__(self):
            self.calls = []

        def record_result(self, result, *, attributes=None):
            self.calls.append((result, attributes))

    monkeypatch.setattr(Agent, "run", fake_run)

    usage_tracker = FakeUsageTracker()
    agent = BDI(usage_tracker=usage_tracker)
    agent.cycle_count = 2

    result = await agent.run("first")

    assert usage_tracker.calls == [
        (
            result,
            {
                "bdi_cycle_count": 2,
                "bdi_beliefs": 0,
                "bdi_desires": 0,
                "bdi_desire_statuses": {},
                "bdi_intentions": 0,
            },
        )
    ]


@pytest.mark.asyncio
async def test_bdi_run_emits_stdout_event_when_enabled(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        usage_module,
        "_estimate_cost_usd",
        lambda _usage, _model_name: 0.045,
    )

    async def fake_run(self, user_prompt=None, **_kwargs):
        assistant_text = f"assistant:{user_prompt}"
        messages = [
            ModelRequest(parts=[UserPromptPart(str(user_prompt))]),
            ModelResponse(
                parts=[TextPart(content=assistant_text)],
                model_name="gpt-test",
            ),
        ]
        return _build_result(
            messages=messages,
            output=assistant_text,
            usage=SimpleNamespace(
                requests=1,
                input_tokens=30,
                cache_read_tokens=10,
                output_tokens=6,
            ),
        )

    monkeypatch.setattr(Agent, "run", fake_run)

    agent = BDI(
        usage_tracker=usage_module.BDIUsageTracker(model_name="gpt-test"),
        emit_run_events_to_stdout=True,
    )
    capsys.readouterr()

    await agent.run("first")

    event = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert event == {
        "type": "voluntas.agent.run.completed",
        "run_index": 1,
        "user": "first",
        "assistant": "assistant:first",
        "tool_calls": [],
        "usage": {
            "requests": 1,
            "tool_calls": 0,
            "input_tokens": 30,
            "cached_input_tokens": 10,
            "cache_read_tokens": 10,
            "cache_write_tokens": 0,
            "input_audio_tokens": 0,
            "cache_audio_read_tokens": 0,
            "output_tokens": 6,
            "output_audio_tokens": 0,
            "total_tokens": 36,
            "details": {},
        },
        "cost": {"usd": 0.045, "estimated": True},
        "model": "gpt-test",
    }


@pytest.mark.asyncio
async def test_bdi_run_persists_structured_log_after_each_call(
    tmp_path, monkeypatch
) -> None:
    structured_log_path = tmp_path / "agent-run.json"

    async def fake_run(self, user_prompt=None, **_kwargs):
        assistant_text = f"assistant:{user_prompt}"
        messages = [
            ModelRequest(parts=[UserPromptPart(str(user_prompt))]),
            ModelResponse(parts=[TextPart(content=assistant_text)]),
        ]
        return _build_result(messages=messages, output=assistant_text)

    monkeypatch.setattr(Agent, "run", fake_run)

    agent = BDI(structured_log_file_path=str(structured_log_path))

    assert json.loads(structured_log_path.read_text()) == []

    await agent.run("first")
    assert json.loads(structured_log_path.read_text()) == [
        {
            "user": "first",
            "assistant": "assistant:first",
            "tool_calls": [],
        }
    ]

    await agent.run("second")
    assert json.loads(structured_log_path.read_text()) == [
        {
            "user": "first",
            "assistant": "assistant:first",
            "tool_calls": [],
        },
        {
            "user": "second",
            "assistant": "assistant:second",
            "tool_calls": [],
        },
    ]


@pytest.mark.asyncio
async def test_bdi_run_persists_structured_log_usage_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    structured_log_path = tmp_path / "agent-run.json"
    monkeypatch.setattr(
        usage_module,
        "_estimate_cost_usd",
        lambda _usage, _model_name: 0.045,
    )

    async def fake_run(self, user_prompt=None, **_kwargs):
        assistant_text = f"assistant:{user_prompt}"
        messages = [
            ModelRequest(parts=[UserPromptPart(str(user_prompt))]),
            ModelResponse(
                parts=[TextPart(content=assistant_text)],
                model_name="gpt-test",
            ),
        ]
        return _build_result(
            messages=messages,
            output=assistant_text,
            usage=SimpleNamespace(
                requests=1,
                input_tokens=30,
                cache_read_tokens=10,
                output_tokens=6,
            ),
        )

    monkeypatch.setattr(Agent, "run", fake_run)

    agent = BDI(structured_log_file_path=str(structured_log_path))
    await agent.run("first")

    assert json.loads(structured_log_path.read_text()) == [
        {
            "user": "first",
            "assistant": "assistant:first",
            "tool_calls": [],
            "usage": {
                "requests": 1,
                "tool_calls": 0,
                "input_tokens": 30,
                "cached_input_tokens": 10,
                "cache_read_tokens": 10,
                "cache_write_tokens": 0,
                "input_audio_tokens": 0,
                "cache_audio_read_tokens": 0,
                "output_tokens": 6,
                "output_audio_tokens": 0,
                "total_tokens": 36,
                "details": {},
            },
            "cost": {"usd": 0.045, "estimated": True},
            "model": "gpt-test",
        }
    ]

def test_bdi_initializes_text_and_structured_logs_independently(
    tmp_path, monkeypatch
) -> None:
    mirrored_paths: list[str] = []

    def fake_configure_terminal_output_mirror(
        log_file_path: str,
        *,
        strip_ansi: bool = True,
    ) -> None:
        assert strip_ansi is True
        mirrored_paths.append(log_file_path)

    monkeypatch.setattr(
        agent_module,
        "configure_terminal_output_mirror",
        fake_configure_terminal_output_mirror,
    )

    text_log_path = tmp_path / "agent.log"
    structured_log_path = tmp_path / "agent-run.json"

    BDI(
        log_file_path=str(text_log_path),
        structured_log_file_path=str(structured_log_path),
    )

    assert mirrored_paths == [str(text_log_path)]
    assert text_log_path.exists()
    assert json.loads(structured_log_path.read_text()) == []
