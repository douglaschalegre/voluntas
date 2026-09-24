"""Logging and state formatting utilities for the BDI agent.

This module provides functions for:
- mirroring terminal output to a log file
- formatting beliefs, desires, and intentions for display or LLM prompts
"""

from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional
import atexit
import os
import re
import sys
import threading

from voluntas._utils import bcolors

if TYPE_CHECKING:
    from voluntas.agent import BDI


_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class _TerminalMirrorStream:
    """Mirror a terminal stream to a file handle."""

    def __init__(
        self,
        terminal_stream,
        log_file,
        *,
        strip_ansi: bool,
        write_lock: threading.Lock,
    ):
        self._terminal_stream = terminal_stream
        self._log_file = log_file
        self._strip_ansi = strip_ansi
        self._write_lock = write_lock

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            data = str(data)

        written = self._terminal_stream.write(data)

        if data:
            log_data = _ANSI_ESCAPE_RE.sub("", data) if self._strip_ansi else data
            with self._write_lock:
                self._log_file.write(log_data)
                self._log_file.flush()

        return written

    def flush(self) -> None:
        self._terminal_stream.flush()
        with self._write_lock:
            self._log_file.flush()

    def isatty(self) -> bool:
        return self._terminal_stream.isatty()

    def fileno(self) -> int:
        return self._terminal_stream.fileno()

    def __getattr__(self, name):
        return getattr(self._terminal_stream, name)


class _TerminalMirrorState:
    """Runtime state for active stdout/stderr mirroring."""

    def __init__(
        self,
        *,
        log_file_path: str,
        log_file,
        original_stdout,
        original_stderr,
    ):
        self.log_file_path = log_file_path
        self.log_file = log_file
        self.original_stdout = original_stdout
        self.original_stderr = original_stderr


_terminal_mirror_state: Optional[_TerminalMirrorState] = None
_terminal_mirror_state_lock = threading.RLock()


def configure_terminal_output_mirror(
    log_file_path: str,
    *,
    strip_ansi: bool = True,
) -> None:
    """Mirror terminal stdout/stderr output into a log file.

    If mirroring is already active for the same path, this is a no-op.
    If mirroring is active for a different path, the previous mirror is replaced.
    """
    global _terminal_mirror_state

    if not log_file_path:
        return

    normalized_path = os.path.abspath(log_file_path)

    with _terminal_mirror_state_lock:
        if (
            _terminal_mirror_state
            and _terminal_mirror_state.log_file_path == normalized_path
        ):
            return

        if _terminal_mirror_state:
            disable_terminal_output_mirror()

        log_file = open(normalized_path, "a", encoding="utf-8")
        write_lock = threading.Lock()

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        sys.stdout = _TerminalMirrorStream(
            original_stdout,
            log_file,
            strip_ansi=strip_ansi,
            write_lock=write_lock,
        )
        sys.stderr = _TerminalMirrorStream(
            original_stderr,
            log_file,
            strip_ansi=strip_ansi,
            write_lock=write_lock,
        )

        _terminal_mirror_state = _TerminalMirrorState(
            log_file_path=normalized_path,
            log_file=log_file,
            original_stdout=original_stdout,
            original_stderr=original_stderr,
        )


def disable_terminal_output_mirror() -> None:
    """Disable active terminal output mirroring and restore streams."""
    global _terminal_mirror_state

    with _terminal_mirror_state_lock:
        if not _terminal_mirror_state:
            return

        state = _terminal_mirror_state
        _terminal_mirror_state = None

        sys.stdout = state.original_stdout
        sys.stderr = state.original_stderr

        try:
            state.log_file.flush()
        finally:
            state.log_file.close()


atexit.register(disable_terminal_output_mirror)


def format_beliefs_for_context(agent: "BDI") -> str:
    """Format current beliefs for inclusion in LLM prompts.

    Args:
        agent: The BDI agent instance

    Returns:
        Formatted string containing all beliefs with their values and certainty
    """
    if not agent.beliefs.beliefs:
        return "No beliefs recorded yet."

    beliefs_lines = []
    for name, belief in agent.beliefs.beliefs.items():
        beliefs_lines.append(
            f"- {name}: {belief.value} (Certainty: {belief.certainty:.2f})"
        )
    return "\n".join(beliefs_lines)


def log_states(
    agent: "BDI",
    types: list[Literal["beliefs", "desires", "intentions"]],
    message: str | None = None,
) -> None:
    """Log current agent state to the terminal.

    Args:
        agent: The BDI agent instance
        types: List of state types to log (beliefs, desires, intentions)
        message: Optional message to display with the state
    """
    if message:
        print(f"{bcolors.SYSTEM}{message}{bcolors.ENDC}")

    if "beliefs" in types:
        if agent.verbose:
            belief_str = "\n".join(
                [
                    f"  - {name}: {b.value} (Source: {b.source}, Certainty: {b.certainty:.2f}, Time: {datetime.fromtimestamp(b.timestamp).isoformat()})"
                    for name, b in agent.beliefs.beliefs.items()
                ]
            )
            print(f"{bcolors.BELIEF}Beliefs:\n{belief_str or '  (None)'}{bcolors.ENDC}")
        else:
            print(
                f"{bcolors.BELIEF}Beliefs: {len(agent.beliefs.beliefs)} items{bcolors.ENDC}"
            )

    if "desires" in types:
        if agent.verbose:
            desire_str = "\n".join(
                [
                    f"  - {d.id}: {d.description} (Status: {d.status.value}, Priority: {d.priority})"
                    for d in agent.desires
                ]
            )
            print(f"{bcolors.DESIRE}Desires:\n{desire_str or '  (None)'}{bcolors.ENDC}")
        else:
            summaries = [
                f"Desire '{desire.id}' {desire.status.value}"
                for desire in agent.desires
            ]
            print(
                f"{bcolors.DESIRE}Desires: {len(agent.desires)} items"
                f"{(' | ' + ' | '.join(summaries)) if summaries else ''}{bcolors.ENDC}"
            )

    if "intentions" in types:
        intentions = [agent.active_intention] if agent.active_intention else []
        if agent.verbose:
            intention_str = "\n".join(
                [
                    f"  - Desire '{i.desire_id}' | Intention: {i.description or '(no description)'} | Plan: {i.active_plan.status.value} | Current Plan Step -> {i.active_plan.steps[i.active_plan.current_step_index].description if i.active_plan.current_step_index < len(i.active_plan.steps) else '(Completed)'} (Step {i.active_plan.current_step_index + 1}/{len(i.active_plan.steps)})"
                    for i in intentions
                ]
            )
            print(
                f"{bcolors.INTENTION}Intentions:\n{intention_str or '  (None)'}{bcolors.ENDC}"
            )
        else:
            summaries = []
            for intention in intentions:
                plan = intention.active_plan
                step_count = len(plan.steps)
                if step_count:
                    step_number = min(plan.current_step_index + 1, step_count)
                    current_step = plan.current_step()
                    step_description = (
                        current_step.description if current_step else "(Completed)"
                    )
                else:
                    step_number = 0
                    step_description = "(No Plan Steps)"

                summaries.append(
                    f"Desire '{intention.desire_id}' "
                    f"Intention '{intention.description or '(no description)'}' "
                    f"Plan {plan.status.value} "
                    f"Plan Step {step_number}/{step_count}: {step_description}"
                )
            print(
                f"{bcolors.INTENTION}Intentions: {len(intentions)} items"
                f"{(' | ' + ' | '.join(summaries)) if summaries else ''}{bcolors.ENDC}"
            )


__all__ = [
    "configure_terminal_output_mirror",
    "disable_terminal_output_mirror",
    "format_beliefs_for_context",
    "log_states",
]
