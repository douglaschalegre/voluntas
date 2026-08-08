"""Centralized belief update helpers for BDI flows."""

import json
from difflib import SequenceMatcher
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Dict, Iterable, Literal, Tuple, cast

from voluntas._utils import bcolors
from voluntas.errors import is_validation_output_error
from voluntas.schemas.belief_schemas import BatchBeliefResolutionResult

if TYPE_CHECKING:
    from voluntas.agent import BDI
    from voluntas.schemas import ExtractedBelief


BeliefMutation = Literal["created", "updated", "unchanged"]
BeliefStats = Dict[BeliefMutation, int]
_MUTATION_MARKERS: Dict[BeliefMutation, str] = {
    "created": "+",
    "updated": "~",
    "unchanged": "=",
}


def _normalize_belief_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _current_belief_value_map(agent: "BDI") -> Dict[str, Any]:
    return {name: belief.value for name, belief in agent.beliefs.beliefs.items()}


def _empty_stats() -> BeliefStats:
    return {"created": 0, "updated": 0, "unchanged": 0}


def _mutation_marker(mutation: BeliefMutation) -> str:
    return _MUTATION_MARKERS[mutation]


def _stable_value_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return repr(value)


def _deduplicate_beliefs(
    beliefs: Iterable[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    deduplicated: list[Dict[str, Any]] = []
    seen: Dict[tuple[str, str], int] = {}

    for belief_dict in beliefs:
        belief_name = _normalize_belief_name(str(belief_dict["name"]))
        belief_value = belief_dict["value"]
        belief_certainty = float(belief_dict.get("certainty", 0.8))
        key = (belief_name, _stable_value_key(belief_value))

        if key in seen:
            existing = deduplicated[seen[key]]
            existing["certainty"] = max(existing["certainty"], belief_certainty)
            continue

        seen[key] = len(deduplicated)
        deduplicated.append(
            {
                "name": belief_name,
                "value": belief_value,
                "certainty": belief_certainty,
                "source": belief_dict["source"],
            }
        )

    return deduplicated


_NAME_TOKEN_ALIASES = {
    "directories": "directory",
    "dirs": "directory",
    "dir": "directory",
    "repositories": "repository",
    "repo": "repository",
}


def _canonical_name_token(token: str) -> str:
    token = token.strip().lower()
    token = _NAME_TOKEN_ALIASES.get(token, token)
    if len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def _belief_name_tokens(name: str) -> set[str]:
    return {
        _canonical_name_token(token)
        for token in name.replace("-", "_").split("_")
        if token
    }


def _is_potential_name_match(incoming_name: str, existing_name: str) -> bool:
    incoming_tokens = _belief_name_tokens(incoming_name)
    existing_tokens = _belief_name_tokens(existing_name)
    shared_tokens = incoming_tokens & existing_tokens

    if incoming_tokens and incoming_tokens == existing_tokens:
        return True
    if len(shared_tokens) >= 2:
        return True
    return SequenceMatcher(None, incoming_name, existing_name).ratio() >= 0.72


def _ambiguous_existing_names(agent: "BDI", incoming_name: str) -> list[str]:
    return [
        existing_name
        for existing_name in agent.beliefs.beliefs
        if existing_name != incoming_name
        and _is_potential_name_match(incoming_name, existing_name)
    ]


def _build_batch_belief_resolution_prompt(
    *,
    pending_beliefs: list[Dict[str, Any]],
    existing_beliefs: Dict[str, Any],
) -> str:
    incoming_beliefs = [
        {
            "incoming_index": belief["index"],
            "name": belief["name"],
            "value": belief["value"],
            "certainty": belief["certainty"],
            "source": belief["source"],
            "resolution_reason": belief["resolution_reason"],
            "candidate_existing_names": belief["candidate_existing_names"],
        }
        for belief in pending_beliefs
    ]

    return dedent(f"""
        Resolve this batch of incoming belief updates against the current belief state.

        Current beliefs (name -> value):
        {json.dumps(existing_beliefs, default=str, sort_keys=True)}

        Incoming beliefs needing resolution:
        {json.dumps(incoming_beliefs, default=str, sort_keys=True)}

        Rules:
        1. Return exactly one decision for each incoming belief by incoming_index.
        2. For ambiguous names, reuse an existing belief name only when it represents the same concept.
        3. For conflicting values, decide whether the incoming value should replace the existing value.
        4. If values are semantically equivalent, set should_update=false and keep the existing value.
        5. If creating a new belief, use the incoming normalized name as resolved_name.
        6. Certainty must be between 0.0 and 1.0.

        Return structured output with a decisions list. Each decision must include:
        - incoming_index
        - resolved_name
        - should_update
        - normalized_value
        - certainty
        - rationale
        """)


async def _llm_resolve_belief_updates_batch(
    agent: "BDI",
    pending_beliefs: list[Dict[str, Any]],
) -> Dict[int, tuple[str, bool, Any, float]]:
    prompt = _build_batch_belief_resolution_prompt(
        pending_beliefs=pending_beliefs,
        existing_beliefs=_current_belief_value_map(agent),
    )

    try:
        result = await agent.run(prompt, output_type=BatchBeliefResolutionResult)
        if not result or not result.output:
            return {}

        pending_by_index = {belief["index"]: belief for belief in pending_beliefs}
        decisions: Dict[int, tuple[str, bool, Any, float]] = {}
        duplicate_indices: set[int] = set()
        for decision in result.output.decisions:
            index = decision.incoming_index
            pending = pending_by_index.get(index)
            if pending is None or index in decisions:
                duplicate_indices.add(index)
                continue

            resolved_name = _normalize_belief_name(decision.resolved_name)
            allowed_names = {
                pending["name"],
                *(
                    _normalize_belief_name(name)
                    for name in pending["candidate_existing_names"]
                ),
            }
            if resolved_name not in allowed_names:
                continue
            decisions[index] = (
                resolved_name,
                decision.should_update,
                decision.normalized_value,
                max(0.0, min(1.0, float(decision.certainty))),
            )
        for index in duplicate_indices:
            decisions.pop(index, None)
        return decisions
    except Exception as e:
        if agent.verbose and not is_validation_output_error(e):
            print(
                f"{bcolors.WARNING}  Batch belief resolution failed, using fallback: {e}{bcolors.ENDC}"
            )

    return {}


def _fallback_batch_decision(
    agent: "BDI", pending_belief: Dict[str, Any]
) -> tuple[str, bool, Any, float]:
    belief_name = pending_belief["name"]
    existing = agent.beliefs.get(belief_name)

    if existing and existing.value == pending_belief["value"]:
        should_update = pending_belief["certainty"] > existing.certainty
        return (
            belief_name,
            should_update,
            pending_belief["value"],
            pending_belief["certainty"],
        )

    return belief_name, True, pending_belief["value"], pending_belief["certainty"]


def _apply_resolved_belief_update(
    agent: "BDI",
    *,
    belief_name: str,
    value_to_store: Any,
    certainty_to_store: float,
    source: str,
    should_update: bool,
) -> Tuple[str, Any, float, BeliefMutation]:
    if not should_update:
        existing = agent.beliefs.get(belief_name)
        if existing:
            return belief_name, existing.value, existing.certainty, "unchanged"
        return belief_name, value_to_store, certainty_to_store, "unchanged"

    mutation = cast(
        BeliefMutation,
        agent.beliefs.upsert(
            name=belief_name,
            value=value_to_store,
            source=source,
            certainty=certainty_to_store,
        ),
    )
    stored = agent.beliefs.get(belief_name)
    if stored:
        return belief_name, stored.value, stored.certainty, mutation
    return belief_name, value_to_store, certainty_to_store, mutation


async def _apply_belief_batch(
    agent: "BDI", beliefs: Iterable[Dict[str, Any]]
) -> tuple[BeliefStats, list[Tuple[str, Any, float, BeliefMutation]]]:
    stats = _empty_stats()
    applied_results: list[Tuple[str, Any, float, BeliefMutation]] = []
    pending_resolution: list[Dict[str, Any]] = []
    prepared = _deduplicate_beliefs(beliefs)
    values_by_name: Dict[str, set[str]] = {}
    for belief in prepared:
        values_by_name.setdefault(belief["name"], set()).add(
            _stable_value_key(belief["value"])
        )

    for index, belief_dict in enumerate(prepared):
        belief_name = belief_dict["name"]
        incoming_value = belief_dict["value"]
        incoming_certainty = belief_dict["certainty"]
        source = belief_dict["source"]
        existing = agent.beliefs.get(belief_name)

        if (existing and existing.value != incoming_value) or len(
            values_by_name[belief_name]
        ) > 1:
            pending_resolution.append(
                {
                    "index": index,
                    "name": belief_name,
                    "value": incoming_value,
                    "certainty": incoming_certainty,
                    "source": source,
                    "resolution_reason": "conflicting_value",
                    "candidate_existing_names": [belief_name],
                }
            )
            continue

        ambiguous_names = (
            [] if existing else _ambiguous_existing_names(agent, belief_name)
        )
        if ambiguous_names:
            pending_resolution.append(
                {
                    "index": index,
                    "name": belief_name,
                    "value": incoming_value,
                    "certainty": incoming_certainty,
                    "source": source,
                    "resolution_reason": "ambiguous_name",
                    "candidate_existing_names": ambiguous_names,
                }
            )
            continue

        applied_results.append(
            _apply_resolved_belief_update(
                agent,
                belief_name=belief_name,
                value_to_store=incoming_value,
                certainty_to_store=incoming_certainty,
                source=source,
                should_update=True,
            )
        )

    if pending_resolution:
        decisions = await _llm_resolve_belief_updates_batch(agent, pending_resolution)
        for pending_belief in pending_resolution:
            belief_name, should_update, value_to_store, certainty_to_store = (
                decisions.get(pending_belief["index"])
                or _fallback_batch_decision(agent, pending_belief)
            )
            applied_results.append(
                _apply_resolved_belief_update(
                    agent,
                    belief_name=belief_name,
                    value_to_store=value_to_store,
                    certainty_to_store=certainty_to_store,
                    source=pending_belief["source"],
                    should_update=should_update,
                )
            )

    for belief_name, value_to_store, certainty_to_store, mutation in applied_results:
        stats[mutation] += 1
        if agent.verbose:
            print(
                f"{bcolors.BELIEF}    {_mutation_marker(mutation)} {belief_name}: {value_to_store} (Certainty: {certainty_to_store:.2f}) [{mutation}]{bcolors.ENDC}"
            )

    return stats, applied_results


async def update_beliefs_from_desire_extraction(
    agent: "BDI", beliefs: Iterable["ExtractedBelief"]
) -> BeliefStats:
    """Apply beliefs extracted from initial desire descriptions."""
    prepared = (
        {
            "name": belief.name,
            "value": belief.value,
            "certainty": belief.certainty,
            "source": "desire_description",
        }
        for belief in beliefs
    )
    stats, _ = await _apply_belief_batch(agent, prepared)
    return stats


async def update_beliefs_from_step_extraction(
    agent: "BDI", beliefs: Iterable[Dict[str, Any]], source: str
) -> BeliefStats:
    """Apply beliefs extracted from a step result."""
    prepared = (
        {
            "name": belief["name"],
            "value": belief["value"],
            "certainty": belief.get("certainty", 0.8),
            "source": source,
        }
        for belief in beliefs
    )
    stats, _ = await _apply_belief_batch(agent, prepared)
    return stats


__all__ = [
    "update_beliefs_from_desire_extraction",
    "update_beliefs_from_step_extraction",
]
