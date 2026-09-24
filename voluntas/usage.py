"""Usage and cost tracking helpers for BDI agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


TOKEN_FIELDS = (
    "input_tokens",
    "cache_write_tokens",
    "cache_read_tokens",
    "input_audio_tokens",
    "cache_audio_read_tokens",
    "output_tokens",
    "output_audio_tokens",
)


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _as_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float | Decimal):
        return float(value)
    return None


def _usage_value(usage: Any, field_name: str) -> int:
    return _as_int(getattr(usage, field_name, 0))


def _usage_details(usage: Any) -> dict[str, int]:
    details = getattr(usage, "details", None)
    if not isinstance(details, dict):
        return {}

    return {
        str(key): _as_int(value) for key, value in details.items() if _as_int(value)
    }


def _extract_result_usage(result: Any) -> Any | None:
    usage = getattr(result, "usage", None)
    if callable(usage):
        return usage()
    return usage


def _price_to_float(price: Any) -> float | None:
    if (numeric := _as_numeric(price)) is not None:
        return numeric

    if isinstance(price, dict):
        for key in ("total_price", "price", "cost", "total", "usd"):
            if (numeric := _as_numeric(price.get(key))) is not None:
                return numeric
        return None

    for attr_name in ("total_price", "price", "cost", "total", "usd"):
        value = getattr(price, attr_name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if (numeric := _as_numeric(value)) is not None:
            return numeric

    return None


def _build_request_usage(usage: Any) -> Any:
    try:
        from pydantic_ai.usage import RequestUsage

        values = {
            field_name: _usage_value(usage, field_name) for field_name in TOKEN_FIELDS
        }
        return RequestUsage(**values, details=_usage_details(usage))
    except Exception:
        return usage


def _estimate_cost_usd(usage: Any, model_name: str | None) -> float | None:
    if not model_name:
        return None

    request_usage = _build_request_usage(usage)

    calculators = []
    try:
        from genai_prices import calc_price

        calculators.append(calc_price)
    except Exception:
        pass

    try:
        from genai_prices.price import calc_price

        calculators.append(calc_price)
    except Exception:
        pass

    for calc_price in calculators:
        attempts = (
            lambda: calc_price(request_usage, model_ref=model_name),
            lambda: calc_price(request_usage, model_name),
            lambda: calc_price(model_name, request_usage),
        )
        for attempt in attempts:
            try:
                if (cost := _price_to_float(attempt())) is not None:
                    return cost
            except Exception:
                continue

    return None


def _increment_eval_metric(name: str, amount: int | float) -> None:
    try:
        from pydantic_evals import increment_eval_metric

        increment_eval_metric(name, amount)
    except Exception:
        return


def _set_eval_attribute(name: str, value: Any) -> None:
    try:
        from pydantic_evals import set_eval_attribute

        set_eval_attribute(name, value)
    except Exception:
        return


def summarize_usage(usage: Any) -> dict[str, Any]:
    """Return the stable token/request summary shape used by BDI logs."""
    input_tokens = _usage_value(usage, "input_tokens")
    cache_read_tokens = _usage_value(usage, "cache_read_tokens")
    output_tokens = _usage_value(usage, "output_tokens")

    return {
        "requests": _as_int(getattr(usage, "requests", 1)),
        "tool_calls": _as_int(getattr(usage, "tool_calls", 0)),
        "input_tokens": input_tokens,
        "cached_input_tokens": cache_read_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": _usage_value(usage, "cache_write_tokens"),
        "input_audio_tokens": _usage_value(usage, "input_audio_tokens"),
        "cache_audio_read_tokens": _usage_value(usage, "cache_audio_read_tokens"),
        "output_tokens": output_tokens,
        "output_audio_tokens": _usage_value(usage, "output_audio_tokens"),
        "total_tokens": input_tokens + output_tokens,
        "details": dict(sorted(_usage_details(usage).items())),
    }


@dataclass
class BDIUsageTracker:
    """Aggregate Pydantic AI usage across BDI model calls."""

    model_name: str | None = None
    requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    input_audio_tokens: int = 0
    cache_audio_read_tokens: int = 0
    output_tokens: int = 0
    output_audio_tokens: int = 0
    details: dict[str, int] = field(default_factory=dict)
    cost_usd: float | None = None
    priced_requests: int = 0
    unpriced_requests: int = 0

    def record_result(
        self, result: Any, *, attributes: dict[str, Any] | None = None
    ) -> None:
        usage = _extract_result_usage(result)
        if usage is None:
            return
        self.record_usage(usage, attributes=attributes)

    def record_usage(
        self, usage: Any, *, attributes: dict[str, Any] | None = None
    ) -> None:
        requests = _as_int(getattr(usage, "requests", 1))
        tool_calls = _as_int(getattr(usage, "tool_calls", 0))
        input_tokens = _usage_value(usage, "input_tokens")
        cache_write_tokens = _usage_value(usage, "cache_write_tokens")
        cache_read_tokens = _usage_value(usage, "cache_read_tokens")
        input_audio_tokens = _usage_value(usage, "input_audio_tokens")
        cache_audio_read_tokens = _usage_value(usage, "cache_audio_read_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        output_audio_tokens = _usage_value(usage, "output_audio_tokens")
        details = _usage_details(usage)

        self.requests += requests
        self.tool_calls += tool_calls
        self.input_tokens += input_tokens
        self.cache_write_tokens += cache_write_tokens
        self.cache_read_tokens += cache_read_tokens
        self.input_audio_tokens += input_audio_tokens
        self.cache_audio_read_tokens += cache_audio_read_tokens
        self.output_tokens += output_tokens
        self.output_audio_tokens += output_audio_tokens

        for key, value in details.items():
            self.details[key] = self.details.get(key, 0) + value

        cost = _estimate_cost_usd(usage, self.model_name)
        if cost is None:
            self.unpriced_requests += requests
        else:
            self.priced_requests += requests
            self.cost_usd = (self.cost_usd or 0.0) + cost

        self._record_eval_metrics(
            requests=requests,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
            input_audio_tokens=input_audio_tokens,
            cache_audio_read_tokens=cache_audio_read_tokens,
            output_tokens=output_tokens,
            output_audio_tokens=output_audio_tokens,
            details=details,
            cost_usd=cost,
            attributes=attributes,
        )

    def usage_summary(self) -> dict[str, Any]:
        return summarize_usage(self)

    def cost_summary(self) -> dict[str, Any]:
        return {
            "usd": self.cost_usd,
            "estimated": self.cost_usd is not None,
            "priced_requests": self.priced_requests,
            "unpriced_requests": self.unpriced_requests,
        }

    def _record_eval_metrics(
        self,
        *,
        requests: int,
        tool_calls: int,
        input_tokens: int,
        cache_write_tokens: int,
        cache_read_tokens: int,
        input_audio_tokens: int,
        cache_audio_read_tokens: int,
        output_tokens: int,
        output_audio_tokens: int,
        details: dict[str, int],
        cost_usd: float | None,
        attributes: dict[str, Any] | None,
    ) -> None:
        metric_values = {
            "requests": requests,
            "tool_calls": tool_calls,
            "input_tokens": input_tokens,
            "cached_input_tokens": cache_read_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "input_audio_tokens": input_audio_tokens,
            "cache_audio_read_tokens": cache_audio_read_tokens,
            "output_tokens": output_tokens,
            "output_audio_tokens": output_audio_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        for metric_name, amount in metric_values.items():
            _increment_eval_metric(metric_name, amount)
        for key, amount in details.items():
            _increment_eval_metric(key, amount)
        if cost_usd is not None:
            _increment_eval_metric("cost", cost_usd)

        if self.model_name is not None:
            _set_eval_attribute("model", self.model_name)
        if attributes:
            for key, value in attributes.items():
                _set_eval_attribute(key, value)


__all__ = [
    "BDIUsageTracker",
    "summarize_usage",
]
