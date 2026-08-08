# Voluntas

Voluntas is a BDI (Belief-Desire-Intention) agent framework built on top of
[Pydantic AI](https://ai.pydantic.dev/). It provides structured beliefs,
desires, intentions, adaptive planning, execution, reconsideration, usage
tracking.

## Installation

```bash
pip install voluntas
```

Or with uv:

```bash
uv add voluntas
```

The distribution name is `voluntas` and the Python import is also `voluntas`.

## Quick start

```python
import asyncio

from pydantic_ai.models.test import TestModel
from voluntas import BDI


async def main() -> None:
    agent = BDI(
        model=TestModel(),
        desires=["Prepare a concise project status report"],
        intentions=["Inspect the available project information"],
    )

    await agent.bdi_cycle()


asyncio.run(main())
```

For production use, replace `TestModel` with a model supported by Pydantic AI
and install any provider-specific dependencies required by that model.

## Public API

The main agent and commonly used schemas are available from the package root:

```python
from voluntas import (
    BDI,
    BDIUsageTracker,
    Belief,
    BeliefSet,
    Desire,
    DesireStatus,
    Intention,
    Plan,
)
```

The complete schema surface is available from `voluntas.schemas`:

```python
from voluntas.schemas import (
    BeliefExtractionResult,
    HighLevelIntentionList,
    ReconsiderResult,
)
```

## BDI lifecycle

Each cycle coordinates the following stages:

1. Update beliefs from the current context and action outcomes.
2. Deliberate over pending desires and their priorities.
3. Generate a high-level intention when no active intention exists.
4. Execute one intention step using Pydantic AI tools and toolsets.
5. Reconsider the remaining plan after failed or changed work.

The framework supports MCP servers through the Pydantic AI integration passed
to `BDI`, as well as structured logs and aggregate usage tracking through
`BDIUsageTracker`.

## Development

Clone the repository and install development dependencies with uv:

```bash
uv sync --group dev
uv run lint
uv run tests
```

## License

Voluntas is released under the MIT license.
