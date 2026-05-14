"""Retry helpers built on tenacity.

Pattern adopted from the perp-dex-tools reference (`exchanges/base.py`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

P = ParamSpec("P")
R = TypeVar("R")

_log = logging.getLogger(__name__)


def query_retry(
    default_return: Any = None,
    exception_type: type[BaseException] | tuple[type[BaseException], ...] = (Exception,),
    max_attempts: int = 5,
    min_wait: float = 0.5,
    max_wait: float = 5.0,
    reraise: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that retries on transient errors.

    Used for read-only / idempotent queries (fetching markets, positions, books).
    Do NOT wrap order placement with this — replays could double-fill.
    """

    def _on_giveup(state: RetryCallState) -> Any:
        fn_name = state.fn.__name__ if state.fn else "<unknown>"
        exc = state.outcome.exception() if state.outcome else None
        _log.warning(
            "Retry exhausted: %s after %d attempts: %r",
            fn_name, state.attempt_number, exc,
        )
        return default_return

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exception_type),
        retry_error_callback=None if reraise else _on_giveup,
        reraise=reraise,
    )
