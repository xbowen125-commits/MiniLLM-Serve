"""Model-independent execution phases for one inference request."""

from enum import Enum


class ExecutionPhase(str, Enum):
    """The current stage of a request's model execution lifecycle.

    ``WAITING`` means the request has not started. ``PREFILL`` processes the
    prompt and prepares its future KV cache. ``DECODE`` represents iterative
    autoregressive token generation. ``FINISHED`` means no further model step
    should run for the request.
    """

    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
