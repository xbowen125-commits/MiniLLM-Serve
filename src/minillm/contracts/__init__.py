"""Shared runtime contracts independent of execution and batching."""

from .phase import ExecutionPhase
from .result import DecodeStepResult, PrefillStepResult
from .status import DecodeStateProtocol, DecodeStatus, FinishReason

__all__ = [
    "DecodeStateProtocol", "DecodeStatus", "DecodeStepResult",
    "ExecutionPhase", "FinishReason", "PrefillStepResult",
]
