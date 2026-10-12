"""Offline request ownership abstractions; not connected to serving."""

from .context import DecodeState, RuntimeMetrics, RuntimeRequestContext

__all__ = ["DecodeState", "RuntimeMetrics", "RuntimeRequestContext", "RuntimeFacade"]


def __getattr__(name):
    if name == "RuntimeFacade":
        from .facade import RuntimeFacade
        return RuntimeFacade
    raise AttributeError(name)
