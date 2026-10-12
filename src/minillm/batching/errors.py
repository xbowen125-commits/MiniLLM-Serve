"""Error categories for model-independent batch execution."""


class RequestExecutionError(RuntimeError):
    """An error isolated to one request in an otherwise valid batch step."""


class BatchWorkerError(RuntimeError):
    """A worker-level failure that invalidates the entire batch step."""
