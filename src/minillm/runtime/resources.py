"""Shared-worker serialization and identity claims, not request lifecycle state."""

import threading
import weakref

_bindings = weakref.WeakKeyDictionary()
_guard = threading.RLock()


class WorkerBinding:
    def __init__(self):
        self.lock = threading.RLock()
        self.clients = {}
        self.claims = {}

    def attach(self, owner, owns_worker):
        with self.lock:
            if self.clients and (owns_worker or any(self.clients.values())):
                raise ValueError("owned Worker cannot be shared")
            self.clients[owner] = owns_worker


def bind_worker(worker):
    with _guard:
        if worker not in _bindings:
            _bindings[worker] = WorkerBinding()
        return _bindings[worker]
