"""Bounded slots owning real cache references, not paged GPU allocations."""

import heapq


class SlotKVPool:
    """Serial worker-owned request -> slot -> cache reference registry.

    Capacity is a request count, not a byte budget. Releasing drops references;
    PyTorch's allocator may retain freed device memory. No tensors are shared
    between requests by the pool and no GPU memory is allocated here.
    """

    def __init__(self, capacity: int = 4):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._free = list(range(capacity))
        self._slots = {}

    def allocate(self, request_id, cache):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id required")
        if request_id in self._slots:
            raise ValueError("request already owns a slot")
        if not self._free:
            raise MemoryError("KV slots exhausted")
        slot = heapq.heappop(self._free)
        self._slots[request_id] = (slot, cache)
        return slot

    def lookup(self, request_id):
        return self._slots[request_id][1]

    def slot_id(self, request_id):
        return self._slots[request_id][0]

    def release(self, request_id):
        entry = self._slots.pop(request_id, None)
        if entry is not None:
            heapq.heappush(self._free, entry[0])

    def snapshot(self):
        return {"capacity": self.capacity, "allocated_slots": len(self._slots),
                "request_slots": {r: entry[0] for r, entry in self._slots.items()}}

    # Existing CPU batch algorithm's cache adapter interface.
    def add_request_cache(self, request_id, cache):
        self.allocate(request_id, cache)

    def get_batch_kv(self, request_ids):
        return tuple(self.lookup(r) for r in request_ids)

    def update_request_cache(self, request_id, cache):
        slot = self.slot_id(request_id)
        self._slots[request_id] = (slot, cache)

    def release_request(self, request_id):
        self.release(request_id)
