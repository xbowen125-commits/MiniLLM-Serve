"""Show existing runtime ownership, continuous admission and simulated KV pages."""

import json
from dataclasses import asdict
from typing import Callable

from minillm.contracts import ExecutionPhase, PrefillStepResult, DecodeStepResult
from minillm.continuous import (
    BatchExecutor, BatchScheduler, ContinuousBatchStepResult, PagedContinuousBatchWorker,
)
from minillm.engine import GenerationConfig, RequestState
from minillm.memory import PagedKVCacheManager
from minillm.runtime import RuntimeFacade, RuntimeRequestContext


class FakeWorker:
    """Deterministic token IDs; no tokenizer, tensors, model, or GPU calls.

    The synthetic prompt represents exactly two logical tokens. Only the
    coordinator applies returned tokens to authoritative request state.
    """

    def prefill(self, requests):
        return [PrefillStepResult(r.request_id, 2, False, False) for r in requests]

    def decode(self, requests):
        return [DecodeStepResult(r.request_id, token_id=100 + len(r.generated_tokens))
                for r in requests]


class FakeBatchWorker:
    """Same token sequence under the continuous batch protocol."""

    def __init__(self):
        self.released = []

    def prefill_batch(self, requests):
        return [ContinuousBatchStepResult(c.request_id, ExecutionPhase.PREFILL)
                for c in requests]

    def decode_batch(self, requests):
        return [ContinuousBatchStepResult(c.request_id, ExecutionPhase.DECODE,
                token_id=100 + len(c.request.generated_tokens)) for c in requests]

    def release_request(self, request_id):
        self.released.append(request_id)


def run_demo(emit: Callable[[str], None] = print) -> dict:
    """Run two bounded demonstrations and return final observations for tests."""
    def log(event, **fields):
        emit(json.dumps({"event": event, **fields}, sort_keys=True))

    seen = {}

    def observe(context):
        # PREFILL can be transient within one synchronous step. Report actual
        # recorded history, not an invented separately observable execution step.
        history = context.observations.get("phase_history", [ExecutionPhase.WAITING])
        offset = seen.get(context.request_id, 0)
        for phase in history[offset:]:
            log("phase_history", request_id=context.request_id, phase=phase.name)
        seen[context.request_id] = len(history)
        log("request_state", request_id=context.request_id,
            phase=context.decode_state.execution_phase.name,
            status=context.request.status.name,
            generated_tokens=len(context.request.generated_tokens))

    log("scope", description="offline Fake Worker; simulated KV; no model or GPU")
    facade = RuntimeFacade(FakeWorker())
    try:
        context = facade.submit("synthetic prompt", GenerationConfig(max_tokens=2), request_id="facade_A")
        log("submit", request_id=context.request_id, runtime="RuntimeFacade")
        observe(context)
        for _ in range(3):
            selected = facade.step()
            log("facade_step", selected=list(selected))
            observe(context)
            log("runtime_snapshot", runtime="RuntimeFacade", **asdict(facade.snapshot()))
        result = facade.get_result(context.request_id)
        if not context.request.is_terminal:
            raise RuntimeError("facade exceeded demo step budget")
        facade_summary = {"status": result.status.name, "generated_tokens": len(result.generated_tokens)}
    finally:
        facade.close()

    manager = PagedKVCacheManager(64, tokens_per_block=2, bytes_per_token=4)
    backend = FakeBatchWorker()
    worker = PagedContinuousBatchWorker(backend, manager, prompt_token_count=lambda c: 2)
    executor = BatchExecutor(worker, scheduler=BatchScheduler(max_active=2, max_waiting=4))
    contexts = []

    def add(request_id, tokens):
        context = RuntimeRequestContext(RequestState(request_id=request_id, prompt="synthetic prompt",
            generation_config=GenerationConfig(max_tokens=tokens)))
        contexts.append(context)
        executor.add(context)
        log("submit", request_id=request_id, runtime="ContinuousBatch")
        observe(context)

    trace = []
    try:
        add("batch_A", 3)
        for cycle in range(6):
            if cycle == 1:
                # A remains active while B arrives and receives prefill.
                add("batch_B", 2)
            step = executor.step()
            record = {"cycle": cycle, "decoded": list(step.decoded),
                      "prefilled": list(step.prefilled), "retired": list(step.retired)}
            trace.append(record)
            log("continuous_step", **record)
            for context in contexts:
                observe(context)
            snapshot = worker.snapshot()
            log("kv_snapshot", allocated_bytes=snapshot.memory.allocated_bytes,
                block_count=snapshot.memory.allocated_blocks,
                request_tokens=dict(snapshot.request_tokens),
                block_tables={r: dict(table) for r, table in snapshot.block_table.items()})
            log("runtime_snapshot", runtime="ContinuousBatch", **asdict(executor.coordinator.snapshot()))
            if len(contexts) == 2 and all(c.request.is_terminal for c in contexts):
                break
        else:
            raise RuntimeError("continuous runtime exceeded demo step budget")
        continuous_summary = {c.request_id: {"status": c.request.status.name,
            "generated_tokens": len(c.request.generated_tokens)} for c in contexts}
    finally:
        executor.close()
        worker.close()
    final_runtime = asdict(executor.coordinator.snapshot())
    final_kv = worker.snapshot()
    log("cleanup", active_requests=final_runtime["active_requests"],
        allocated_bytes=final_kv.memory.allocated_bytes,
        registered_requests=len(final_kv.request_tokens))
    return {"facade": facade_summary, "continuous": continuous_summary, "steps": trace,
            "runtime_snapshot": final_runtime, "kv_allocated_bytes": final_kv.memory.allocated_bytes,
            "kv_registered_requests": len(final_kv.request_tokens), "released": backend.released}


def main():
    run_demo()


if __name__ == "__main__":
    main()
