"""Opt-in, process-local read-only hooks for correctness trace collection.

Never install in the ordinary HTTP host. Hooks observe original object IDs and
return original results/exceptions without controlling scheduling or generation.
No tensors, prompts or generated text are retained by the collector.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
import weakref

from .artifacts import Journal, write_new

VERSION = 1


def result_view(results):
    if results is None:
        return None
    if not isinstance(results, (list, tuple)):
        return [{"unobservable_result": type(results).__name__, "reason": "non-list result not consumed by observer"}]
    rows = []
    for result in results:
        try:
            rows.append({"request_id": result.request_id, "phase": result.phase.value,
                "has_token": result.token_id is not None, "finished": result.finished,
                "error": str(result.error) if result.error else None})
        except Exception as error:
            rows.append({"unobservable_result": type(result).__name__, "reason": str(error)})
    return rows


def cache_view(cache):
    if cache is None:
        return {"length": None, "layers": None, "source": "no cache reference"}
    try:
        layers = [{"keys": list(layer.keys.shape), "values": list(layer.values.shape)}
                  for layer in cache.layers if getattr(layer, "keys", None) is not None]
        getter = getattr(cache, "get_seq_length", None)
        length = int(getter()) if getter else (layers[0]["keys"][-2] if layers else 0)
        return {"length": length, "layers": layers,
                "source": "get_seq_length" if getter else "layer0.keys.shape[-2] or observed empty layers"}
    except Exception as error:
        return {"length": None, "layers": None, "source": "unobservable: " + str(error)}


class RuntimeTrace:
    def __init__(self, directory, identity):
        self.directory, self.identity = directory, dict(identity)
        self.journal = Journal(directory / "runtime-trace.jsonl")
        self.local = threading.local()
        self.sequence = 0
        self.errors = []
        self.patches = []
        self.iterations = weakref.WeakKeyDictionary()
        self.iteration_sequence = 0
        self.lock = threading.RLock()

    def emit(self, kind, **fields):
        # Instrumentation failure must not cancel a request or change a result.
        # It instead invalidates the complete trace proof in trace-end.json.
        try:
            with self.lock:
                self.sequence += 1
                self.journal.append({**self.identity, "schema_version": VERSION,
                    "collector": "runtime-boundaries-v1", "pid": os.getpid(),
                    "thread_id": threading.get_ident(),
                    "seq": self.sequence, "event_id": uuid.uuid4().hex,
                    "iteration_id": getattr(self.local, "iteration", None),
                    "execution_stage": ("warmup" if (self.directory / "warmup-active").exists() and not
                        (self.directory / "warmup-finished").exists() else
                        "fault" if (self.directory / "fault-active").exists() else "correctness"),
                    "monotonic_s": time.perf_counter(), "kind": kind, **fields})
        except Exception as error:
            self.errors.append(type(error).__name__ + ":" + str(error))

    def worker_view(self, worker):
        try:
            rows = {}
            for rid, slot in worker.pool.snapshot()["request_slots"].items():
                cache = worker.pool.lookup(rid)
                session = worker._sessions.get(rid)
                retirement = getattr(worker, "_retiring", {}).get(rid)
                rows[rid] = {"slot": slot, "cache": cache_view(cache),
                    "cache_object_identity": id(cache),
                    "context_matches_cache": bool(session and session.context.request.kv_cache_ref is cache),
                    "context_request_id": session.context.request_id if session else None,
                    "retirement_matches_cache": bool(retirement and retirement.request is not None and
                        retirement.request.kv_cache_ref is cache),
                    "pending_first": session.pending_first if session else None,
                    "mask_shape": list(session.mask.shape) if session else None}
            return {"slots": rows, "capacity": worker.pool.capacity,
                "session_request_ids": list(worker._sessions),
                "retirements": {rid: {"request_object_identity": id(record.request) if record.request is not None else None,
                    "attempts": record.attempts, "slot_released": record.slot_released,
                    "reference_cleared": record.reference_cleared, "last_error": record.last_error,
                    "observation_error": record.observation_error}
                    for rid, record in getattr(worker, "_retiring", {}).items()}}
        except Exception as error:
            self.errors.append("worker snapshot:" + str(error))
            return {"slots": None, "unobservable": str(error)}

    def runtime_view(self, runtime):
        try:
            return {"requests": {rid: {"status": c.request.status.value,
                "phase": c.decode_state.execution_phase.value,
                "generated": len(c.request.generated_tokens), "max_tokens": c.request.generation_config.max_tokens,
                "prompt_tokens": c.decode_state.prefill_tokens,
                "created_time": c.request.created_time, "started_time": c.request.started_time,
                "finished_time": c.request.finished_time,
                "request_object_identity": id(c.request), "has_cache_ref": c.request.kv_cache_ref is not None}
                for rid, c in runtime._coordinator.contexts.items()},
                "worker": self.worker_view(runtime._worker),
                "runtime_identity": id(runtime), "owns_worker": runtime._owns_worker}
        except Exception as error:
            self.errors.append("runtime snapshot:" + str(error))
            return {"requests": None, "unobservable": str(error)}

    def patch(self, cls, name, replacement):
        original = getattr(cls, name)
        self.patches.append((cls, name, original))
        setattr(cls, name, replacement(original))

    def install(self):
        from minillm.runtime.facade import RuntimeFacade
        from minillm.continuous.iteration_scheduler import IterationScheduler
        from minillm.continuous.executor import BatchExecutor
        from minillm.workers.batched_transformers_worker import BatchedTransformersWorker
        from minillm.workers.transformers_batch_worker import TransformersBatchWorker
        from minillm.continuous.inputs import BatchInputBuilder
        from minillm.memory.slot_kv_pool import SlotKVPool
        from minillm.runtime.http_owner import ContinuousHTTPOwner
        from minillm.scheduler import SchedulerLease, RequestScheduler
        trace = self

        def grant(original):
            def call(scheduler, *args, **kwargs):
                lease = original(scheduler, *args, **kwargs)
                trace.emit("lease_granted", request_id=lease.request_state.request_id,
                    request_object_identity=id(lease.request_state), lease_id=lease.request_id,
                    running_lease_ids=list(scheduler._running))
                return lease
            return call
        self.patch(RequestScheduler, "_grant_locked", grant)

        def owner_publish(original):
            def call(owner, runtime, worker, pending):
                result = original(owner, runtime, worker, pending)
                trace.emit("owner_publish", owner_identity=id(owner),
                    owner_thread=threading.get_ident(), closing=owner._closing.is_set(),
                    status=owner.status(), snapshot=trace.runtime_view(runtime),
                    deliveries={rid: {"request_object_identity": id(d.state),
                        "terminal": d.state.is_terminal, "completion_done": d.future.done()}
                        for rid, d in pending.items()})
                return result
            return call
        self.patch(ContinuousHTTPOwner, "_publish", owner_publish)

        def lease_release(original):
            async def call(lease, outcome):
                error = None
                trace.emit("lease_release_begin", request_id=lease.request_state.request_id,
                    request_object_identity=id(lease.request_state), lease_id=lease.request_id,
                    released=lease._released, scheduler=await lease._scheduler.snapshot(),
                    running_lease_ids=list(lease._scheduler._running))
                try:
                    return await original(lease, outcome)
                except BaseException as exc:
                    error = str(exc)
                    raise
                finally:
                    trace.emit("lease_release_end", request_id=lease.request_state.request_id,
                        request_object_identity=id(lease.request_state), lease_id=lease.request_id,
                        released=lease._released, outcome=outcome, error=error,
                        scheduler=await lease._scheduler.snapshot(),
                        running_lease_ids=list(lease._scheduler._running))
            return call
        self.patch(SchedulerLease, "release", lease_release)

        def worker_close(original):
            def call(worker):
                runtime = getattr(trace.local, "runtime", None)
                fields = {"runtime_identity": id(runtime) if runtime else None,
                    "owns_worker": runtime._owns_worker if runtime else None}
                trace.emit("worker_close_begin", **fields, snapshot=trace.worker_view(worker))
                error = None
                try:
                    return original(worker)
                except BaseException as exc:
                    error = str(exc)
                    raise
                finally:
                    trace.emit("worker_close_end", **fields, error=error, snapshot=trace.worker_view(worker))
            return call
        self.patch(TransformersBatchWorker, "close", worker_close)

        def facade_action(name):
            def decorate(original):
                def call(runtime, *args, **kwargs):
                    if runtime._continuous is None:
                        return original(runtime, *args, **kwargs)
                    old_runtime = getattr(trace.local, "runtime", None)
                    old_iteration = getattr(trace.local, "iteration", None)
                    trace.local.runtime = runtime
                    if name == "step":
                        trace.iterations[runtime] = trace.iterations.get(runtime, 0) + 1
                        # Collector-scoped ID avoids collisions between two
                        # borrowed Facades; scheduler IDs remain separately observed.
                        with trace.lock:
                            trace.iteration_sequence += 1
                            trace.local.iteration = trace.iteration_sequence
                    before = trace.runtime_view(runtime)
                    # Stack-local references survive registry removal for observation
                    # only. They are never retained in the collector or used to act.
                    watched = {key: ctx.request for key, ctx in runtime._coordinator.contexts.items()}
                    rid = args[0] if name in {"cancel", "acknowledge"} and args else None
                    operation_id = uuid.uuid4().hex
                    trace.emit(name + "_begin", operation_id=operation_id, request_id=rid, snapshot=before)
                    error = None
                    try:
                        result = original(runtime, *args, **kwargs)
                        if name == "submit":
                            rid = result.request_id
                        return result
                    except BaseException as exc:
                        error = {"type": type(exc).__name__, "message": str(exc)}
                        raise
                    finally:
                        trace.emit(name + "_end", operation_id=operation_id, request_id=rid, error=error, snapshot=trace.runtime_view(runtime))
                        if name in {"cancel", "acknowledge", "close"}:
                            trace.emit("retirement_evidence", action=name, error=error,
                                requests={key: {"status": req.status.value,
                                    "request_object_identity": id(req),
                                    "has_cache_ref": req.kv_cache_ref is not None}
                                    for key, req in watched.items()})
                        trace.local.runtime, trace.local.iteration = old_runtime, old_iteration
                return call
            return decorate
        for name in ("submit", "step", "cancel", "acknowledge", "close"):
            self.patch(RuntimeFacade, name, facade_action(name))

        def plan_event(original):
            def call(scheduler, batch):
                result = original(scheduler, batch)
                def member(context):
                    return {"request_id": context.request_id, "request_object_identity": id(context.request),
                        "status": context.request.status.value, "phase": context.decode_state.execution_phase.value,
                        "created_time": context.request.created_time}
                trace.emit("scheduler_plan", scheduler_iteration_id=scheduler.last_state.iteration_id,
                    prefill=[c.request_id for c in result.prefill_requests],
                    decode=[c.request_id for c in result.decode_requests],
                    selection_input={"active": [member(c) for c in batch.active_requests.values()],
                        "waiting": [member(c) for c in batch.waiting_requests.values()], "max_active": scheduler.max_active})
                return result
            return call
        self.patch(IterationScheduler, "build_plan", plan_event)

        def execute_plan(original):
            def call(executor, plan):
                worker = executor.batch_worker
                trace.emit("executor_plan", prefill=[c.request_id for c in plan.prefill_requests],
                    decode=[c.request_id for c in plan.decode_requests],
                    free_slots=worker.pool.capacity - worker.pool.snapshot()["allocated_slots"])
                return original(executor, plan)
            return call
        self.patch(BatchExecutor, "execute_plan", execute_plan)

        def phase_call(original):
            def call(executor, contexts, phase):
                trace.local.phase = phase.value
                trace.local.phase_id = uuid.uuid4().hex
                trace.emit("executor_submit", phase=phase.value, phase_id=trace.local.phase_id,
                    members=[c.request_id for c in contexts],
                    snapshot=trace.runtime_view(trace.local.runtime) if getattr(trace.local, "runtime", None) else None)
                error = None
                try:
                    return original(executor, contexts, phase)
                except BaseException as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
                    raise
                finally:
                    runtime = getattr(trace.local, "runtime", None)
                    trace.emit("executor_phase_end", phase=phase.value, phase_id=trace.local.phase_id, error=error,
                        snapshot=trace.runtime_view(runtime) if runtime else None)
            return call
        self.patch(BatchExecutor, "_execute_batch_phase", phase_call)

        def worker_call(name):
            def decorate(original):
                def call(worker, contexts):
                    previous = getattr(trace.local, "worker", None)
                    trace.local.worker = worker
                    trace.local.members = None
                    phase_id = getattr(trace.local, "phase_id", None)
                    trace.emit("worker_begin", phase=name, phase_id=phase_id,
                        members=[c.request_id for c in contexts], snapshot=trace.worker_view(worker))
                    results, error = None, None
                    try:
                        results = original(worker, contexts)
                        return results
                    except BaseException as exc:
                        error = {"type": type(exc).__name__, "message": str(exc)}
                        raise
                    finally:
                        trace.emit("worker_end", phase=name, phase_id=phase_id, error=error,
                            results=result_view(results),
                            snapshot=trace.worker_view(worker))
                        trace.local.worker = previous
                return call
            return decorate
        self.patch(BatchedTransformersWorker, "prefill_batch", worker_call("prefill"))
        self.patch(BatchedTransformersWorker, "decode_batch", worker_call("decode"))

        def build_inputs(original):
            def call(builder, contexts):
                result = original(builder, contexts)
                if getattr(trace.local, "worker", None) is not None:
                    trace.local.members = list(result.request_ids)
                    trace.emit("prefill_inputs", members=list(result.request_ids),
                        prompt_lengths=result.attention_mask.sum(dim=1).tolist(),
                        phase_id=getattr(trace.local, "phase_id", None))
                return result
            return call
        self.patch(BatchInputBuilder, "build_prefill", build_inputs)

        def merge(original):
            def call(worker, caches):
                if getattr(trace.local, "worker", None) is not worker:
                    return original(worker, caches)
                mapping = worker.pool.snapshot()["request_slots"]
                owners = [[rid for rid in mapping if worker.pool.lookup(rid) is cache] for cache in caches]
                trace.local.members = [row[0] if len(row) == 1 else None for row in owners]
                return original(worker, caches)
            return call
        self.patch(TransformersBatchWorker, "_merge", merge)

        def forward(original):
            def call(worker, ids, mask, cache):
                if getattr(trace.local, "worker", None) is not worker:
                    return original(worker, ids, mask, cache)
                fid = uuid.uuid4().hex
                fields = {"forward_id": fid, "phase_id": getattr(trace.local, "phase_id", None),
                    "phase": getattr(trace.local, "phase", None), "members": trace.local.members,
                    "input_shape": list(ids.shape), "mask_shape": list(mask.shape),
                    "device": ids.device.type, "cache_before": cache_view(cache),
                    "snapshot": trace.worker_view(worker)}
                trace.emit("forward_begin", **fields)
                output, error = None, None
                try:
                    output = original(worker, ids, mask, cache)
                    return output
                except BaseException as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
                    raise
                finally:
                    trace.emit("forward_end", forward_id=fid, error=error,
                        cache_after=cache_view(output.past_key_values) if output is not None else None)
            return call
        self.patch(TransformersBatchWorker, "_forward", forward)

        def pool_action(name):
            def decorate(original):
                def call(pool, rid, *args):
                    before = pool.snapshot()["request_slots"].get(rid)
                    error = None
                    try:
                        return original(pool, rid, *args)
                    except BaseException as exc:
                        error = str(exc)
                        raise
                    finally:
                        trace.emit("pool_" + name, request_id=rid, before_slot=before,
                            after_slot=pool.snapshot()["request_slots"].get(rid), error=error,
                            slots=pool.snapshot()["request_slots"])
                return call
            return decorate
        for name in ("allocate", "update_request_cache", "release"):
            self.patch(SlotKVPool, name, pool_action(name))

        def release(original):
            def call(worker, rid):
                error = None
                runtime = getattr(trace.local, "runtime", None)
                ctx = runtime._coordinator.contexts.get(rid) if runtime is not None else None
                record = getattr(worker, "_retiring", {}).get(rid)
                session = worker._sessions.get(rid)
                request = (ctx.request if ctx is not None else record.request if record is not None else
                           session.context.request if session is not None else None)
                identity = id(request) if request is not None else None
                release_id = uuid.uuid4().hex
                trace.emit("release_begin", request_id=rid, snapshot=trace.worker_view(worker),
                    retirement_version=2, request_object_identity=identity, release_id=release_id)
                try:
                    return original(worker, rid)
                except BaseException as exc:
                    error = str(exc)
                    raise
                finally:
                    trace.emit("release_end", request_id=rid, error=error, snapshot=trace.worker_view(worker),
                        retirement_version=2, request_object_identity=identity,
                        release_id=release_id,
                        request_cache_ref=request.kv_cache_ref is not None if request is not None else None)
            return call
        self.patch(BatchedTransformersWorker, "release_request", release)
        return self

    def close(self):
        for cls, name, original in reversed(self.patches):
            setattr(cls, name, original)
        self.patches.clear()
        self.emit("collector_end", errors=list(self.errors))
        self.journal.close()
        write_new(self.directory / "trace-end.json", {**self.identity, "schema_version": VERSION,
            "pid": os.getpid(), "events": self.sequence, "errors": self.errors})
