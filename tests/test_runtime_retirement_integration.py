"""Composition tests: real admission/Owner/Facade with CPU fake tensors."""
import threading
import asyncio

import pytest

from benchmarks.runtime.online_http.artifacts import git_snapshot
from benchmarks.runtime.online_http.runtime_trace import RuntimeTrace
from benchmarks.runtime.online_http.trace_check import seal
from test_runtime_continuous import backend, runtime
from test_http_continuous import application, asgi_request, wait_idle
from minillm.workers import BatchedTransformersWorker
from benchmarks.runtime.online_http.retirement import check_retirement
from benchmarks.runtime.online_http.retirement_composition import check_composition


@pytest.fixture
def trace_directory(tmp_path):
    source = git_snapshot()
    trace = RuntimeTrace(tmp_path, {"run_id": tmp_path.name, "config_sha256": "synthetic-composition",
        "model_revision": "fake", "dtype": "float32", "device": "cpu",
        "git": {key: source[key] for key in ("sha", "dirty", "source_tree_sha256")},
        "geometry": {"layers": 1, "kv_heads": 1, "head_dim": 1}}).install()
    try:
        yield tmp_path, trace
    finally:
        trace.close()
        result = seal(tmp_path)
        # Preserve raw evidence even when the test itself raises. Do not turn
        # old taxonomy UNKNOWN into a composition PASS automatically.
        assert result["status"] != "FAIL", result


def test_cleanup_failure_does_not_block_independent_decode(trace_directory, monkeypatch):
    worker = backend()
    facade = runtime(worker)
    permit = threading.Event()
    original = worker.pool.release
    def release(rid):
        if rid == "A" and not permit.is_set():
            raise RuntimeError("A cleanup held")
        original(rid)
    monkeypatch.setattr(worker.pool, "release", release)
    try:
        a = facade.submit("short", request_id="A")
        b = facade.submit("long", request_id="B")
        facade.step()
        with pytest.raises(Exception):
            facade.cancel("A")
        assert a.request.is_terminal and a.request.kv_cache_ref is not None
        assert worker._retiring["A"].request is a.request
        assert worker.pool.slot_id("A") != worker.pool.slot_id("B")
        facade.step()
        assert len(b.request.generated_tokens) == 1
        assert not b.request.is_terminal
    finally:
        permit.set()
        facade.close(); worker.close()


def test_peer_cancel_and_ack_do_not_release_or_wait_for_failed_peer(trace_directory, monkeypatch):
    worker = backend()
    facade = runtime(worker)
    permit = threading.Event()
    original = worker.pool.release
    def release(rid):
        if rid == "A" and not permit.is_set():
            raise RuntimeError("A cleanup held")
        original(rid)
    monkeypatch.setattr(worker.pool, "release", release)
    try:
        a = facade.submit("short", request_id="A")
        b = facade.submit("long", request_id="B")
        facade.step()
        with pytest.raises(Exception):
            facade.cancel("A")
        facade.cancel("B")
        facade.acknowledge("B")
        assert b.request.kv_cache_ref is None
        assert worker._retiring["A"].request is a.request
        assert worker.pool.lookup("A") is a.request.kv_cache_ref
        assert facade.retention()["contexts"] == 1
    finally:
        permit.set()
        facade.close(); worker.close()


async def until(predicate, timeout=5):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.002)


@pytest.mark.parametrize("mode", ["cancel", "unknown", "shutdown", "decode_error", "reference", "shutdown_reference"])
def test_owner_admission_finalization_and_retry(trace_directory, mode, monkeypatch):
    directory, trace = trace_directory
    prefilled, resume_prefill = threading.Event(), threading.Event()
    peer_decode, resume_peer = threading.Event(), threading.Event()
    attempted, permit = threading.Event(), threading.Event()
    refs, slot_at_prefill = {}, {}
    target = []
    owners = set()

    class Controlled(BatchedTransformersWorker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            original = self.pool.release
            def release(rid):
                owners.add(threading.get_ident())
                if target and rid == target[0] and not permit.is_set() and "reference" not in mode:
                    attempted.set()
                    raise RuntimeError("held target cleanup")
                original(rid)
            self.pool.release = release

        def _cache_absent(self, rid):
            if mode == "unknown" and target and rid == target[0] and not permit.is_set():
                attempted.set()
                raise RuntimeError("resource query unavailable")
            return super()._cache_absent(rid)

        def _clear_request_cache(self, request):
            if request is not None and target and request.request_id == target[0] and not permit.is_set() and "reference" in mode:
                attempted.set()
                raise RuntimeError("original reference clearing held")
            return super()._clear_request_cache(request)

        def prefill_batch(self, contexts):
            owners.add(threading.get_ident())
            result = super().prefill_batch(contexts)
            for context in contexts:
                refs[context.request_id] = context.request
                slot_at_prefill[context.request_id] = self.pool.slot_id(context.request_id)
                if not target:
                    target.append(context.request_id)
                    prefilled.set()
                    assert resume_prefill.wait(5), "prefill test barrier timed out"
            return result

        def decode_batch(self, contexts):
            owners.add(threading.get_ident())
            if any(c.request.prompt == "long" for c in contexts):
                peer_decode.set()
                assert resume_peer.wait(5), "peer test barrier timed out"
            return super().decode_batch(contexts)

    if mode == "decode_error":
        import minillm.workers.transformers_batch_worker as module
        sample = module.sample
        injected = [False]
        def sample_once(*args, **kwargs):
            if not injected[0]:
                injected[0] = True
                raise RuntimeError("request-local sampling failure")
            return sample(*args, **kwargs)
        monkeypatch.setattr(module, "sample", sample_once)

    app, evidence = application(worker_class=Controlled, request_queue_capacity=2,
        queue_timeout_seconds=10, inference_timeout_seconds=30)

    async def observe(label):
        scheduler = await app.state.request_scheduler.snapshot()
        owner = evidence["owner"]
        trace.emit("http_observation", label=label, scheduler=scheduler,
            status=owner.status(), owner_thread_alive=owner.thread.is_alive(),
            pending_finalizers=len(app.state.cleanup_tasks),
            request_evidence={rid: {"status": req.status.value,
                "request_object_identity": id(req), "has_cache_ref": req.kv_cache_ref is not None}
                for rid, req in refs.items()}, snapshot=trace.worker_view(evidence["worker"]))
        return scheduler

    async def run():
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        disconnect = asyncio.Event()
        tasks = []
        shutdown = None
        try:
            first = asyncio.create_task(asgi_request(app, max_tokens=8, disconnect=disconnect))
            tasks.append(first)
            await until(prefilled.is_set)
            leases = list(app.state.request_scheduler._running.values())
            first_lease = leases[0]
            assert first_lease.request_state is refs[target[0]]
            if mode == "decode_error":
                from minillm.batching.coordinator import DecodeCoordinator
                failed_decode_applied = asyncio.Event()
                loop = asyncio.get_running_loop()
                apply_decode_results = DecodeCoordinator.apply_decode_results

                def observe_failed_decode(coordinator, batch, results):
                    target_failure = any(request is first_lease.request_state for request in batch.requests) and any(
                        result.request_id == first_lease.request_state.request_id and result.error is not None
                        for result in results)
                    retired = apply_decode_results(coordinator, batch, results)
                    if target_failure:
                        loop.call_soon_threadsafe(failed_decode_applied.set)
                    return retired

                monkeypatch.setattr(DecodeCoordinator, "apply_decode_results", observe_failed_decode)
            if mode != "decode_error":
                disconnect.set()
                await until(first_lease.cancel_event.is_set)
            resume_prefill.set()
            await until(attempted.is_set)
            if mode == "decode_error":
                # Cleanup can be attempted before the Worker returns its result.
                # Wait for the real Coordinator application, not a pending count.
                await asyncio.wait_for(failed_decode_applied.wait(), 5)
            await until(lambda: evidence["owner"].status().get("pending") == 1)
            worker = evidence["worker"]
            original = refs[target[0]]
            assert original.is_terminal and original.kv_cache_ref is not None
            await until(lambda: target[0] in worker._retiring)
            assert worker._retiring[target[0]].request is original
            attempts = worker._retiring[target[0]].attempts
            await until(lambda: worker._retiring[target[0]].attempts >= attempts + 3)
            # Failure remains held beyond several retries; it is not a timer-based success.
            second = asyncio.create_task(asgi_request(app, prompt="long", max_tokens=3))
            tasks.append(second)
            await until(peer_decode.is_set)
            third = asyncio.create_task(asgi_request(app, max_tokens=1))
            tasks.append(third)
            async with asyncio.timeout(5):
                while (await app.state.request_scheduler.snapshot())["queued_requests"] != 1:
                    await asyncio.sleep(.002)
            before = await observe("failure-with-queued-arrival")
            assert before["running_requests"] == 2 and before["queued_requests"] == 1
            assert worker.pool.snapshot()["allocated_slots"] == (1 if "reference" in mode else 2)
            assert len(app.state.cleanup_tasks) == 2
            assert not first.done() and not first_lease._released
            assert len(slot_at_prefill) == 2
            peer_id = next(rid for rid, req in refs.items() if req.prompt == "long")
            if "reference" not in mode:
                assert slot_at_prefill[peer_id] != slot_at_prefill[target[0]]
            resume_peer.set()
            assert (await asyncio.wait_for(second, 5))[0] == 200
            assert (await asyncio.wait_for(third, 5))[0] == 200
            after_peer = await observe("peer-completed-target-still-pending")
            assert after_peer["running_requests"] == 1
            assert after_peer["queued_requests"] == 0
            if "reference" in mode:
                assert worker.pool.snapshot()["allocated_slots"] == 0
                assert worker._retiring[target[0]].slot_released is True
            else:
                assert worker.pool.slot_id(target[0]) == slot_at_prefill[target[0]]
            new_id = next(rid for rid in refs if rid not in {target[0], peer_id})
            assert slot_at_prefill[new_id] == slot_at_prefill[peer_id]
            assert refs[peer_id].kv_cache_ref is refs[new_id].kv_cache_ref is None
            assert worker._retiring[target[0]].request is original
            assert not first_lease._released and not first.done()
            if mode.startswith("shutdown"):
                shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
                evidence["owner"].close()
                await until(lambda: evidence["owner"]._closing.is_set())
                await asyncio.sleep(.03)
                assert not shutdown.done() and not evidence["owner"].stopped.done()
                await observe("shutdown-cleanup-still-unconfirmed")
            permit.set()
            status, _, _ = await asyncio.wait_for(first, 5)
            assert status == (500 if mode == "decode_error" else 503)
            await wait_idle(app)
            assert original.kv_cache_ref is None
            assert worker._retiring == {} and worker._sessions == {}
            await first_lease.release("cancelled")  # real duplicate finalization
            final = await observe("confirmed-final-drain")
            assert final["running_requests"] == final["queued_requests"] == 0
            assert not app.state.cleanup_tasks
            assert owners == {evidence["owner"].thread.ident}
            if shutdown is not None:
                await asyncio.wait_for(shutdown, 5)
                shutdown = None
                lifespan = None
            else:
                await lifespan.__aexit__(None, None, None)
                lifespan = None
            assert not evidence["owner"].thread.is_alive()
            assert evidence["owner"].stopped.done()
            await observe("owner-stopped")
        finally:
            permit.set(); resume_prefill.set(); resume_peer.set()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if shutdown is not None:
                await asyncio.wait_for(shutdown, 5)
            elif lifespan is not None:
                await lifespan.__aexit__(None, None, None)
    asyncio.run(run())
    import json
    rows = [json.loads(line) for line in (directory / "runtime-trace.jsonl").read_text(encoding="utf-8").splitlines()]
    # Collector end marker is added by fixture teardown; resource checks use
    # raw real objects and explicit final observation, not saved flags.
    result = check_composition(rows)
    assert result["status"] == "PASS", result


@pytest.mark.parametrize("owned", [False, True])
def test_worker_ownership_close_order_and_external_references(trace_directory, owned):
    directory, _ = trace_directory
    class Counted(BatchedTransformersWorker):
        closes = 0
        def close(self):
            self.closes += 1
            assert self.pool.snapshot()["allocated_slots"] == 0
            assert self._retiring == {}
            super().close()
    worker = backend(Counted)
    a = runtime(worker, owns_worker=owned)
    b = None if owned else runtime(worker)
    try:
        first = a.submit("short", request_id="A")
        a.step()
        if b:
            second = b.submit("long", request_id="B")
            b.step()
            other_cache = second.request.kv_cache_ref
        a.close(); a.close()
        assert first.request.kv_cache_ref is None
        assert worker.closes == int(owned)
        if b:
            assert worker.pool.lookup("B") is other_cache
            assert second.request.kv_cache_ref is other_cache
            b.step()
            b.close(); b.close()
            assert second.request.kv_cache_ref is None
            assert worker.closes == 0
        assert not worker._retiring and not worker._sessions
    finally:
        a.close()
        if b:
            b.close()
        if not owned:
            worker.close()
    import json
    rows = [json.loads(line) for line in (directory / "runtime-trace.jsonl").read_text(encoding="utf-8").splitlines()]
    result = check_composition(rows)
    ownership = next(c for c in result["checks"] if c["code"] == "WORKER_CLOSE_OWNERSHIP")
    assert ownership["status"] == "PASS", result
