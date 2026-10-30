"""Facade-to-batched-backend lifecycle and ownership, with CPU fake tensors."""

import threading
import time

import pytest

from minillm.runtime import RuntimeFacade
from minillm.engine import GenerationConfig, RequestStatus
from minillm.workers import BatchedTransformersWorker, LoadedModel
from minillm.continuous import IterationScheduler, ContinuousBatchStepResult
from test_transformers_batch_worker import FakeModel, FakeTokenizer, FakeCache


def backend(cls=BatchedTransformersWorker, capacity=2):
    return cls(LoadedModel(FakeModel(), FakeTokenizer()), slot_capacity=capacity,
               cache_factory=lambda _: FakeCache())


def runtime(worker, **options):
    return RuntimeFacade(worker, max_batch_size=2, execution_mode="continuous", **options)


def test_complete_chain_and_metrics():
    w = backend()
    r = runtime(w)
    a = r.submit('short', GenerationConfig(max_tokens=2, temperature=0), request_id='a')
    b = r.submit('long', GenerationConfig(max_tokens=3, temperature=0), request_id='b')
    try:
        assert r._coordinator is r._continuous.coordinator
        r.step()
        assert a.request.started_time >= a.request.created_time
        assert a.decode_state.first_token_time is None
        assert r.get_result('a').metrics.prefill_tokens == 2
        assert r.get_result('b').metrics.prefill_tokens == 3
        assert w.pool.snapshot()['allocated_slots'] == 2
        r.step(); r.step()
        assert a.request.generated_tokens == [3, 4]
        assert b.request.generated_tokens == [4, 5]
        assert w.pool.snapshot()['allocated_slots'] == 1
        r.step()
        assert b.request.generated_tokens == [4, 5, 6]
        assert w.pool.snapshot()['allocated_slots'] == 0
        assert r.snapshot().active_requests == 0
        assert r.get_result('a').metrics.arrival_ttft_ms >= r.get_result('a').metrics.first_token_latency_ms
        assert r.get_result('a').metrics.tpot_ms is not None
        assert r._continuous.scheduler.last_state.iteration_id == 4
    finally:
        r.close(); w.close()


@pytest.mark.parametrize('steps', [0, 1, 2])
def test_cancel_waiting_prefilled_decoding(steps):
    w = backend(); r = runtime(w)
    r.submit('short', request_id='a'); r.submit('long', request_id='b')
    try:
        for _ in range(steps):
            r.step()
        assert r.cancel('a')
        assert not r.cancel('a')
        assert r.get_result('a').status is RequestStatus.CANCELLED
        assert w.pool.snapshot()['allocated_slots'] == (1 if steps else 0)
        assert r.snapshot().active_requests == 1
    finally:
        r.close(); r.close(); w.close()
    assert w.pool.snapshot()['allocated_slots'] == 0


@pytest.mark.parametrize('bad_phase', ['prefill', 'decode'])
@pytest.mark.parametrize('violation', ['missing', 'duplicate', 'unknown'])
def test_invalid_result_sets_fail_and_release(bad_phase, violation):
    class Bad(BatchedTransformersWorker):
        def corrupt(self, results):
            if violation == 'missing':
                return results[:1]
            if violation == 'duplicate':
                return [results[0], results[0]]
            from dataclasses import replace
            return [replace(results[0], request_id='unknown', token_result=None), results[1]]
        def prefill_batch(self, contexts):
            results = super().prefill_batch(contexts)
            return self.corrupt(results) if bad_phase == 'prefill' else results
        def decode_batch(self, contexts):
            results = super().decode_batch(contexts)
            return self.corrupt(results) if bad_phase == 'decode' else results
    w = backend(Bad); r = runtime(w)
    r.submit('short', request_id='a'); r.submit('long', request_id='b')
    try:
        if bad_phase == 'decode':
            r.step()
        with pytest.raises(RuntimeError):
            r.step()
        assert r.snapshot().failed_requests == 2
        assert r.snapshot().active_requests == 0
        assert w.pool.snapshot()['allocated_slots'] == 0
        assert not r._continuous.batch.waiting_requests
    finally:
        r.close(); w.close()


def test_forward_failure_and_input_error():
    w = backend(); r = runtime(w)
    r.submit('bad', request_id='bad'); r.submit('long', request_id='good')
    try:
        r.step()
        assert r.get_result('bad').status is RequestStatus.FAILED
        assert w.pool.snapshot()['allocated_slots'] == 1
        r.step()
        def failure(**kwargs):
            raise RuntimeError('forward failure')
        w.loaded.model.forward = failure
        r.step()
        assert r.get_result('good').status is RequestStatus.FAILED
        assert r.snapshot().active_requests == 0
        assert w.pool.snapshot()['allocated_slots'] == 0
    finally:
        r.close(); w.close()


def test_shared_worker_close_isolation_and_id_claims():
    w = backend(); a = runtime(w); b = runtime(w)
    try:
        a.submit('short', request_id='a'); b.submit('long', request_id='b')
        with pytest.raises(ValueError, match='claimed'):
            b.submit('short', request_id='a')
        a.step(); b.step()
        assert w.pool.snapshot()['allocated_slots'] == 2
        a.close(); a.close()
        assert w.pool.snapshot()['allocated_slots'] == 1
        assert w.pool.lookup('b') is b._coordinator.contexts['b'].request.kv_cache_ref
        b.step()
        with pytest.raises(ValueError, match='shared'):
            runtime(w, owns_worker=True)
    finally:
        a.close(); b.close(); w.close()


def test_shared_pool_admission_waits_without_failing_peers():
    w = backend(); a = runtime(w); b = runtime(w)
    try:
        a.submit('short', request_id='a1'); a.submit('long', request_id='a2')
        b.submit('short', request_id='b')
        a.step()
        assert b.step() == ()
        assert b.get_result('b').status is RequestStatus.WAITING
        assert w.pool.snapshot()['allocated_slots'] == 2
        a.cancel('a1')
        assert b.step() == ('b',)
        assert w.pool.snapshot()['allocated_slots'] == 2
        assert a.get_result('a2').status is RequestStatus.RUNNING
    finally:
        a.close(); b.close(); w.close()


def test_owned_worker_close_once_and_capacity_validation():
    class Owned(BatchedTransformersWorker):
        closes = 0
        def close(self):
            self.closes += 1
            super().close()
    w = backend(Owned)
    r = runtime(w, owns_worker=True)
    r.submit('short'); r.step()
    with pytest.raises(ValueError, match='shared'):
        runtime(w)
    r.close(); r.close()
    assert w.closes == 1
    assert w.pool.snapshot()['allocated_slots'] == 0
    with pytest.raises(ValueError, match='capacity'):
        RuntimeFacade(w, execution_mode='continuous', max_batch_size=3)
    with pytest.raises(ValueError, match='capacity'):
        runtime(backend(capacity=1))
    with pytest.raises(ValueError, match='scheduler'):
        runtime(backend(), scheduler=IterationScheduler(max_batch_size=1))


def test_cleanup_failure_preserves_execution_error_and_retries():
    class Broken(BatchedTransformersWorker):
        broken = True
        def prefill_batch(self, requests):
            super().prefill_batch(requests)
            raise ValueError('original execution error')
        def release_request(self, request_id):
            if self.broken:
                raise RuntimeError('cleanup unavailable')
            super().release_request(request_id)
    w = backend(Broken); r = runtime(w)
    c = r.submit('short', request_id='a')
    with pytest.raises(ValueError, match='original execution') as error:
        r.step()
    assert error.value.cleanup_errors
    assert c.observations['cleanup_errors']
    assert c.request.status is RequestStatus.FAILED
    assert r.snapshot().active_requests == 0
    with pytest.raises(RuntimeError, match='cleanup'):
        r.close()
    assert w.pool.snapshot()['allocated_slots'] == 1
    w.broken = False
    r.close(); r.close()
    assert w.pool.snapshot()['allocated_slots'] == 0


def test_cross_thread_cancel_waits_for_worker():
    w = backend(); r = runtime(w); r.submit('short', request_id='a')
    entered, resume = threading.Event(), threading.Event()
    original = w.loaded.model.forward
    def blocking(**kwargs):
        entered.set()
        assert resume.wait(3)
        return original(**kwargs)
    w.loaded.model.forward = blocking
    errors = []
    def step():
        try:
            r.step()
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=step); thread.start()
    assert entered.wait(3)
    cancelled = threading.Event()
    cancel = threading.Thread(target=lambda: (r.cancel('a'), cancelled.set()))
    cancel.start()
    assert not cancelled.wait(.05)
    resume.set(); thread.join(3); cancel.join(3)
    assert not errors and cancelled.is_set()
    assert w.pool.snapshot()['allocated_slots'] == 0
    r.close(); w.close()


def test_sampling_error_is_isolated(monkeypatch):
    import minillm.workers.transformers_batch_worker as module
    original = module.sample
    calls = []
    def sample(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError('sampling failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'sample', sample)
    w = backend(); r = runtime(w)
    r.submit('short', request_id='a'); r.submit('long', request_id='b')
    try:
        r.step(); r.step()
        assert r.get_result('a').status is RequestStatus.FAILED
        assert r.get_result('b').generated_tokens == (4,)
        assert r.snapshot().active_requests == 1
        assert w.pool.snapshot()['allocated_slots'] == 1
    finally:
        r.close(); w.close()


def test_whole_prefill_forward_error_with_cleanup_failure_preserved():
    class Broken(BatchedTransformersWorker):
        broken = True
        def release_request(self, request_id):
            if self.broken:
                raise RuntimeError('release failed')
            super().release_request(request_id)
    w = backend(Broken); r = runtime(w)
    a = r.submit('short', request_id='a'); b = r.submit('long', request_id='b')
    def failure(**kwargs):
        raise ValueError('original forward failed')
    w.loaded.model.forward = failure
    try:
        with pytest.raises(RuntimeError, match='cleanup'):
            r.step()
        for c in (a, b):
            assert c.request.status is RequestStatus.FAILED
            assert isinstance(c.observations['execution_error'], ValueError)
            assert c.observations['cleanup_errors']
        w.broken = False
    finally:
        w.broken = False
        r.close(); w.close()


def test_unknown_release_and_prefill_time_boundary():
    w = backend(); r = runtime(w)
    c = r.submit('short', request_id='a')
    original = w.loaded.model.forward
    def forward(**kwargs):
        assert c.request.started_time is not None
        assert c.decode_state.first_token_time is None
        return original(**kwargs)
    w.loaded.model.forward = forward
    try:
        w.release_request('unknown'); w.release_request('unknown')
        r.step()
        w.release_request('unknown')
        assert w.pool.snapshot()['allocated_slots'] == 1
        assert c.decode_state.first_token_time is None
    finally:
        r.close(); w.close()
