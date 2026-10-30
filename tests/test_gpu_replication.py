"""Independent-session grouping and raw validity, without GPU execution."""

from copy import deepcopy
import pytest
from benchmarks.runtime.gpu_batch.replication import analyze_sessions, report
from benchmarks.runtime.gpu_batch.analysis import execution_order


def session(identity):
    runs = []
    for repeat, size in execution_order(2):
        runs.append({'repeat': repeat, 'batch_size': size, 'phase': 'measured', 'forward_call_count': 16,
            'cleanup_slots': {'allocated_slots': 0}, 'generated_tokens': 16 * size,
            'tokens_per_second': 50 * size, 'latency_s': .32,
            'gpu_memory_bytes': {'peak_allocated': 100, 'peak_increment': 10},
            'requests': [dict(generated_tokens=16, ttft_s=.02, tpot_s=.02, latency_s=.32) for _ in range(size)]})
    return {'session_id': identity, 'pid': len(identity), 'started_at': 'start', 'timestamp': 'end',
            'configuration': {'repeats': 2}, 'runs': runs}


def test_separate_sessions_and_all_samples_retained():
    result = analyze_sessions([session('a'), session('bb'), session('ccc')])
    assert result['session_count'] == 3
    assert result['pooled_runs']['1']['run_count'] == 6
    assert result['sessions'][0]['batch2_over_batch1_median_ratios']['tokens_per_second'] == 2
    assert '| 3 | 2 |' in report(result)


def test_no_duplicate_identity_or_warmup_merge():
    a = session('a')
    with pytest.raises(ValueError):
        analyze_sessions([a, a])
    b = deepcopy(a)
    b['runs'][0]['phase'] = 'warmup'
    with pytest.raises(ValueError):
        analyze_sessions([b])


def test_reject_config_mismatch_and_invalid_forward_count():
    a, b = session('a'), session('b')
    b['model_revision'] = 'different'
    with pytest.raises(ValueError):
        analyze_sessions([a, b])
    a['runs'][0]['forward_call_count'] = 0
    with pytest.raises(ValueError):
        analyze_sessions([a])
