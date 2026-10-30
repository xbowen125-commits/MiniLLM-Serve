"""Sequential independent-process replication; explicit inputs, no glob merging."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .analysis import summarize, distribution


def analyze_sessions(sessions):
    if not sessions or len({s['session_id'] for s in sessions}) != len(sessions):
        raise ValueError('sessions must have distinct identities')
    reference = sessions[0]
    fixed = ('model', 'model_revision', 'dtype', 'torch', 'cuda', 'transformers', 'gpu',
             'python', 'git_sha', 'benchmark_source_sha256', 'prompt_token_count', 'prompt_token_sha256', 'configuration')
    for session in sessions:
        if any(session.get(k) != reference.get(k) for k in fixed):
            raise ValueError('inconsistent experiment configuration or source')
        repeats = session['configuration']['repeats']
        from .analysis import execution_order
        if [(r['repeat'], r['batch_size']) for r in session['runs']] != execution_order(repeats):
            raise ValueError('measured run order/count mismatch')
        for r in session['runs']:
            if (r['phase'] != 'measured' or r['forward_call_count'] != 16
                    or r['cleanup_slots']['allocated_slots'] != 0
                    or any(q['generated_tokens'] != 16 for q in r['requests'])
                    or len(r['requests']) != r['batch_size']):
                raise ValueError('run validity checks failed; retain and investigate artifact')
    details = []
    for session in sessions:
        summary = summarize(session['runs'])
        for size in (1, 2):
            selected = [r for r in session['runs'] if r['batch_size'] == size]
            summary[str(size)]['peak_allocated_bytes'] = distribution(r['gpu_memory_bytes']['peak_allocated'] for r in selected)
            summary[str(size)]['peak_increment_bytes'] = distribution(r['gpu_memory_bytes']['peak_increment'] for r in selected)
        ratios = {name: summary['2'][name]['median'] / summary['1'][name]['median']
                  for name in ('tokens_per_second', 'request_ttft_s', 'request_tpot_s', 'request_latency_s')}
        details.append({'session_id': session['session_id'], 'pid': session['pid'],
                        'started_at': session['started_at'], 'ended_at': session['timestamp'],
                        'summary': summary, 'batch2_over_batch1_median_ratios': ratios})
    runs = [r for s in sessions for r in s['runs']]
    return {'session_count': len(sessions), 'sessions': details, 'pooled_runs': summarize(runs),
            'session_median_throughput': {str(size): distribution(
                d['summary'][str(size)]['tokens_per_second']['median'] for d in details) for size in (1, 2)}}


def report(analysis):
    lines = ['# Cross-session GPU replication', '',
        'Independent Python processes, sequential on one machine; no restart/day-to-day independence is claimed.', '',
        '| Session | Batch | Mean tokens/s | Median | SD | Min | Max | P95 | Median TTFT ms | Median TPOT ms | Median latency ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for i, session in enumerate(analysis['sessions'], 1):
        for size in ('1', '2'):
            s = session['summary'][size]
            t = s['tokens_per_second']
            lines.append(f'| {i} | {size} | ' + ' | '.join(f'{t[k]:.6f}' for k in
                ('mean', 'median', 'stddev', 'min', 'max', 'p95')) + ' | ' + ' | '.join(
                f"{s[k]['median'] * 1000:.6f}" for k in ('request_ttft_s', 'request_tpot_s', 'request_latency_s')) + ' |')
    lines.extend(['', '## Complete summaries and memory', '', '```json', json.dumps(analysis, indent=2), '```', '',
        'Pooled run summaries do not turn dependent requests into independent sessions. Warmups are excluded.',
        'All raw measured rows are retained. Boundary telemetry and CUDA events cannot establish the cause of historical stratification.', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-slow', action='store_true')
    parser.add_argument('--sessions', type=int, default=3)
    parser.add_argument('--existing-session', type=Path, action='append', default=[],
                        help='explicit complete session JSON to retain; total count includes these')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmarks/runtime/results/gpu_batch/replication_' +
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')))
    args = parser.parse_args()
    if not args.run_slow or not 3 <= args.sessions <= 6 or len(args.existing_session) >= args.sessions:
        parser.error('--run-slow and 3..6 sessions required')
    from huggingface_hub import snapshot_download
    revision = Path(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True)).name
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    loaded, manifest = [], []
    for existing in args.existing_session:
        data = json.loads(existing.read_text(encoding='utf-8'))
        journal = [json.loads(line) for line in existing.with_suffix('.jsonl').read_text(encoding='utf-8').splitlines()]
        if journal != data['warmup_runs'] + data['runs'] or data['model_revision'] != revision:
            raise ValueError('existing session journal or revision mismatch')
        loaded.append(data)
        manifest.append({'session': len(loaded), 'pid': data['pid'], 'existing_file': existing.as_posix(),
                         'sha256': hashlib.sha256(existing.read_bytes()).hexdigest()})
    for index in range(len(loaded) + 1, args.sessions + 1):
        path = args.output_dir.resolve() / f'session_{index}.json'
        command = [sys.executable, '-m', 'benchmarks.runtime.gpu_batch.benchmark', '--run-slow',
            '--revision', revision, '--max-tokens', '16', '--warmups', '1', '--repeats', '10',
            '--diagnostics', '--output', str(path)]
        print(f'Starting independent session {index}/{args.sessions}', flush=True)
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8', errors='replace')
        output, _ = process.communicate()
        (args.output_dir / f'session_{index}.log').write_text(output, encoding='utf-8')
        manifest.append({'session': index, 'pid': process.pid, 'exit_code': process.returncode,
                         'file': path.name, 'model_revision': revision})
        (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if process.returncode:
            raise RuntimeError(f'session {index} failed; retained log and partial records')
        data = json.loads(path.read_text(encoding='utf-8'))
        journal = [json.loads(line) for line in path.with_suffix('.jsonl').read_text(encoding='utf-8').splitlines()]
        if journal != data['warmup_runs'] + data['runs']:
            raise RuntimeError('journal mismatch')
        # Windows venv redirectors can spawn the real interpreter with another PID.
        manifest[-1]['launcher_pid'] = process.pid
        manifest[-1]['pid'] = data['pid']
        loaded.append(data)
        manifest[-1]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"Completed session {index}, python_pid={data['pid']}, launcher_pid={process.pid}", flush=True)
    analysis = analyze_sessions(loaded)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (args.output_dir / 'summary.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
    (args.output_dir / 'summary.md').write_text(report(analysis), encoding='utf-8')
    print(args.output_dir, flush=True)


if __name__ == '__main__':
    main()
