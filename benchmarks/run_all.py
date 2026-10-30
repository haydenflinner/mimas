#!/usr/bin/env python3
"""run_all.py — time every benchmark implementation, best-of-N and mean.

Each workload prints a checksum line; we capture it once to verify all
languages agree, then report wall-clock best/mean over N runs.
"""
import subprocess, sys, time, statistics, os

HERE = os.path.dirname(os.path.abspath(__file__))
MIM = os.path.join(HERE, 'runners', 'target', 'release', 'mimas-run')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5

BENCHES = [
    ('fib_rec', [
        ('mimas',  [MIM, 'fib_rec/fib_rec.mim']),
        ('c',      ['bin/fib_rec-c']),
        ('c++',    ['bin/fib_rec-cpp']),
        ('rust',   ['bin/fib_rec-rs']),
        ('python', ['python3', 'fib_rec/fib_rec.py']),
    ]),
    ('fib_iter', [
        ('mimas',  [MIM, 'fib_iter/fib_iter.mim']),
        ('c',      ['bin/fib_iter-c']),
        ('c++',    ['bin/fib_iter-cpp']),
        ('rust',   ['bin/fib_iter-rs']),
        ('python', ['python3', 'fib_iter/fib_iter.py']),
    ]),
    ('trips_gen', [
        ('mimas',  [MIM, 'trips_query/trips_gen.mim']),
    ]),
    ('trips_query', [
        ('mimas',  [MIM, 'trips_query/trips_query.mim']),
        ('c',      ['bin/trips_query-c']),
        ('c++',    ['bin/trips_query-cpp']),
        ('rust',   ['bin/trips_query-rs']),
        ('python', ['python3', 'trips_query/trips_query.py']),
    ]),
]

def run(cmd):
    t = time.perf_counter()
    out = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    dt = time.perf_counter() - t
    if out.returncode != 0:
        return dt, None
    return dt, out.stdout.strip()

expected = {}
for bench, impls in BENCHES:
    print(f'== {bench} ==')
    for lang, cmd in impls:
        run(cmd)  # warmup
        times, outs = [], None
        for _ in range(N):
            dt, o = run(cmd)
            times.append(dt)
            outs = o
        key = f'{bench}'
        if outs is None:
            print(f'  {lang:8s}  FAILED')
            continue
        if key in expected and expected[key] != outs:
            print(f'  {lang:8s}  OUTPUT MISMATCH: {outs[:60]!r} vs {expected[key][:60]!r}')
        else:
            expected.setdefault(key, outs)
            best, mean = min(times), statistics.mean(times)
            print(f'  {lang:8s}  best {best*1000:8.1f} ms   mean {mean*1000:8.1f} ms   ({outs[:40]})')
