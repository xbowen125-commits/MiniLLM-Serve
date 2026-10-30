# Runtime Benchmark Metric Definitions

Scope: Orca-inspired offline Fake Worker simulation. Virtual seconds are not GPU
or HTTP measurements. Wall-clock UTC timestamps identify artifacts only.

For request i, a = arrival, s = prefill start, f = first fake token completion,
z = terminal completion, g = generated tokens:

- Queue delay: `s - a`.
- TTFT: `f - a`, including queue and prefill. Mean TTFT is `sum(TTFT_i) / N`.
- Total latency: `z - a`. Mean latency is `sum(latency_i) / N`.
- Elapsed simulated time: `max(z_i) - min(a_i)`, including idle gaps.
- Token throughput: `sum(g_i) / elapsed`; request throughput: `N / elapsed`.
- Iteration count I: number of executed plans, including prefill-only cycles.
- Average batch size: `sum_k (D_k + P_k) / I`, where D/P are selected decode/prefill
  counts. Requests may recur across iterations. This is not active-group size
  after execution and is not duration-weighted.
- Batch utilization: `sum_j (b_j * duration_j) / (capacity * sum_j duration_j)`
  over worker phase calls j; b is call group size. Excludes idle gaps, includes
  both Prefill and Decode. Decode and Prefill are serial calls, not one GPU batch.
- Decode fairness: Jain index `sum(x_i)^2 / (N * sum(x_i^2))`, where
  `x_i = generated_tokens_i / eligible_decode_iterations_i`. If all x are zero,
  score is zero. Admission waiting is excluded; heterogeneous raw token totals
  are not compared. Serving all eligible requests each iteration normally scores
  1 even when short requests wait too long.

Virtual group cost is `base + per_request * group_size`, not a measured speedup.
Prompt token lengths are workload metadata, not tokenizer measurements; the
unchanged Fake Worker does not vary Prefill cost with prompt length.

Arrival is gated by `arrival_time <= iteration_start`. Equal arrivals preserve
input order. Future requests stay outside the waiting queue; arrivals during a
worker call wait until the next boundary. Slots freed during decode become
eligible for Prefill at the next iteration in IterationScheduler.

JSONL `timestamp` is virtual start time. `finished_request_ids` lists actual
terminal retirements this cycle; `scheduler_finished_request_ids` lists terminal
IDs newly observed at plan selection. Neither indicates only successful outcomes.

Other experiments have distinct scopes: HTTP TTFT observes first nonempty text;
real runtime timing uses wall-clock observations; scheduler research synthesizes
first-token timing from service fractions. Do not mix their metrics with these.
