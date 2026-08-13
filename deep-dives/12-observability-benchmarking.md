# Observability, SLOs, and Benchmarking: Knowing Your System Is Healthy Before Users Do

**Deep-dive #12 · Written August 2026 · Environment: on-prem, air-gapped, NVIDIA B200/B300, vLLM v0.26.x, agentic workloads.**
Companion to the [deployment SOP](../SOP-production-ai-model-deployment.md) (§7 Observability & SLOs) and the [PRIMER](../PRIMER.md). Everything here is self-hosted; nothing phones home.

> **Version warning up front.** Metric names, benchmark CLIs, and dashboard JSON are the fastest-rotting artifacts in this ecosystem. Several vLLM metrics were renamed between the old "v0" engine and the current "v1" engine, and guidellm's CLI changed shape between releases. Every name in this document was checked against vLLM `main` / v0.26-era sources in August 2026. When you upgrade the engine, re-verify your dashboards and alert rules against `/metrics` output — a renamed metric fails silently (the panel just goes blank).

---

## Key takeaways

- **Alert on symptoms users feel (latency, errors), graph causes (queue depth, KV pressure, GPU health).** The SOP's rule "alert on SLO burn rate, not raw utilization" is the industry-standard multi-window multi-burn-rate pattern from Google's SRE Workbook, and it maps cleanly onto vLLM's histogram metrics.
- **vLLM exposes everything you need at `/metrics` with a `vllm:` prefix** — queue gauges, TTFT/ITL histograms, token counters, KV-cache usage, prefix-cache hit counters, preemption and speculative-decoding counters. Learn ~15 metric families and you can diagnose almost any serving incident.
- **`DCGM_FI_DEV_GPU_UTIL` is nearly useless for LLM serving** — it answers "was any kernel running?" and pins at 100% while the GPU may be 20% busy. Use the profiling metrics (`DCGM_FI_PROF_GR_ENGINE_ACTIVE`, `DCGM_FI_PROF_SM_ACTIVE`, `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`) instead.
- **Prefix-cache hit rate is a first-class production signal for agent workloads.** Compute it from the `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` counters; a sudden drop after a router or prompt change is an incident, not a curiosity.
- **The four golden signals per model pool:** traffic (requests + tokens/s), latency (TTFT p99, ITL p99), errors (5xx + tool-call parse failures), saturation (WAITING queue, KV-cache %, preemptions). One Grafana dashboard per pool, same layout everywhere.
- **Benchmark with guidellm (`ghcr.io/vllm-project/guidellm`, runs fine air-gapped) and `vllm bench serve`;** NVIDIA's genai-perf has been superseded by **AIPerf**. Always warm up, control prefix-cache effects deliberately, and use token-length distributions that look like your real agent traffic.
- **Your capacity unit is "max concurrency at SLO"** per (model, profile, hardware) tuple — measured by a concurrency sweep, recorded next to the recipe, re-measured on every engine or model upgrade. GPU count is not a capacity number.
- **Infra health is not agent health.** A pool can be green on every GPU metric while agents silently fail: track tool-call parse failure rate, retry rate, task abandonment, and structured-output validity — mostly derived at the gateway, not from vLLM.

---

## 1. The enclave observability stack

### 1.1 What we run and why

In a disconnected enclave you cannot use any SaaS observability product (Datadog, Grafana Cloud, Honeycomb). The good news: the self-hosted open-source stack is mature, and it is what most of the vLLM ecosystem documents against anyway.

| Component | Role | Notes for the enclave |
|---|---|---|
| **Prometheus** | Metrics: scrapes `/metrics` endpoints on a schedule, stores time series, evaluates alert rules | One instance (or an HA pair) per enclave; 15s scrape interval is the standard default |
| **Grafana** | Dashboards over Prometheus/Loki/Tempo | Dashboard JSON lives in the config repo and ships in release bundles |
| **Alertmanager** | Routes alerts from Prometheus to humans (email/webhook/chat) | Point it at whatever paging mechanism exists inside the enclave |
| **Loki** (+ Alloy or Promtail agent) | Log aggregation with label-based indexing; queried with LogQL | The place gateway/vLLM logs become *metrics* (parse-failure rates) |
| **Tempo** (optional) + **OTel Collector** | Distributed traces (OpenTelemetry) | Worth adding once you debug multi-hop agent latency; see §7 |

Definitions: **Prometheus** is a pull-based metrics database — it fetches ("scrapes") a plain-text page of counters from each service. **PromQL** is its query language. **OTel** is **OpenTelemetry**, the vendor-neutral standard for traces (per-request timelines), metrics, and logs; **OTLP** is its wire protocol.

All of these are ordinary containers: mirror them into Harbor by digest like everything else, and keep their configuration (scrape configs, alert rules, dashboard JSON, Loki/Tempo configs) in the internal git config repo so that every bundle release pins the observability config it was tested with.

### 1.2 Scrape architecture

```
                 ┌────────────────────────── enclave ─────────────────────────┐
                 │                                                             │
  Grafana ◄──────┤  Prometheus ── scrapes every 15s ──►  vLLM replica :8000/metrics
     │           │      │                               vLLM replica :8001/metrics
     │           │      │                               gateway (LiteLLM /
  Alertmanager ◄─┤──────┘                                 agentic-api) /metrics
     │           │      ▲                               dcgm-exporter :9400/metrics
  pager/email    │      │                                 (one per GPU node)
                 │      └── also scrapes: node_exporter, Harbor, Loki, itself │
                 │                                                             │
                 │  Loki ◄── Alloy/Promtail tails container logs (gateway,    │
                 │            vLLM, transfer jobs)                             │
                 │  Tempo ◄── OTel Collector ◄── OTLP from gateway + vLLM     │
                 └─────────────────────────────────────────────────────────────┘
```

Key decisions baked into this picture:

- **Scrape every vLLM replica directly**, not just the gateway. Per-replica metrics are how you see one bad replica (cold prefix cache, ECC-degraded GPU) hiding behind a load balancer. Every vLLM API server serves Prometheus metrics at `GET /metrics` on its normal serving port — there is no separate metrics port to open.
- **One dcgm-exporter per GPU node.** **DCGM** is NVIDIA's **Data Center GPU Manager**, the health/telemetry daemon; `dcgm-exporter` translates its fields into Prometheus metrics (default port 9400). On Kubernetes the GPU Operator deploys it for you; under Docker Compose (Tier 2) run it as a per-node container with `--gpus all`.
- **Labels are your join keys.** vLLM stamps `model_name` on its metrics (the `--served-model-name`); make Prometheus relabeling add `pool` (e.g. `interactive-agent`), `node`, and `replica` labels at scrape time so dashboards can slice per pool. dcgm-exporter stamps `gpu`, `UUID`, and (on K8s) pod labels; correlating "which model was on the GPU that overheated" happens through the node/pod labels.

Minimal static scrape config for a Tier 2 (Compose) node:

```yaml
scrape_configs:
  - job_name: vllm
    scrape_interval: 15s
    static_configs:
      - targets: ["10.0.1.11:8000", "10.0.1.11:8001"]
        labels: { pool: interactive-agent, node: hgx-01 }
  - job_name: dcgm
    static_configs:
      - targets: ["10.0.1.11:9400"]
        labels: { node: hgx-01 }
  - job_name: gateway
    static_configs:
      - targets: ["10.0.1.10:4000"]
```

On Tier 3 (Kubernetes) replace `static_configs` with `ServiceMonitor`/`PodMonitor` objects from the Prometheus Operator; the metrics themselves are identical.

**Retention:** metrics at full resolution for 30–90 days is plenty for SLO math (a 30-day SLO window needs 30 days of data — plan disk accordingly; a busy 8-replica pool with DCGM produces on the order of tens of GB/month, cheap by GPU-cluster standards). Keep benchmark results forever — they are tiny JSON files (§9).

---

## 2. The vLLM `/metrics` catalog (v0.26-era names)

Everything below was verified against vLLM `main` (v0.26 era, August 2026) — the metric definitions live in `vllm/v1/metrics/loggers.py` and the design rationale in `docs/design/metrics.md` in the vLLM repo.

**Exposition gotcha:** vLLM defines counters with names like `vllm:prompt_tokens`, but the Prometheus client library appends `_total` to counters on the wire. In PromQL you therefore query `vllm:prompt_tokens_total`, `vllm:prefix_cache_hits_total`, and so on. Histograms expand into `_bucket`, `_sum`, and `_count` series. Gauges keep their bare name.

### 2.1 Queue and saturation gauges

| Metric | Type | Meaning |
|---|---|---|
| `vllm:num_requests_running` | Gauge | Requests currently in the model's execution batch (being prefilled/decoded right now) |
| `vllm:num_requests_waiting` | Gauge | Requests queued, not yet scheduled |
| `vllm:num_requests_waiting_by_reason` | Gauge | Same, split by reason label (e.g. capacity vs deferred) — newer addition |
| `vllm:kv_cache_usage_perc` | Gauge | Fraction of KV-cache blocks in use, 0–1 (1 = 100%) |

**How to read them.** `running` bouncing around your configured `--max-num-seqs` with `waiting` at zero is a healthy busy server. **Sustained `waiting > 0` is the single clearest saturation signal vLLM gives you** — requests are queuing because the batch is full or KV space is exhausted, and TTFT for queued requests grows with the queue. Healthy interactive pools sit at `waiting ≈ 0` almost all the time; a throughput-batch pool may deliberately run with a queue.

`kv_cache_usage_perc` in the 0.3–0.9 range is normal operation; pinned near 1.0 is the precondition for the pathological state below.

### 2.2 Preemptions — the "KV thrash" counter

| Metric | Type | Meaning |
|---|---|---|
| `vllm:num_preemptions_total` | Counter | Times the scheduler evicted a running request (its KV blocks were reclaimed) to make room; the request restarts later |

When KV cache fills, vLLM preempts the lowest-priority running sequences and recomputes them later. Each preemption converts already-paid decode work into future prefill work — throughput collapses and tail latency explodes. **Healthy value: ~0.** Any sustained nonzero `rate(vllm:num_preemptions_total[5m])` means you are oversubscribed for your context lengths: lower concurrency caps, enable KV FP8, add LMCache offload, or add hardware. This metric plus `kv_cache_usage_perc ≈ 1.0` is the classic "agent sessions got longer than we planned" incident signature.

### 2.3 Latency histograms

| Metric | Type | Meaning |
|---|---|---|
| `vllm:time_to_first_token_seconds` | Histogram | **TTFT** (time to first token), measured from request arrival at the frontend |
| `vllm:inter_token_latency_seconds` | Histogram | **ITL** — gap between consecutive output tokens, one sample per token |
| `vllm:request_time_per_output_token_seconds` | Histogram | Per-request average **TPOT** (decode time ÷ output tokens), one sample per request |
| `vllm:e2e_request_latency_seconds` | Histogram | Full request latency, arrival → completion |

Plus a very useful decomposition of where a request's life went, each a histogram in seconds: `vllm:request_queue_time_seconds` (WAITING), `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`, and `vllm:request_inference_time_seconds` (RUNNING total). When e2e latency degrades, these four tell you immediately whether it is queueing (saturation), prefill (long prompts / cache-hit regression), or decode (batch too large, spec-decode regressed).

**ITL vs TPOT naming:** older vLLM (v0 engine) exposed a histogram called `vllm:time_per_output_token_seconds`. In the v1 engine this was replaced by `vllm:inter_token_latency_seconds` (per-token samples) plus `vllm:request_time_per_output_token_seconds` (per-request average). Old dashboards and blog posts still reference the old name — this is the most common "blank panel after upgrade" cause.

**Bucket-boundary caveat for SLOs.** Prometheus histograms only know their configured bucket edges. vLLM's TTFT buckets include edges like 0.25, 0.5, 0.75, 1.0, 2.5, 5, 10 seconds (verify against your build's `/metrics` output). `histogram_quantile()` interpolates within buckets, so a p99 estimate is approximate; and an exact "fraction of requests ≤ X" is only computable when X is an actual bucket edge. **Pick SLO thresholds that sit on bucket boundaries** (e.g. TTFT ≤ 2.5 s rather than ≤ 2.0 s) or accept interpolation error. The SOP's "TTFT p99 ≤ 2 s" target is fine for dashboards; for burn-rate math, align to a real edge.

### 2.4 Token and request counters

| Metric | Type | Meaning |
|---|---|---|
| `vllm:prompt_tokens_total` | Counter | Prefill tokens processed |
| `vllm:prompt_tokens_cached_total` | Counter | Prompt tokens served from cache (local + external) rather than recomputed |
| `vllm:generation_tokens_total` | Counter | Output tokens generated |
| `vllm:request_success_total` | Counter | Finished requests, labeled by `finished_reason` (`stop`, `length`, `abort`) |

`rate(vllm:generation_tokens_total[1m])` is your output-token throughput — the number to compare against benchmark results. Watch the `finished_reason="length"` share of `request_success_total`: a rising fraction means responses are being truncated at `max_tokens`, which for agents usually means broken tool calls (truncated JSON). There are also distribution histograms `vllm:request_prompt_tokens` and `vllm:request_generation_tokens` (tokens per request) — graph these to know what your *real* traffic shape is, then feed that shape into benchmarks (§8.4).

### 2.5 Prefix-cache counters and the hit-rate recipe

| Metric | Type | Meaning |
|---|---|---|
| `vllm:prefix_cache_queries_total` | Counter | Prompt tokens checked against the prefix cache |
| `vllm:prefix_cache_hits_total` | Counter | Prompt tokens found in the cache (prefill skipped for them) |
| `vllm:external_prefix_cache_queries_total` / `..._hits_total` | Counter | Same, for an external KV connector (e.g. LMCache tier) |

Both counters are in **tokens**, not requests. The hit-rate recipe (this replaces the deprecated v0 gauge `vllm:gpu_prefix_cache_hit_rate`):

```promql
sum(increase(vllm:prefix_cache_hits_total{pool="interactive-agent"}[10m]))
/
sum(increase(vllm:prefix_cache_queries_total{pool="interactive-agent"}[10m]))
```

For agentic traffic with byte-stable prompts and session-affinity routing, expect this to be **high — commonly 0.6–0.9+**, because every agent turn resends a mostly-identical prefix. Two operational rules from the SOP follow directly: (a) a sudden drop after a router, gateway, or prompt-template change is an incident (you just multiplied your prefill load); (b) compare the *per-replica* hit rate — one cold or mis-routed replica drags the pool average and shows up here first. `vllm:prompt_tokens_cached_total / vllm:prompt_tokens_total` gives a similar "fraction of prefill work avoided" view.

### 2.6 Speculative-decoding counters

Defined in `vllm/v1/spec_decode/metrics.py` (the older design doc listed these as "future"; they exist in current source):

| Metric | Type | Meaning |
|---|---|---|
| `vllm:spec_decode_num_drafts_total` | Counter | Draft rounds proposed |
| `vllm:spec_decode_num_draft_tokens_total` | Counter | Draft tokens proposed |
| `vllm:spec_decode_num_accepted_tokens_total` | Counter | Draft tokens accepted by the verifier |
| `vllm:spec_decode_num_accepted_tokens_per_pos` | Counter | Acceptances by draft position (position label) |

Acceptance rate (fraction of drafted tokens accepted):

```promql
rate(vllm:spec_decode_num_accepted_tokens_total[5m])
/ rate(vllm:spec_decode_num_draft_tokens_total[5m])
```

Mean acceptance length (tokens per verifier step, including the bonus token):

```promql
1 + rate(vllm:spec_decode_num_accepted_tokens_total[5m])
  / rate(vllm:spec_decode_num_drafts_total[5m])
```

Why you must watch this in production: speculative decoding only pays when acceptance is high. Acceptance rate is **model+workload dependent** — tool-call JSON accepts well, freeform prose less so. If a model promotion or prompt change drops acceptance substantially, your "1.5–2.5× decode speedup" quietly becomes a slowdown (you pay drafting cost for rejected tokens). Baseline the acceptance rate during the canary phase and alert on large regressions.

### 2.7 Renamed and deprecated names (dashboard-breaker table)

| Old (v0-era) name | Current (v1 / v0.26) replacement |
|---|---|
| `vllm:gpu_cache_usage_perc` | `vllm:kv_cache_usage_perc` |
| `vllm:gpu_prefix_cache_hit_rate` (gauge) | `vllm:prefix_cache_queries_total` + `vllm:prefix_cache_hits_total` (counters; compute the ratio) |
| `vllm:time_per_output_token_seconds` | `vllm:inter_token_latency_seconds` (+ `vllm:request_time_per_output_token_seconds`) |
| `vllm:num_requests_swapped`, `vllm:cpu_cache_usage_perc` | Removed (CPU-swap preemption mode is gone) |
| `vllm:avg_prompt_throughput_toks_per_s` etc. | Removed; use `rate()` over the token counters |

**Runbook rule:** after any vLLM image upgrade, `curl -s localhost:8000/metrics | grep '^vllm:' | cut -d'{' -f1 | sort -u` on the canary and diff against the previous list before rolling dashboards forward.

---

## 3. GPU telemetry: DCGM exporter, and the utilization trap

### 3.1 The trap, precisely

`DCGM_FI_DEV_GPU_UTIL` (the same number `nvidia-smi` shows as "GPU-Util") measures **the fraction of time at least one kernel was resident on the GPU**. It does not measure how much of the GPU's compute was used. An inference server almost always has *some* kernel running, so this metric pins at ~100% while the silicon may be mostly idle — real cases show `GPU_UTIL = 100%` with only ~18% of **SM** (Streaming Multiprocessor — the GPU's compute core cluster) time actually occupied. It is also incompatible with **MIG** (Multi-Instance GPU partitioning). Treat it as a "is anything at all happening" boolean, never as a capacity or efficiency signal.

The profiling counters (`DCGM_FI_PROF_*`, sourced from the same hardware counters as Nsight) are what you want:

| Field | Meaning | How to read it for LLM serving |
|---|---|---|
| `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | Fraction of time the graphics/compute engine was active (0–1) | The honest replacement for GPU_UTIL; MIG-aware |
| `DCGM_FI_PROF_SM_ACTIVE` | Fraction of SM-time with at least one warp resident | "How busy are the cores" |
| `DCGM_FI_PROF_SM_OCCUPANCY` | Fraction of resident-warp capacity in use | "How full are the busy cores" |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor-core pipe activity | High during prefill (compute-bound); much lower during decode |
| `DCGM_FI_PROF_DRAM_ACTIVE` | HBM memory-interface activity | High during decode (memory-bandwidth-bound) |

The prefill/decode signature is diagnostic gold: a decode-heavy pool shows **high `DRAM_ACTIVE`, modest `PIPE_TENSOR_ACTIVE`** — that is correct and healthy, not "wasted GPUs." If someone reads a 35% tensor-core number and proposes packing more work onto the node, the decode pool's ITL will pay for it. Conversely, prefill-heavy phases light up the tensor pipe. Note that dcgm-exporter's default config (`etc/default-counters.csv` in the repo) enables `GR_ENGINE_ACTIVE`, `PIPE_TENSOR_ACTIVE`, `DRAM_ACTIVE`, and PCIe bytes, but leaves `SM_ACTIVE`/`SM_OCCUPANCY` commented out — uncomment them in your mounted counters file.

### 3.2 Memory, power, thermals

| Field | Meaning | Watch for |
|---|---|---|
| `DCGM_FI_DEV_FB_USED` / `FB_FREE` (MiB) | Framebuffer (HBM) used/free | vLLM grabs `--gpu-memory-utilization` (e.g. 0.90) at startup, so this is *static* per process — a *change* means restarts or a stray process on the GPU |
| `DCGM_FI_DEV_POWER_USAGE` (W) | Instantaneous board power | B200 is a ~1.0–1.2 kW part depending on SKU/cooling; B300 higher (check your SKU's rated TGP). Sustained power far below the norm *under load* suggests throttling or stalled work |
| `DCGM_FI_DEV_GPU_TEMP` / `DCGM_FI_DEV_MEMORY_TEMP` (°C) | Die / HBM temperature | HBM temperature throttles before the die on modern parts; alert well below the throttle point per your node vendor's specs |
| `DCGM_FI_DEV_SM_CLOCK` (MHz) | Current SM clock | Sagging clocks under load = thermal/power throttling |

Because HBM occupancy is static by design, "GPU memory used" panels are *not* how you watch memory pressure in vLLM — `vllm:kv_cache_usage_perc` (§2.1) is. FB_USED earns its place for detecting the wrong thing running on the GPU.

### 3.3 Errors and fabric: ECC, XID, NVLink

- **ECC** (Error-Correcting Code memory) errors: `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` (single-bit, corrected — a wear indicator) and `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` (double-bit, uncorrectable — the serious one; typically kills the process and may retire memory pages). These are **commented out in dcgm-exporter's default counters file — enable them.** Any DBE = drain the node and run `dcgmi diag` (the DCGM diagnostic suite the SOP already requires at node acceptance).
- **XID errors**: `DCGM_FI_DEV_XID_ERRORS` exposes the most recent driver-reported error code (XIDs are the driver's error taxonomy — e.g. 48 is a DBE, 79 is "GPU fell off the bus"). Also ship kernel logs to Loki and alert on `Xid` lines; the exporter gauge alone can miss bursts.
- **NVLink**: `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` (enabled by default) for traffic, plus NVLink CRC/replay/recovery **error** counters (commented out by default — enable on NVLink-heavy TP=8 nodes). Rising NVLink error counts on one link manifest as mysterious tail-latency on tensor-parallel pools long before anything crashes.
- **PCIe**: `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` for link health; `DCGM_FI_PROF_PCIE_TX/RX_BYTES` for host-transfer volume (interesting once LMCache CPU offload is in play).

**Pitfall — profiling metrics and passes:** the `DCGM_FI_PROF_*` counters use the profiling infrastructure; run one dcgm-exporter per node only, and be aware that concurrent profilers (Nsight on the same GPU) can conflict with it.

---

## 4. Gateway-layer metrics

The gateway (LiteLLM in passthrough, or `agentic-api` once the Responses migration starts) is where *client-visible* truth lives — vLLM cannot see requests that never reached it, and it cannot attribute anything to an API key or team. Scrape the gateway's `/metrics` (LiteLLM exposes Prometheus metrics such as `litellm_proxy_total_requests_metric` and request-latency families with `model`/`team`/`api_key` labels; exact names vary by version and its `prometheus_metrics_config` — pin and verify like everything else).

The gateway signals that earn dashboard space:

| Signal | Why | Source |
|---|---|---|
| Requests by API key / team / alias | Attribution, abuse detection, migration progress (chat-completions → responses traffic per client, per SOP §6.3) | Gateway metrics |
| Error rate by class (4xx vs 5xx vs timeout) | The user-facing error SLI; vLLM's own success counter can look clean while the gateway times out | Gateway metrics |
| Upstream latency per pool (gateway → vLLM) vs total latency | Separates "gateway overhead / queueing" from "engine slow"; the gap is your gateway tax | Gateway metrics |
| **Tool-call parse failure rate** | The #1 "new model silently breaks agents" signal (SOP §5); infra metrics stay green while this burns | Gateway/vLLM logs via Loki, or a gateway counter if you add one |
| Rate-limit (429) and retry counts per key | Distinguishes "we are saturated" from "one client is hammering us" | Gateway metrics |

Deriving a log-based metric with Loki (LogQL), when the gateway has no native counter for it:

```logql
sum(rate({container="agentic-gateway"} |= "tool_call_parse_error" [5m]))
/
sum(rate({container="agentic-gateway"} |= "tool_call_attempt" [5m]))
```

Loki's *ruler* can evaluate expressions like this on a schedule and push the result into Prometheus, so parse-failure rate becomes an alertable first-class metric. Whatever your gateway logs, make it log tool-call outcomes in a stable, greppable form — that one logging decision buys you the most important agent-health metric for free (§10).

---

## 5. Dashboard design: the per-pool golden-signals layout

One dashboard per model pool, identical layout across pools, with `pool` and `model_name` as template variables. Grafana JSON lives in the config repo; the vLLM repo ships starter dashboards (`examples/observability/prometheus_grafana/` and `examples/online_serving/dashboards/grafana/` with `performance_statistics.json` / `query_statistics.json`) that you should mirror and adapt rather than start from zero — but check every query's metric names against §2.7 first.

Row by row (each panel with a PromQL sketch; `$pool` is the template variable):

**Row 1 — Header stats (single-stat panels).** Replicas up (`count(up{job="vllm", pool="$pool"} == 1)`); current running (`sum(vllm:num_requests_running{pool="$pool"})`); current waiting (same for `waiting`); engine/config fingerprint from `vllm:cache_config_info` labels (block size, KV dtype) so "what exactly is deployed" is on the dashboard, not in someone's memory.

**Row 2 — Traffic.** Request rate by outcome: `sum by (finished_reason) (rate(vllm:request_success_total{pool="$pool"}[5m]))`; token throughput in/out: `sum(rate(vllm:prompt_tokens_total{pool="$pool"}[5m]))` and `sum(rate(vllm:generation_tokens_total{pool="$pool"}[5m]))`; gateway requests per key (top 10).

**Row 3 — Latency (the SLO row).** TTFT p50/p99:

```promql
histogram_quantile(0.99,
  sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{pool="$pool"}[5m])))
```

ITL p99 (same shape over `vllm:inter_token_latency_seconds_bucket`); e2e p99; and a stacked "where did time go" panel of p95 queue / prefill / decode time from the §2.3 decomposition histograms. Draw the SLO thresholds as horizontal lines on the TTFT and ITL panels.

**Row 4 — Saturation.** Waiting queue per replica (`vllm:num_requests_waiting`); KV usage per replica (`vllm:kv_cache_usage_perc` — plot per replica, not averaged: one full replica preempts while the average looks fine); preemption rate (`sum(rate(vllm:num_preemptions_total{pool="$pool"}[5m]))`, expected flat zero).

**Row 5 — Cache & speculation (the agent-efficiency row).** Prefix-cache hit rate, pool-wide and per replica (§2.5 recipe); fraction of prompt tokens cached (`rate(vllm:prompt_tokens_cached_total[10m]) / rate(vllm:prompt_tokens_total[10m])`); spec-decode acceptance rate and mean acceptance length (§2.6).

**Row 6 — GPU health (DCGM, filtered to the pool's nodes).** `avg by (gpu) (DCGM_FI_PROF_GR_ENGINE_ACTIVE{node=~"$nodes"})`; tensor vs DRAM activity overlaid (the prefill/decode signature, §3.1); power and temperatures; a stat panel of ECC DBE + XID counts (should read 0 and be boring forever).

**Row 7 — Agent health (gateway + Loki).** Tool-call parse failure rate; structured-output validity; retry rate per key; task abandonment (§10). This row is what the model-promotion runbook (SOP §5 Phase 3) watches during a canary.

Design rules: golden signals above the fold; SLO thresholds drawn on the graphs so "are we OK" needs no tribal knowledge; per-replica breakdowns available one row below every pool aggregate; and nothing on the dashboard that no one would act on (utilization vanity panels breed alarm fatigue).

---

## 6. Alerting: SLO burn rates, symptoms before causes

### 6.1 Concepts in one paragraph

An **SLI** (service level indicator) is a measured ratio of good events to total events — e.g. "fraction of requests with TTFT ≤ 2.5 s." An **SLO** (service level objective) is a target for it over a window — "99% over 30 days." The **error budget** is the allowance the SLO implies (1% of requests may be slow). The **burn rate** is how fast you are spending that budget: burn rate 1 means exactly on budget; burn rate 14.4 means you will exhaust a 30-day budget in ~2 days. Alerting on burn rate — instead of "p99 crossed a line for 5 minutes" — makes alerts fire fast for big problems, slowly or not at all for noise.

### 6.2 The multi-window, multi-burn-rate pattern

The standard (Google SRE Workbook) configuration, for a 30-day SLO window — each alert requires **both** a long and a short window to exceed the same burn rate, so alerts stop quickly once the problem stops:

| Severity | Burn rate | Windows (long AND short) | Budget consumed at trigger |
|---|---|---|---|
| Page | 14.4× | 1 h and 5 m | 2% of 30-day budget |
| Page | 6× | 6 h and 30 m | 5% |
| Ticket | 1× | 3 d and 6 h | 10% |

The short window is ~1/12 of the long window. Two burn-rate SLIs cover most of what matters for an agent pool: **TTFT-latency SLO** and **error-rate SLO** (gateway 5xx + timeouts). ITL can be a third once the first two are trustworthy.

### 6.3 Worked example: TTFT p99-style SLO as burn-rate rules

SLO: **99% of interactive-agent requests have TTFT ≤ 2.5 s, over 30 days** (2.5 s because it is a real histogram bucket edge — §2.3). First, recording rules for the bad-fraction at each needed window:

```yaml
groups:
- name: slo_ttft_interactive
  rules:
  # fraction of requests with TTFT > 2.5s, per window
  - record: slo:ttft_bad_fraction:rate5m
    expr: |
      1 - (
        sum(rate(vllm:time_to_first_token_seconds_bucket{pool="interactive-agent",le="2.5"}[5m]))
        /
        sum(rate(vllm:time_to_first_token_seconds_count{pool="interactive-agent"}[5m]))
      )
  # ... identical rules for [30m], [1h], [6h], [3d] ...
```

Then the alerts (budget = 1 − 0.99 = 0.01):

```yaml
  - alert: TTFTSLOBurnFast
    expr: |
      slo:ttft_bad_fraction:rate1h  > (14.4 * 0.01)
      and
      slo:ttft_bad_fraction:rate5m  > (14.4 * 0.01)
    labels: { severity: page, pool: interactive-agent }
    annotations:
      summary: "TTFT SLO burning at >14x — 30-day budget gone in ~2 days"
      runbook: "runbooks/latency-spike.md"   # SOP §8.5: check WAITING, then prefix hit rate, then long-prefill interference

  - alert: TTFTSLOBurnSlow
    expr: |
      slo:ttft_bad_fraction:rate6h  > (6 * 0.01)
      and
      slo:ttft_bad_fraction:rate30m > (6 * 0.01)
    labels: { severity: page }

  - alert: TTFTSLOBurnTrickle
    expr: |
      slo:ttft_bad_fraction:rate3d  > (1 * 0.01)
      and
      slo:ttft_bad_fraction:rate6h  > (1 * 0.01)
    labels: { severity: ticket }
```

The error-rate SLO is the same skeleton with the SLI swapped in from the gateway: `bad = 5xx + timeouts`, `total = all requests`, per pool. Structured-output validity (§10) also fits this mold with a much tighter objective (e.g. 99.9%).

### 6.4 Symptom vs cause: what pages, what doesn't

**Page only on symptoms users experience** (SLO burn on latency and errors, plus hard availability: `up == 0` for a whole pool). **Everything else is a cause** — it belongs on dashboards and, at most, in ticket-severity alerts that provide diagnosis when a symptom pages:

| Signal | Treat as | Rationale |
|---|---|---|
| TTFT/ITL/error SLO burn | **Page** | Users feel it now |
| `vllm:num_requests_waiting` sustained high | Ticket / dashboard | Cause of future TTFT burn; the burn alert will page if it matters |
| Preemption rate > 0, KV usage ≈ 1.0 | Ticket | Cause; capacity action needed this week, not this minute |
| Prefix hit rate dropped > N points after a change | Ticket (page during canary windows) | Cause; SOP treats it as an incident but it degrades rather than breaks |
| ECC DBE, XID, NVLink errors | Ticket + auto-drain policy | Hardware causes; act via runbook, not adrenaline |
| GPU "utilization" anything | Never alert | See §3.1 |

The payoff of this discipline in a small on-call rotation: pages are rare, always actionable, and every page maps to a runbook (SOP §8).

---

## 7. Tracing the agent loop with OpenTelemetry

### 7.1 Why metrics aren't enough for agents

An agent task is a *chain*: client → gateway → vLLM (turn 1) → tool execution → gateway → vLLM (turn 2) → … Metrics tell you each hop's aggregate health; they cannot tell you why *this* task took 90 seconds. A **trace** is a tree of timed **spans** (one per hop/operation) sharing a trace ID, propagated between services via the W3C `traceparent` HTTP header. For agent debugging, the trace answers: how many model turns, how long was each tool call, where did the 40-second gap hide (queue? prefill? the client's own code?).

### 7.2 vLLM's OTLP support (verified against v0.26-era docs/examples)

```bash
export OTEL_SERVICE_NAME="vllm-interactive-agent"
export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true   # plain gRPC inside the enclave; use your PKI if required
vllm serve /models/org/model-NVFP4 \
  --otlp-traces-endpoint "grpc://otel-collector.internal:4317" \
  --collect-detailed-traces model
```

- `--otlp-traces-endpoint` — where to ship spans (your in-enclave OTel Collector; gRPC 4317 by default, or set `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf` and use `:4318/v1/traces`).
- `--collect-detailed-traces {model,worker,all}` (comma-combinable) — adds fine-grained timing (e.g. model forward/execute time) at some performance cost; only meaningful with the endpoint set. Leave it off in steady state; flip on `model` when hunting a latency mystery.
- vLLM's request spans carry **`gen_ai.*` semantic-convention attributes** — request ID, model, prompt/completion token counts, and latency breakdowns (time-in-queue, time-to-first-token, e2e) — so a Tempo query like "traces where queue time > 1 s" works without log spelunking. Exact attribute inventory varies by version; check one span in Tempo after each upgrade.
- Wrapping the server with `opentelemetry-instrument vllm serve ...` (after installing `opentelemetry-instrumentation-fastapi` from your devpi mirror) adds HTTP-layer spans and, critically, **honors incoming `traceparent`** so vLLM spans attach to the caller's trace instead of starting orphan traces.

### 7.3 Assembling the end-to-end picture

1. **Client/agent framework:** instrument with the OTel SDK; one root span per agent *task*, child spans per turn and per tool call. This is where task-level latency truth lives.
2. **Gateway:** LiteLLM and the agentic-api gateway both support OTel export (verify the version you pin); at minimum the gateway must *propagate* `traceparent` to vLLM.
3. **vLLM:** as above.
4. **Collector + Tempo:** one OTel Collector (mirrored image) receiving OTLP from all three, exporting to Tempo; Grafana queries Tempo alongside Prometheus. Sample head-based at 5–10% in steady state and 100% for canary aliases during promotions — agent debugging wants whole tasks, so sample by *trace ID* (which OTel does by default), never per-span.

Air-gap note: all of this is in-enclave traffic; the only "external" thing to mirror is the images (otel-collector, tempo). No SaaS APM, no egress.

---

## 8. Benchmarking deep dive

Two tools cover everything: **guidellm** (workload-realistic load generation and SLO analysis — the SOP's standing performance gate) and **`vllm bench`** (built into vLLM — quick, zero extra install, includes goodput math). NVIDIA's **genai-perf** is in maintenance; its successor **AIPerf** (`pip install aiperf`, from the ai-dynamo org) is the NVIDIA-side option — fine as a cross-check, not required in our stack.

### 8.1 guidellm

guidellm (a vLLM project) drives an OpenAI-compatible endpoint with controlled traffic patterns and produces percentile reports. **CLI drift warning:** current releases (v0.5.x, mid-2026) use `guidellm run` with `--profile`/`--data`/`--constraint`; older versions (≤0.3, and most 2025 blog posts) used `guidellm benchmark --rate-type ... --rate ...`. Pin one version in the enclave and write your harness against it.

The profiles (traffic shapes), current syntax:

| Profile | What it does | Use it for |
|---|---|---|
| `synchronous` | One request at a time, sequentially | Floor numbers: best-case TTFT/ITL, zero queueing |
| `concurrent` | Fixed number of parallel streams (`streams=1,2,4,...` sweeps them) | **The capacity method (§9):** SLO compliance vs concurrency |
| `throughput` | Fire as fast as possible (optional `max_concurrency`) | Ceiling: max tokens/s, saturation behavior |
| `constant` | Fixed arrival rate (req/s), open-loop | SLO validation at a target arrival rate |
| `poisson` | Poisson-distributed arrivals at a mean rate | Same, with realistic burstiness |
| `sweep` | Auto: synchronous → throughput → interpolated rates between them (`sweep_size` steps) | First contact with a new (model, hardware) pair |
| `replay` | Replay a recorded trace with original timing (`time_scale` to compress/stretch) | Re-running *real* production traffic shapes |

A performance-gate run shaped like our agent traffic (long prompt, short output — see §8.4):

```bash
guidellm run \
  --target http://gateway.internal:4000/v1 \
  --model  agent-default \
  --profile kind=concurrent,streams=8,16,32,64,rampup_duration=60 \
  --data   kind=synthetic_text,prompt_tokens=12000,prompt_tokens_stdev=6000,output_tokens=400,output_tokens_stdev=250 \
  --constraint kind=max_duration,seconds=600 \
  --constraint kind=max_error_rate,rate=0.02 \
  --output kind=json,path=results/agentX-b200x8-interactive.json \
  --output kind=html,path=results/agentX-b200x8-interactive.html
```

(Synthetic-data parameter names vary slightly across versions — check `guidellm run --help` on your pinned build.) Output includes full TTFT/ITL/e2e distributions per sub-benchmark (each `streams` value is a separate benchmark) as JSON/CSV/HTML — read "max concurrency at SLO" straight off the concurrency sweep by finding the largest `streams` whose p99s sit inside the SLO.

**Datasets beyond synthetic:** local JSON/CSV/parquet files of real prompts (`--data kind=json_file,path=...`), HuggingFace-format datasets from a *local* path (never the Hub, in the enclave), and trace replay (Mooncake-style trace files, plus `--profile kind=replay` for timing-faithful replay). **Building datasets from real agent traces** is the highest-value option: log (prompt token count, output token count, arrival timestamp, and — if privacy allows — the prompt text) at the gateway for a representative day, and either replay it directly or fit your synthetic distributions to it. Real traces capture the two things synthetic defaults miss: the huge shared prefixes and the bursty arrival pattern of tool-calling loops.

**Air-gapped install:** mirror `ghcr.io/vllm-project/guidellm:<pinned-tag>` into Harbor (or pip-install from the devpi snapshot into a tools venv). Point it at a **local tokenizer path** (`--processor /models/org/model-NVFP4` — guidellm needs the model's tokenizer to count tokens; with `HF_HUB_OFFLINE=1` set, anything that tries to download will fail fast rather than silently hang). The synthetic-data text corpus ships inside the package, so no external dataset is needed. The Red Hat article on air-gapped guidellm (Sept 2025) documents the same pattern for OpenShift, but note it used an older image with different flags (`--rate-type`-era) — follow its mirroring approach, not its flag names.

### 8.2 `vllm bench` — the built-in kit

Ships in the vLLM container (`vllm bench serve | latency | throughput | sweep`), so it is always available in the enclave with zero extra mirroring:

- **`vllm bench serve`** — online benchmark against a running server; reports TTFT, **TPOT** (time per output token), ITL, e2e, and throughput, with `--percentile-metrics ttft,tpot,itl,e2el` and `--metric-percentiles 50,90,99`. Key flags: `--dataset-name {random,sharegpt,hf,custom}`, `--random-input-len` / `--random-output-len`, `--num-prompts`, `--request-rate` (`inf` = throughput mode), `--max-concurrency`, `--burstiness` (1.0 = Poisson), ramp-up flags, `--save-result`.
- **The `--goodput` flag** is the killer feature: `--goodput ttft:2500 tpot:60` (milliseconds) makes it report **goodput** — request throughput counting only requests that met the stated SLOs (the DistServe definition). This turns "we serve 41 req/s" into "we serve 29 req/s *within SLO*," which is the only throughput number that should ever reach a capacity plan.
- **`vllm bench latency` / `throughput`** — offline (no server; loads the model in-process) single-batch latency and max engine throughput. Useful for engine-level A/B (kernel/quantization changes), not for serving SLOs.
- **`vllm bench sweep serve` / `sweep plot`** — drives Cartesian products of server-params × bench-params and plots the resulting curves; handy for recipe tuning days (e.g. `--max-num-batched-tokens` vs ITL trade-off, SOP §2.3).

Example gate run through the gateway:

```bash
vllm bench serve \
  --backend openai-chat --base-url http://gateway.internal:4000 \
  --model agent-default \
  --dataset-name random --random-input-len 12000 --random-output-len 400 \
  --num-prompts 512 --max-concurrency 32 --request-rate inf \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --goodput ttft:2500 tpot:60 \
  --save-result
```

### 8.3 genai-perf → AIPerf

**genai-perf** (part of Triton's perf-analyzer tooling) benchmarks OpenAI-compatible endpoints and reports TTFT/ITL/throughput, but NVIDIA has stopped feature development on it. Its successor **AIPerf** (`pip install aiperf`; `aiperf profile --model <alias> --endpoint-type chat --url http://gateway:4000 ...`) adds concurrency and request-rate sweeps, goodput measurement, and p50/p90/p99 reporting. If you mirror one NVIDIA benchmarking tool for cross-validation of guidellm numbers, mirror AIPerf; keep genai-perf only if you inherit scripts that use it.

### 8.4 Methodology pitfalls (the honest-numbers box)

> **Pitfall 1 — No warmup.** The first requests after startup pay CUDA-graph capture, allocator warmup, and cold caches; they will wreck your p99. Use ramp-up (`rampup_duration` in guidellm profiles; `--ramp-up-strategy` in `vllm bench serve`) or discard the first minute. Never benchmark a replica that just booted and call it steady-state.
>
> **Pitfall 2 — Prefix-cache contamination.** If your load generator sends the same (or heavily shared-prefix) prompts repeatedly, the prefix cache absorbs most prefill and your TTFT numbers become fantasy for uncached traffic. Decide *which* number you are measuring: for a **worst-case baseline**, start the server with `--no-enable-prefix-caching` (or use fully randomized prompts) and record it as such; for a **realistic agent number**, keep caching on but replay real traces so the hit rate matches production (§2.5), and *record the hit rate observed during the run* alongside the results. A benchmark result without its cache-hit context is uninterpretable.
>
> **Pitfall 3 — Wrong token-length distribution.** The classic 512-in/128-out synthetic profile resembles nothing about agent traffic, which is more like 5k–50k in (system prompt + tools + trajectory) and 100–800 out (a tool call or short answer), with high variance. Pull the real distribution from `vllm:request_prompt_tokens` / `vllm:request_generation_tokens` histograms and match it. Prefill-heavy traffic stresses completely different resources (compute, chunked-prefill interference) than decode-heavy traffic (HBM bandwidth, batch effects).
>
> **Pitfall 4 — The client is the bottleneck.** At high concurrency, a Python load generator can saturate its own CPU parsing SSE streams, and you end up benchmarking your benchmarker (flat throughput + rising "latency" while the GPUs idle is the tell). Watch client CPU during runs, run the generator on a separate machine from the server, and shard across processes if needed. vLLM's Rust frontend line now includes a native bench port (`vllm-project/vllm-bench`) precisely for this.
>
> **Pitfall 5 — Direct-to-engine numbers sold as production numbers.** Benchmark through the gateway for gate/capacity numbers (it is the path users get, including its overhead and any routing effects) and directly against a replica when isolating engine behavior. Record which path a result used. If gateway-path p99 diverges from direct-path p99 by more than a few tens of milliseconds, that gap is itself a finding.
>
> **Pitfall 6 — Closed-loop vs open-loop confusion.** `concurrent` (closed-loop: next request waits for a slot) can't reveal queue collapse under overload; `constant`/`poisson` (open-loop: arrivals don't care about completions) can. Use closed-loop for capacity ("max concurrency at SLO"), open-loop to validate behavior at and beyond expected arrival rates.

---

## 9. Capacity planning: "max concurrency at SLO"

The SOP's capacity rule of thumb, operationalized:

1. **For each (model, profile, hardware) tuple**, run a guidellm `concurrent` sweep (e.g. `streams=1,2,4,8,16,24,32,48,64`) with production-shaped data (§8.4), through the gateway, after warmup.
2. **Find the largest concurrency where all SLOs hold** — for `interactive-agent`: TTFT p99 ≤ 2.5 s AND ITL p99 ≤ 60 ms AND error rate ≈ 0. That integer — say **C\* = 32 sessions on 8× B200 for model X, interactive profile** — is the tuple's capacity number.
3. **Record it in the recipe file** in the config repo, next to the exact `vllm serve` flags it was measured with, plus the benchmark JSON, the engine image digest, and the observed prefix-cache hit rate during the run. A capacity number divorced from its config is a rumor.
4. **Headroom policy:** plan steady-state peak at **≤ 70–80% of C\*** per pool, and keep N+1 at the pool level (lose one replica/node and remain under C\* on the survivors). The 20–30% margin absorbs traffic burstiness, cache-hit-rate variance, and the measurement's own optimism.
5. **Enforce it live:** the gateway's per-pool concurrency limit and the alert `sum(vllm:num_requests_running + vllm:num_requests_waiting) > 0.8 * C*` (ticket severity) close the loop between the plan and reality.
6. **Regression-benchmark on every change:** engine image upgrade, model promotion, recipe flag change, driver bump — re-run the same sweep on the canary slice and diff against the stored JSON before rollout. vLLM performance genuinely moves release to release, in both directions; the eval gate catches quality regressions, this catches performance regressions. Keep all historical results — the time series of C\* per model across engine versions is your most persuasive procurement document.

---

## 10. Agent-health signals beyond infrastructure

A pool can be perfectly green — TTFT fine, zero errors, GPUs healthy — while every agent task fails, because the model is emitting tool calls the parser mangles, or valid-but-wrong JSON, or the agents are looping. These four signals close that gap; all fit the burn-rate framework of §6 with tight objectives:

| Signal | Definition | Derived from |
|---|---|---|
| **Tool-call parse failure rate** | Model outputs intended as tool calls that the parser could not convert to structured calls | vLLM logs parser failures; the gateway sees malformed/absent `tool_calls` on responses that should have them. Count both via Loki→ruler (§4) or a gateway counter. Healthy: ≈ 0. Rises abruptly with a wrong `--tool-call-parser`, a chat-template change, or an engine upgrade (SOP §8.1: re-run tool-call tests on every upgrade) |
| **Structured-output validity** | Fraction of schema-constrained outputs that validate against their schema | Validate at the gateway (or agent SDK wrapper) on every `response_format`/tool-args payload. With xgrammar-guided decoding this should be ≈ 100%; failures indicate truncation (`finished_reason="length"`, §2.4), guided decoding silently off, or client-side schema drift |
| **Retry rate** | Model-turn requests that are retries of a failed/timed-out/unparseable turn | Cleanest: agent SDK wrapper tags retries with a header the gateway counts per key. Proxy: bursts of near-identical requests per session. Rising retries inflate load (watch with §2.1 saturation) and usually *precede* user complaints |
| **Task abandonment** | Agent tasks that end without reaching a terminal success/failure state | The agent framework's own task events (best); post-Responses-migration, the gateway can approximate it from `previous_response_id` chains that stop mid-flight with no terminal turn. A jump after a model promotion is the canary-abort signal even when infra is clean |

During a promotion canary (SOP §5 Phase 3), Row 7 of the dashboard — these four signals, split by alias — is what you actually stare at. The infra rows tell you the new model is *fast*; this row tells you it *works*.

---

## 11. The ten PromQL queries to know by heart

```promql
# 1. TTFT p99, per pool (the headline SLO number)
histogram_quantile(0.99, sum by (le) (
  rate(vllm:time_to_first_token_seconds_bucket{pool="interactive-agent"}[5m])))

# 2. ITL p99, per pool
histogram_quantile(0.99, sum by (le) (
  rate(vllm:inter_token_latency_seconds_bucket{pool="interactive-agent"}[5m])))

# 3. SLO bad-fraction for burn-rate rules (TTFT > 2.5s share)
1 - sum(rate(vllm:time_to_first_token_seconds_bucket{pool="interactive-agent",le="2.5"}[1h]))
  / sum(rate(vllm:time_to_first_token_seconds_count{pool="interactive-agent"}[1h]))

# 4. Saturation: waiting queue depth, per replica
sum by (instance) (vllm:num_requests_waiting{pool="interactive-agent"})

# 5. KV-cache pressure with thrash detection (usage AND preemptions)
max by (instance) (vllm:kv_cache_usage_perc{pool="interactive-agent"})
# alongside:
sum(rate(vllm:num_preemptions_total{pool="interactive-agent"}[5m]))

# 6. Prefix-cache hit rate (tokens), pool-wide — a drop is an incident
sum(increase(vllm:prefix_cache_hits_total{pool="interactive-agent"}[10m]))
/ sum(increase(vllm:prefix_cache_queries_total{pool="interactive-agent"}[10m]))

# 7. Output-token throughput, per pool (compare to benchmark ceiling)
sum(rate(vllm:generation_tokens_total{pool="interactive-agent"}[1m]))

# 8. Truncation share — rising = agents getting cut-off JSON
sum(rate(vllm:request_success_total{finished_reason="length"}[10m]))
/ sum(rate(vllm:request_success_total[10m]))

# 9. Speculative-decoding acceptance rate — regression = latency feature off
rate(vllm:spec_decode_num_accepted_tokens_total[5m])
/ rate(vllm:spec_decode_num_draft_tokens_total[5m])

# 10. Real GPU busyness (never DCGM_FI_DEV_GPU_UTIL)
avg by (node) (DCGM_FI_PROF_GR_ENGINE_ACTIVE)
```

## 12. Worked SLO document template

Keep one of these per (pool, alias) in the config repo; it is the contract the dashboards and alerts implement.

```markdown
# SLO: interactive-agent pool (alias: agent-default)
Owner: platform-inference          Review: quarterly, and at every model promotion
Window: 30 days rolling            Hardware basis: 8x B200 per replica, 2 replicas

## SLIs and objectives
| SLI                        | Definition (measurement point)                          | Objective |
|----------------------------|---------------------------------------------------------|-----------|
| TTFT latency               | share of requests with TTFT <= 2.5 s (vLLM histogram)   | 99%       |
| Streaming smoothness       | share of tokens with ITL <= 60 ms (vLLM histogram)      | 99%       |
| Availability/errors        | share of requests not 5xx/timeout (gateway)             | 99.9%     |
| Structured-output validity | share of schema-constrained outputs that validate (gw)  | 99.9%     |

## Error budgets (30 d)
1% slow-TTFT requests; 0.1% failed requests; 0.1% invalid structured outputs.

## Alerting
Multi-window multi-burn-rate per SRE Workbook: page at 14.4x (1h+5m) and
6x (6h+30m); ticket at 1x (3d+6h). Rules: alerts/slo-interactive-agent.yaml.

## Capacity basis
C* = 32 concurrent sessions per replica at SLO (guidellm concurrent sweep,
results/agentX-b200x8-interactive.json, engine sha256:..., prefix hit rate
0.74 during run). Plan peak <= 0.8 * C* per replica; pool is N+1.

## Exclusions
Scheduled maintenance windows announced >= 24 h ahead; requests > 128k context
(routed to long-context pool, covered by its own SLO doc).

## Consequences
Budget exhausted -> feature/promotion freeze for the pool until back in budget;
recurring trickle burns -> capacity review with the C* history attached.
```

---

## Common pitfalls (chapter-wide recap)

> - Dashboards written against v0-era metric names (`gpu_cache_usage_perc`, `time_per_output_token_seconds`) go blank after upgrades — diff `/metrics` on every engine bump.
> - Counters need `_total` in PromQL even though vLLM's source defines them without it.
> - Alerting on GPU utilization, or reading `DCGM_FI_DEV_GPU_UTIL` as capacity — it is a boolean in disguise.
> - Averaging `kv_cache_usage_perc` across replicas — one full replica preempts while the average looks fine.
> - SLO thresholds that don't sit on histogram bucket edges — your burn-rate math becomes interpolated fiction.
> - ECC and NVLink error counters left at dcgm-exporter defaults (disabled) — you find out about the dying HBM stack from a crash instead of a ticket.
> - Benchmarks without warmup, without recorded cache-hit rate, with toy token distributions, or through a different path than production — all four produce confident, wrong capacity numbers.
> - Green infra dashboards with no agent-health row — the model can be fast and useless at the same time.

---

## Study questions

1. **Why is `DCGM_FI_DEV_GPU_UTIL` misleading for LLM serving, and what should you use instead?**
   Answer: It measures the fraction of time *any* kernel was resident, so it pins near 100% regardless of how much of the GPU is working (and it's MIG-incompatible). Use `DCGM_FI_PROF_GR_ENGINE_ACTIVE` for overall busyness and `SM_ACTIVE`/`SM_OCCUPANCY`/`PIPE_TENSOR_ACTIVE`/`DRAM_ACTIVE` for how and where the GPU is actually loaded.

2. **How do you compute prefix-cache hit rate in current vLLM, and what units are the counters in?**
   Answer: `increase(vllm:prefix_cache_hits_total[w]) / increase(vllm:prefix_cache_queries_total[w])`; both counters count **tokens**, not requests. The old `gpu_prefix_cache_hit_rate` gauge is deprecated.

3. **What does a sustained nonzero `vllm:num_preemptions_total` rate mean, and what usually accompanies it?**
   Answer: The scheduler is evicting running requests because KV-cache blocks ran out — paid decode work becomes future recompute, so throughput and tail latency degrade. It's typically accompanied by `kv_cache_usage_perc` pinned near 1.0; fixes are lower concurrency caps, KV FP8, offload (LMCache), or more hardware.

4. **Why should your TTFT SLO threshold be 2.5 s rather than 2.0 s in the burn-rate rules?**
   Answer: Prometheus histograms only measure exactly at bucket edges, and 2.5 s is a real edge in vLLM's TTFT buckets while 2.0 s is not — `histogram_quantile` would interpolate. Aligning the SLO to an edge makes the good/bad ratio exact.

5. **State the multi-window multi-burn-rate page conditions for a 30-day SLO.**
   Answer: Page when burn rate > 14.4× over both 1 h and 5 m (2% of budget gone), or > 6× over both 6 h and 30 m (5% gone); ticket at 1× over 3 d and 6 h (10% gone). The short window (~1/12 of the long) makes alerts reset quickly once the problem stops.

6. **What is the difference between ITL and TPOT in vLLM's current metrics, and what changed from v0?**
   Answer: `vllm:inter_token_latency_seconds` samples every gap between consecutive tokens; `vllm:request_time_per_output_token_seconds` records each request's *average* decode time per token. They replace v0's single `vllm:time_per_output_token_seconds` histogram.

7. **When should you disable prefix caching for a benchmark, and when should you leave it on?**
   Answer: Disable it (`--no-enable-prefix-caching`, or use fully random prompts) when you want an honest worst-case/uncached baseline; leave it on and replay real traces when you want a realistic agent number — and in that case record the observed hit rate with the results, since the number is meaningless without it.

8. **What does `vllm bench serve --goodput ttft:2500 tpot:60` report that plain throughput doesn't?**
   Answer: Goodput — request throughput counting only requests that met the stated TTFT/TPOT SLOs (DistServe's definition). It's the only throughput figure that belongs in a capacity plan, because over-saturated servers post high raw throughput while violating every latency target.

9. **Describe the "max concurrency at SLO" method in one breath.**
   Answer: For each (model, profile, hardware) tuple, run a warmed-up guidellm `concurrent` sweep with production-shaped data through the gateway, find the largest concurrency where all SLO percentiles still hold (C\*), record it plus the exact config and benchmark JSON next to the recipe, plan peak at ≤ 70–80% of C\* with N+1, and re-measure on every engine/model/driver change.

10. **Why does spec-decode acceptance rate belong on a production dashboard rather than just in benchmark reports?**
    Answer: Acceptance depends on the model and the live workload; a promotion or prompt change can drop it enough that drafting costs more than it saves, silently erasing the 1.5–2.5× decode win. `rate(accepted)/rate(draft_tokens)` catches that regression in production.

11. **Your infra dashboard is green but agents are failing tasks. Name the four signals that catch this and where each comes from.**
    Answer: Tool-call parse failure rate (vLLM/gateway logs via Loki or a gateway counter), structured-output validity (schema validation at the gateway/SDK), retry rate (tagged retries counted at the gateway), and task abandonment (agent-framework task events, or Responses-API `previous_response_id` chains that never reach a terminal turn).

12. **How does a trace get connected across client → gateway → vLLM, and which vLLM flags are involved?**
    Answer: The client's OTel SDK starts a trace and sends the W3C `traceparent` header; the gateway propagates it; vLLM, run with `--otlp-traces-endpoint` (plus optional `--collect-detailed-traces model|worker|all` and FastAPI instrumentation via `opentelemetry-instrument`), attaches its spans — carrying `gen_ai.*` attributes like queue time and TTFT — to the same trace ID, which Tempo assembles into one timeline.

---

## Sources

Primary sources used for this document (all fetched/verified August 2026):

- vLLM metrics design doc (v1 metric inventory, deprecations, prefix-cache recipe): https://docs.vllm.ai/en/latest/design/metrics/ (source: https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md)
- vLLM v1 metric definitions (source of truth for names): https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py
- vLLM speculative-decoding metrics + PromQL recipes: https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/metrics.py
- vLLM v0.26.0 release: https://github.com/vllm-project/vllm/releases/tag/v0.26.0
- vLLM benchmark CLI docs (`vllm bench serve/latency/throughput/sweep`, `--goodput`, percentiles): https://docs.vllm.ai/en/latest/benchmarking/cli/ and https://docs.vllm.ai/en/latest/cli/bench/serve/
- vLLM OpenTelemetry example (OTLP flags, env vars, FastAPI instrumentation): https://github.com/vllm-project/vllm/blob/main/examples/observability/opentelemetry/README.md
- vLLM observability config (`--collect-detailed-traces` values): https://docs.vllm.ai/en/latest/api/vllm/config/observability/
- vLLM Prometheus/Grafana example dashboards: https://github.com/vllm-project/vllm/tree/main/examples/observability/prometheus_grafana and https://github.com/vllm-project/vllm/tree/main/examples/online_serving/dashboards/grafana
- guidellm README and benchmark docs (profiles, data kinds, constraints, outputs, container): https://github.com/vllm-project/guidellm and https://github.com/vllm-project/guidellm/blob/main/docs/getting-started/benchmark.md
- Red Hat: benchmarking with GuideLLM in air-gapped clusters (mirroring + local tokenizer pattern; older CLI): https://developers.redhat.com/articles/2025/09/15/benchmarking-guidellm-air-gapped-openshift-clusters
- dcgm-exporter default counters (what's enabled/disabled by default): https://github.com/NVIDIA/dcgm-exporter/blob/main/etc/default-counters.csv
- dcgm-exporter issue: replace `DCGM_FI_DEV_GPU_UTIL` with `DCGM_FI_PROF_GR_ENGINE_ACTIVE`: https://github.com/NVIDIA/dcgm-exporter/issues/341
- SuperOrbital: GPU underutilization / GPU_UTIL vs SM_ACTIVE evidence: https://superorbital.io/blog/gpu-kubernetes-underutilization/
- Google SRE Workbook, "Alerting on SLOs" (multi-window multi-burn-rate): https://sre.google/workbook/alerting-on-slos/
- NVIDIA GenAI-Perf docs (maintenance status, AIPerf recommendation): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
- NVIDIA AIPerf (genai-perf successor): https://github.com/ai-dynamo/aiperf
- LiteLLM Prometheus metrics docs: https://docs.litellm.ai/docs/proxy/prometheus
