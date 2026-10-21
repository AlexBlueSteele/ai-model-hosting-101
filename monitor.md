# GPU Inference Monitoring Reference

Oct 21, 2026 · @Alex Steele · vLLM on NVIDIA B200 / B300 / Hopper

Everything to track in Grafana for vLLM model serving on NVIDIA GPUs, with what each signal means and the alert to derive from it. Metric names shift slightly between vLLM and DCGM Exporter versions, so confirm them against each `/metrics` endpoint.

## 1. Top five signals

If you only alert on five things, these catch most outages:

1. Synthetic probe success per model
2. vLLM requests waiting (queue)
3. GPU Xid errors
4. NCCL timeouts in logs
5. Container restarts

## 2. vLLM metrics (serving health)

| Metric | What it tells you | Watch for |
| --- | --- | --- |
| `vllm:num_requests_running` | Requests actively being processed | Baseline for load |
| `vllm:num_requests_waiting` | Requests queued, not yet scheduled | Sustained > 0 means the model can't keep up |
| `vllm:kv_cache_usage_perc` (older: `vllm:gpu_cache_usage_perc`) | How full the KV cache is | Near 100% means stalls and preemption |
| `vllm:num_preemptions_total` | Requests paused to free cache | Rising = cache thrashing / overload |
| `vllm:time_to_first_token_seconds` | Latency until first token | High with empty queue = model/GPU problem, not load |
| `vllm:inter_token_latency_seconds` (older: `time_per_output_token`) | Streaming speed | Jump after upgrade = vLLM regression |
| `vllm:e2e_request_latency_seconds` | Total request time | Trend vs baseline |
| `vllm:request_queue_time_seconds` | Time spent waiting in queue | Confirms queueing vs compute |
| `vllm:request_success_total{finished_reason}` | Completions by reason (stop, length, abort) | Abort spike = failures |
| `vllm:prompt_tokens_total` / `vllm:generation_tokens_total` | Token throughput | Zero while requests wait = hung |
| `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` | Prefix cache efficiency | Drop after config change |
| `vllm:num_requests_swapped` (older versions) | Requests swapped to CPU | Any sustained value = memory pressure |

## 3. DCGM metrics (GPU health)

| Metric | What it tells you | Watch for |
| --- | --- | --- |
| `DCGM_FI_DEV_XID_ERRORS` | Last Xid error code per GPU | Any new value — most important GPU signal |
| `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` | Uncorrectable (double-bit) ECC errors | Any increase = critical |
| `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` | Correctable (single-bit) ECC errors | Steady rise = early warning |
| `DCGM_FI_DEV_ROW_REMAP_FAILURE` | Memory row remap failed | > 0 = replace GPU |
| `DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS` | Rows remapped after uncorrectable errors | Rising = degrading memory |
| `DCGM_FI_DEV_ROW_REMAP_PENDING` | Remap waiting for GPU reset | > 0 = schedule reset |
| `DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL` | NVLink data errors | Rising = NCCL failures likely next |
| `DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL` / `RECOVERY_ERROR_COUNT_TOTAL` | NVLink retries/recoveries | Rising = link instability |
| `DCGM_FI_PROF_NVLINK_TX_BYTES` / `RX_BYTES` | NVLink bandwidth | Drop during TP serving = comms issue |
| `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` | PCIe retransmits | Rising = bus/riser problem |
| `DCGM_FI_DEV_GPU_TEMP` / `DCGM_FI_DEV_MEMORY_TEMP` | GPU and HBM temperature | Sustained high = throttling |
| `DCGM_FI_DEV_POWER_USAGE` | Power draw | Near cap = power throttling |
| `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` (newer: `CLOCKS_EVENT_REASONS`) | Why clocks are reduced | Thermal or power throttle |
| `DCGM_FI_DEV_SM_CLOCK` | Current SM clock | Drops vs max = throttled |
| `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | True engine activity | Near zero while queue > 0 = stuck |
| `DCGM_FI_PROF_SM_ACTIVE` / `SM_OCCUPANCY` | SM usage | Efficiency baseline |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor core usage | Baseline for model efficiency |
| `DCGM_FI_PROF_DRAM_ACTIVE` | HBM bandwidth usage | Decode is usually memory-bound |
| `DCGM_FI_DEV_FB_USED` / `FB_FREE` | GPU memory used/free | Unexpected change after restart |
| `DCGM_FI_DEV_GPU_UTIL` | Coarse utilization | Less reliable than PROF metrics |

Note: ECC, row-remap and NVLink error fields often need enabling in the DCGM Exporter counters CSV.

## 4. Host metrics (node\_exporter)

- CPU, memory and swap usage (tokenization and scheduling are CPU-bound)
- Disk space on model cache and container volumes
- Network errors and throughput (InfiniBand/RoCE for multi-node)
- Node up/down

## 5. Log alerts (Loki)

### GPU / kernel (dmesg, journald)

- `NVRM: Xid` lines. Critical codes to call out:
  - 13, 31 — graphics engine / MMU fault (often software or driver)
  - 43 — GPU stopped processing
  - 48 — double-bit ECC error
  - 63, 64 — row remapping event / failure
  - 74 — NVLink error
  - 79 — GPU has fallen off the bus
  - 94, 95 — contained / uncontained ECC error
  - 119, 120 — GSP firmware error/timeout
- NVIDIA Fabric Manager errors or service stopped (NVSwitch systems such as DGX/HGX)
- Driver or persistence daemon failures

### NCCL (container logs, with NCCL\_DEBUG=WARN)

- `Watchdog caught collective operation timeout`
- `NCCL WARN`
- `NCCL error` / `unhandled system error`
- `unhandled cuda error`

### vLLM (container logs)

- `EngineDeadError` / engine core died
- `CUDA out of memory`
- Model architecture not supported / failed to load weights
- Tokenizer or chat template errors
- Any Python `Traceback` at startup or runtime
- Repeated HTTP 5xx in access logs

## 6. Container and deployment signals

- Restart count, OOMKilled, CrashLoopBackOff, non-zero exit codes (kube-state-metrics, cAdvisor or Docker events)
- Current vLLM image/version per model, as a label or info metric, so incidents line up with upgrades
- Deployment annotations in Grafana for every rollout

## 7. Synthetic probe

- Per-model success of a tiny chat completions and responses request every 60 s
- Probe latency
- Catches "server up, model broken" — the main failure mode after vLLM upgrades

## 8. Alert catalog

| Alert | Severity | Based on | Condition (starting point) |
| --- | --- | --- | --- |
| Model probe failing | Critical | Probe success | Failing 2 min |
| Queue backed up | Warning | Requests waiting | > 5 for 5 min |
| KV cache saturated | Warning | Cache usage + preemptions | > 95% and preemptions rising, 10 min |
| Slow first token | Warning | TTFT p95 | Well above baseline, 10 min |
| Error/abort spike | Critical | Success by finish reason | Aborts > 5%, 5 min |
| Model hung | Critical | Generation tokens + waiting | 0 tokens/s while waiting > 0, 3 min |
| Post-upgrade regression | Critical | Version change + any health alert | Within 1 h of version change |
| GPU Xid error | Critical | Xid metric / kernel log | Any new error |
| Uncorrectable ECC | Critical | DBE count | Any increase |
| Correctable ECC rising | Warning | SBE count | > 100 in 1 h |
| Row remap failure | Critical | Remap failure | > 0 |
| NVLink errors rising | Warning | NVLink CRC/replay | Any increase in 15 min |
| Overheating / throttling | Warning | Temp + throttle reasons | > 85 °C or throttled, 10 min |
| GPU idle with queue | Critical | Engine active + waiting | < 5% while queue > 0, 5 min |
| NCCL timeout | Critical | Container logs | Any in 5 min |
| vLLM engine dead | Critical | Container logs | Any in 5 min |
| Fabric Manager down | Critical | journald | Any error in 5 min |
| Container restarting | Critical | Restart count | > 2 in 15 min |

Write each rule once using the `model` and `node` labels so it covers every model automatically; keep a small per-model overrides file for thresholds like TTFT.

## 9. Suggested dashboard layout (per model)

1. Health: probe success, current vLLM version, restarts, active alerts
2. Load: requests running vs waiting, KV cache usage, preemptions
3. Latency: TTFT p50/p95, inter-token latency, end-to-end latency, abort rate
4. GPUs: engine activity per GPU, temperature, power, throttle reasons
5. Hardware errors: Xid, ECC, NVLink, row remaps
6. Logs: filtered NCCL / Xid / vLLM error panel with rollout annotations