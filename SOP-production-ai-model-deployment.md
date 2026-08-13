# SOP: Production AI Model Deployment (On-Prem, Disconnected)

**Scope:** Standing up and operating production LLM inference on on-prem NVIDIA B200 / B300 (Blackwell / Blackwell Ultra) clusters in a **disconnected (air-gapped) environment**, using vLLM as the serving engine, with a defined fast path from new model drops to production for **agentic workloads**, and a migration plan off `/v1/chat/completions` to the stateful agentic API standard (`/v1/responses`).

**Audience:** Platform / ML infrastructure engineers.
**New to LLM inference?** Read the companion [PRIMER.md](./PRIMER.md) first — it explains every concept and acronym used here in plain language ([published copy](https://claude.ai/code/artifact/d8e30be6-ccf1-4e4a-8df8-169a12b86541)).
**Status:** v1.0 — August 2026. Review quarterly; vLLM moves fast (current stable at time of writing: v0.26.x).

---

## 1. Hardware Baseline: B200 vs B300

| | B200 (Blackwell) | B300 (Blackwell Ultra) |
|---|---|---|
| HBM per GPU | 192 GB HBM3e | 288 GB HBM3e |
| Native low-precision | FP8, **NVFP4** | FP8, **NVFP4** (~1.5× dense FP4 FLOPS vs B200) |
| Typical node | HGX B200 (8× GPU, NVLink 5) | HGX B300 / GB300 NVL72 |
| Sweet spot | Dense models ≤ 200B, quantized MoE up to ~700B on one node | Huge-context agents, frontier MoE (DeepSeek-class) on fewer nodes |

Operational implications:

- **NVFP4 is the headline feature of this generation.** Blackwell has hardware FP4 tensor cores; a 400B-class MoE in NVFP4 fits on 4× B200/B300 that would need 8+ GPUs in BF16. Standardize on NVFP4 (or FP8) checkpoints as the *default* production format, BF16 only for accuracy-sensitive baselining.
- **B300's 288 GB changes the KV-cache math, not just the weights math.** Agentic workloads are KV-cache-heavy (long tool-use traces, large system prompts, many concurrent sessions). Extra HBM → more concurrent sequences at long context → materially higher goodput per node. Prefer B300 nodes for the agentic serving pool; B200 for batch/offline and smaller models.
- **Fabric:** NVLink within the node; InfiniBand NDR/XDR or RoCEv2 with full-mesh connectivity between nodes. Multi-node MoE serving (wide-EP/DeepEP) requires GPU-initiated RDMA (IBGDA over NVSHMEM) — validate the fabric *before* you need multi-node.
- **Driver/CUDA floor:** Blackwell requires CUDA 12.8+; keep the enclave on the driver version matching your frozen vLLM container (see §6). Run the DCGM diagnostics suite as part of node acceptance.

---

## 2. vLLM Technique Catalog (what to turn on, and when)

vLLM is the standard engine here. The canonical, tested per-model configurations live in **[vllm-project/recipes](https://github.com/vllm-project/recipes)** (rendered at recipes.vllm.ai) — structured YAML per model with the exact validated `vllm serve` command per GPU configuration. **Rule: start from the recipe, never from scratch.** Mirror this repo into the enclave (§6).

### 2.1 Quantization

| Format | When | Notes |
|---|---|---|
| **NVFP4** | Default for Blackwell production | Use NVIDIA ModelOpt-produced checkpoints (many published pre-quantized). For FP4 MoE, set `VLLM_USE_FLASHINFER_MOE_FP4=1` to enable the FlashInfer FP4 MoE kernel. |
| **FP8 (W8A8)** | When no NVFP4 checkpoint exists yet at model drop | Near-lossless; often the day-0 format. |
| **KV cache FP8** | Almost always for agentic serving | `--kv-cache-dtype fp8` roughly doubles KV capacity → more concurrent long sessions. Validate on your evals; a few models regress. |
| BF16 | Accuracy baseline, eval reference | Never the production default on Blackwell. |

### 2.2 Parallelism

- **TP (`--tensor-parallel-size`)** — default within a node for dense models. TP=4 or 8 on an HGX node. Don't extend TP across nodes.
- **PP (`--pipeline-parallel-size`)** — only when a model can't fit with TP at your quantization; adds latency.
- **EP + DP (`--enable-expert-parallel`, `--data-parallel-size`)** — for MoE models (DeepSeek-class, Qwen MoE, etc.): experts sharded across GPUs, attention replicated via DP.
- **Wide-EP (multi-node EP)** — EP across 2+ nodes with expert load balancing (EPLB), using **DeepEP** communication kernels: `deepep_high_throughput` backend for prefill, `deepep_low_latency` for decode. This is the highest-throughput way to serve big MoEs, and the operational deep end — requires NVSHMEM/IBGDA fabric validation. Adopt only after single-node NVFP4 stops meeting demand.

### 2.3 KV cache & prefix reuse — the agentic multipliers

- **Automatic prefix caching** (on by default): agentic workflows resend a large, mostly-identical prefix (system prompt + tool schemas + conversation so far) every turn. Prefix cache hits skip that prefill entirely. This is the single biggest win for agent latency — protect it:
  - Keep system prompts and tool definitions byte-stable (no timestamps in prompts).
  - Use **session-affinity / prefix-aware routing** at the load balancer so a session keeps hitting the replica that holds its cache (llm-d's inference scheduler does this natively; with plain round-robin you throw most of the benefit away).
- **KV offload / tiered cache (LMCache)** — spill cold KV to CPU RAM/NVMe so long-idle agent sessions don't evict hot cache. Worth it once concurrent long-lived sessions exceed HBM KV capacity.
- **Chunked prefill** (default on): keeps decode latency stable while long prefills stream through; tune `--max-num-batched-tokens` (8k–32k typical on Blackwell) to trade TTFT vs inter-token latency.

### 2.4 Latency techniques

- **Speculative decoding** — `--speculative-config`: MTP (built-in for models shipping MTP heads, e.g. DeepSeek), EAGLE-3 draft heads, or n-gram for free wins on repetitive agent output (tool-call JSON!). Typically 1.5–2.5× decode speedup at low concurrency; benefit shrinks at high batch — enable on the latency-sensitive interactive pool.
- **CUDA graphs / compilation** — default piecewise CUDA graphs are fine; use full CUDA-graph compilation mode for small models where launch overhead dominates.
- **Async scheduling** (default in current vLLM) — overlaps CPU scheduling with GPU execution; leave on.

### 2.5 Agent-critical serving features

- **Tool calling:** `--enable-auto-tool-choice --tool-call-parser <model-specific>` — the parser must match the model family (hermes, llama, qwen, deepseek_v3, seed-oss, …). A missing/wrong parser is the #1 cause of "new model breaks our agents."
- **Reasoning models:** `--reasoning-parser <parser>` to separate thinking from content; agent frameworks depend on this separation.
- **Structured output** (xgrammar-backed, default): guarantees schema-valid JSON for tool arguments. Enforce `response_format`/tool schemas from the client for all machine-parsed output.
- **Disaggregated prefill/decode (P/D)** — separate prefill and decode GPU pools with KV transfer over **NIXL**. Fixes the specific failure mode where long agent-context prefills stall decode for everyone (ITL spikes). Adopt when: strict TTFT+ITL SLOs, high concurrency, long mixed contexts — and run it via llm-d/Dynamo, not hand-rolled (§4).

### 2.6 Standard serving profiles

Maintain exactly these three profiles per model in the config repo; every deployment names one:

1. **`interactive-agent`** — latency-first: moderate `--max-num-seqs`, speculative decoding on, KV FP8, prefix-aware routing, headroom (`--gpu-memory-utilization 0.90`).
2. **`throughput-batch`** — goodput-first: high batch, no speculation, larger `--max-num-batched-tokens`; used for offline/eval/batch agents.
3. **`long-context`** — B300 pool, KV FP8 + LMCache offload, conservative concurrency caps.

Example (single-node interactive profile, 8× B200, MoE model):

```bash
VLLM_USE_FLASHINFER_MOE_FP4=1 \
vllm serve /models/org/model-NVFP4 \
  --served-model-name prod-agent-v1 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser <parser> \
  --reasoning-parser <parser> \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

(Exact flags come from the mirrored recipes repo for that model — this is a shape, not a copy-paste.)

---

## 3. Docker vs Kubernetes: the decision

### 3.1 The tiers

**Tier 1 — Docker + systemd (1–2 nodes, 1–3 models).**
vLLM container per model, pinned by digest, `systemd` unit with restart policy, node-local Prometheus. Simplest thing that is genuinely production-grade at small scale. A single HGX B200/B300 node serving one NVFP4 model this way is a perfectly respectable production system.

**Tier 2 — Docker Compose + gateway (one to a few nodes, several models).**
Compose stack: N vLLM containers + an OpenAI-compatible gateway (LiteLLM or the agentic-api gateway, §7) + Prometheus/Grafana. Gives you one endpoint, per-model routing, API keys, and basic blue/green (bring up new container, flip gateway route, drain old). Still no scheduler — humans place models on GPUs.

**Tier 3 — Kubernetes (+ GPU Operator, and llm-d / KServe on top).**
Standard 2026 stack: Kubernetes + NVIDIA GPU Operator, **llm-d** (CNCF Sandbox; Kubernetes-native vLLM orchestration: prefix-aware inference scheduling via Gateway API Inference Extension, P/D disaggregation, wide-EP as "well-lit paths"), optionally KServe for the model-serving CRD layer and Kueue for GPU scheduling/quota. NVIDIA **Dynamo** is the alternative orchestration layer for P/D disaggregation that runs above vLLM outside Kubernetes; prefer llm-d if you are on K8s, since it's a first-class K8s citizen using standard CRDs.

### 3.2 When you actually need Kubernetes

You **need** K8s when any of these are true:

1. **Multi-node serving** — wide-EP or P/D disaggregation across nodes needs orchestrated pod placement, RDMA device plugins, and topology-aware scheduling.
2. **Fleet of models with churn** — >~5 models, frequent adds/retires, GPUs as a shared pool with quotas across teams (Kueue).
3. **Autoscaling / bin-packing** — scaling replicas with demand, mixing day-interactive with night-batch on the same GPUs.
4. **Zero-downtime rollout as a hard requirement** — rolling upgrades, canary weighting, automated rollback as platform primitives rather than runbook steps.
5. **Smart routing at scale** — prefix-cache-aware and load-aware routing across many replicas (llm-d's scheduler) measurably beats round-robin for agent traffic.

You **don't** need K8s when: ≤ a couple of nodes, a stable small model set, sessions can tolerate a 2–5 min maintenance window, and one team owns the boxes. In a disconnected environment, K8s is a significant standing cost (airgapped cluster upgrades, image mirroring for the control plane itself, etcd care) — don't pay it before Tier 2 actually hurts.

**Recommendation:** Start Tier 2 even if you own 4+ nodes. Define the migration trigger now (e.g., "3rd concurrent production model, or first multi-node model, or first hard zero-downtime SLA") and treat Tier 3 as a planned project, not an emergency. Everything below (registry, model store, profiles, gateway) is designed to carry over unchanged.

---

## 4. Disconnected (Air-Gapped) Operations

### 4.1 Architecture: two zones, one gate

```
[Connected staging zone]                      [Enclave]
  pull + verify + freeze          transfer      run only
  ─ HF model mirror (hf CLI)     ────────►     ─ Harbor registry (images)
  ─ Harbor staging registry       (media /     ─ Model store (NFS/PVC)
  ─ PyPI mirror snapshot           data diode  ─ PyPI mirror (devpi)
  ─ eval + scan + sign             / CDS)      ─ Git mirror (recipes, configs)
                                               ─ Prometheus/Grafana/PKI
```

Non-negotiable rules:

- **Nothing inside the enclave references an external host.** Every image reference in every manifest/compose file points at the internal Harbor. `HF_HUB_OFFLINE=1` on every vLLM process; models are served from **local paths** (`vllm serve /models/...`), never hub IDs.
- **Everything is content-addressed and signed.** Container images pinned by digest and signed (cosign); model weights carried with SHA-256 manifests generated on the staging side and re-verified after transfer, before first load.
- **Weights are files, not images.** Do not bake weights into containers. They live in the model store (NFS export or K8s PVC), laid out as `/models/<org>/<name>-<revision>-<quant>/`. Images stay small and generic; models version independently.
- **Version-pin the world.** vLLM image tag+digest, driver version, NCCL, PyPI snapshot date — recorded per release in the config repo. An enclave with no internet has no "pip will fix it later."

### 4.2 The transfer pipeline (bundle format)

Every transfer across the gap is one **release bundle**:

```
bundle-2026-08-12-modelX/
  manifest.yaml          # contents, digests, versions, approvals
  images/                # skopeo/oras OCI archives (vLLM, gateway, exporters)
  models/                # weights dir + per-file sha256
  recipes/               # git bundle of config repo + mirrored vllm recipes
  evals/                 # eval results from staging (see §5) — evidence, not promise
  signatures/
```

Staging side: `hf download` → quantize/verify (or fetch pre-quantized NVFP4) → run eval gate → `skopeo copy` images to archive → sign → checksum. Enclave side: verify signatures/checksums → `skopeo copy` into Harbor → rsync weights into model store → the bundle's `manifest.yaml` becomes the deployment record.

### 4.3 Enclave standing services

Harbor (images), devpi or equivalent frozen PyPI mirror (for tooling only — production never pip-installs), internal git (configs + mirrored `vllm-project/recipes`), Prometheus + Grafana + Alertmanager, internal PKI/CA (vLLM and gateway behind TLS), NTP. Log aggregation stays inside (Loki/ELK). No telemetry egress: set `VLLM_NO_USAGE_STATS=1` / `DO_NOT_TRACK=1` on all engines.

---

## 5. Fast Path: Model Drop → Production Agents

Goal: **new open-weights model drop to production agentic traffic in 48–72 h**, gated by evals, not vibes. The way to make it fast is to make it *boring* — everything below is pre-built, so a model drop only fills in parameters.

### Phase 0 — standing readiness (do once, maintain)

- Keep a **current vLLM container** in Harbor at all times (update on a monthly cadence even with no model drop — day-0 model support lands in vLLM releases/nightlies, and the most common blocker is "our frozen vLLM predates this architecture"). Big labs now coordinate day-0 vLLM support for major drops (e.g. Nemotron, DeepSeek releases), so a fresh engine usually means the model just works.
- Pre-authorized **expedited transfer window** with your security office for release bundles (the gap crossing is usually the schedule-killer, not the tech).
- Standing **agentic eval suite** (see gate below) and a `guidellm` load-test harness, both runnable inside the enclave.
- GPU headroom: one reserved canary slice (≥1 node) that never carries production traffic.

### Phase 1 — intake on the connected side (hours 0–12)

Checklist, in order (each item is a hard gate):

1. **vLLM support?** Check the model's recipe in `vllm-project/recipes` and release notes. Native support → proceed. Transformers-backend-only → expect ~2–4× slower; acceptable for eval, usually not for production — note the follow-up.
2. **Quantized checkpoint?** Prefer official/ModelOpt NVFP4; else FP8; else quantize in staging (llm-compressor/ModelOpt) — budget +6–12 h.
3. **Tool-call parser + reasoning parser exist and match?** Test with 20 canned tool-call prompts *on the staging GPU* before shipping anything. No working parser → the model is not agent-ready regardless of benchmarks.
4. **Chat template sanity** — confirm the shipped template handles system prompts, multi-turn, and parallel tool calls.
5. Pull weights, hash, run the **staging eval gate**, build the bundle.

### Phase 2 — transfer + enclave canary (hours 12–48)

Bundle crosses the gap → verify → weights to model store → deploy to the **canary slice** under a new `--served-model-name` (e.g. `candidate-modelX`). Run inside the enclave:

- **Agentic eval gate (blocking):** tool-selection accuracy on your real tool schemas, multi-turn task completion on 20–50 golden agent trajectories, structured-output validity rate (must be ≈100% with guided decoding), long-context retrieval probes at your real context lengths, refusal/safety spot checks. Compare against the incumbent model's recorded scores.
- **Performance gate (blocking):** `guidellm` sweep at production concurrency; record TTFT p50/p99, ITL p99, throughput at SLO. Must fit the target profile's SLOs on target hardware.

### Phase 3 — promote (hours 48–72)

- Register in the gateway under an **alias** (`agent-default` → `candidate-modelX`) at 5% traffic weight → 25% → 100% over hours-to-a-day, watching eval-adjacent runtime metrics (tool-call parse failure rate, retry rate, task abandonment).
- Keep the previous model warm for 1 week for instant rollback (alias flip = rollback; no redeploy).
- **Clients only ever call aliases** (`agent-default`, `agent-long-context`), never model names. This is what makes promotion a config change instead of a client migration.

---

## 6. API Layer: Migrating off `/v1/chat/completions`

### 6.1 Why move

`/v1/chat/completions` is stateless: every agent turn re-sends the entire conversation, and the *client* owns the agent loop. For agentic workloads this costs you: (a) huge repeated payloads and prefill (mitigated but not eliminated by prefix caching), (b) no server-side home for reasoning items between turns — many reasoning models lose quality when clients drop reasoning content on replay, (c) every team reimplements the tool loop differently, (d) no server-side conversation forking/audit.

The industry standard for agents is the **OpenAI Responses API (`/v1/responses`)**: stateful (`previous_response_id` continues a conversation server-side), first-class typed items (reasoning, tool calls, tool outputs), and server-side tool execution. Anthropic's `/v1/messages` is the second protocol worth supporting (Claude Code and other Anthropic-native clients speak it).

### 6.2 Target architecture (on-prem, disconnected)

```
Agent clients ──► agentic gateway (vllm-project/agentic-api)
                    /v1/responses   (OpenAI Responses: state, tools, streaming)
                    /v1/messages    (Anthropic protocol — Claude-Code-style clients)
                    /v1/chat/completions (passthrough, legacy during migration)
                    state store: SQLite → move to Postgres-class backing at fleet scale
                          │
                          ▼
                   vLLM replicas (per-model pools, per §2 profiles)
```

Two supported deployment shapes:

- **vLLM built-in:** current vLLM serves `/v1/responses` natively (usable with the official OpenAI client). Fine for single-replica; state is engine-local, so it doesn't survive restarts or span replicas.
- **Gateway (recommended for production):** **[vllm-project/agentic-api](https://github.com/vllm-project/agentic-api)** — a Rust gateway in front of vLLM (launchable as `vllm serve <MODEL> --agentic-api`, or standalone). It owns response persistence (`previous_response_id` hydration), streaming over SSE/WebSocket, and an explicit **tool-ownership model**: each tool is *gateway-owned* (executed server-side — in the enclave these must be your internal tools only; disable any internet-backed built-ins like web search), *client-owned* (returned to the client to execute, Codex-style), or *provider-owned* (passed through to vLLM). It also exposes `/v1/messages`. Run it as its own container in front of the per-model pools; put the state DB on persistent storage and back it up — it is now production state.

### 6.3 Migration plan (dual-stack, alias-gated)

1. **Stand up the gateway in passthrough mode.** All existing `/v1/chat/completions` traffic flows through it unchanged. You gain: one choke point, uniform authn, per-client metrics — and a place to watch migration progress.
2. **Enable `/v1/responses` alongside.** Same model aliases resolve on both endpoints.
3. **Migrate agents newest-first.** Client change is mechanical: `client.responses.create(model=..., input=..., previous_response_id=...)` replaces rebuilding `messages[]` each turn; `tools` schemas carry over nearly as-is; parse typed output items instead of `choices[0].message`. Reasoning-model agents move first (they benefit most — server-side reasoning persistence between turns).
4. **Move tool execution server-side selectively.** Start with everything client-owned (behavior-identical to today), then promote your stable internal tools (retrieval, code-exec sandbox, internal APIs) to gateway-owned to cut round-trips per agent step.
5. **Deprecate:** per-client metrics from step 1 tell you when chat-completions traffic hits zero; then freeze it (410 for new API keys, grandfather old ones for one release cycle). Keep the endpoint code — batch/eval tooling (guidellm etc.) still speaks it, and that's fine; the *agent* path is what must move.
6. **State ops:** define retention (e.g. purge response state > 30 days), include the state DB in backups and in DR tests, and document that restoring the gateway without its DB orphans all `previous_response_id` references (clients must handle a `response not found` by replaying full history — build that fallback into your agent SDK wrapper from day one).

---

## 7. Observability & SLOs

- **Golden signals per model pool:** TTFT p50/p99, inter-token latency p99, requests in RUNNING/WAITING, KV-cache utilization %, **prefix-cache hit rate** (watch it after any router or prompt change — a drop is an incident), tool-call parse failure rate (from gateway logs), GPU utilization + HBM headroom (DCGM exporter).
- vLLM exposes Prometheus metrics natively (`/metrics`); scrape every replica + the gateway + DCGM into the enclave Prometheus. Standard Grafana dashboards live in the config repo and ship in every bundle.
- **SLO template:** interactive-agent profile — TTFT p99 ≤ 2 s at up to N concurrent sessions, ITL p99 ≤ 60 ms, structured-output validity ≈ 100%. Alert on SLO burn rate, not raw utilization.
- **Capacity rule of thumb:** run `guidellm` against every new (model, profile, hardware) tuple and record "max concurrency at SLO" in the recipe file. That number, not GPU count, is your capacity unit.

---

## 8. Runbooks (index)

Maintain these as separate one-pagers in the config repo; each bundle release links the versions it was tested with:

1. **Engine upgrade** — new vLLM image: canary node → eval gate re-run (parsers break silently across versions — always re-run tool-call tests) → rolling replace.
2. **Model promotion / rollback** — alias flip procedure (§5 Phase 3).
3. **Node loss** — drain, redistribute (Tier 2: manual placement map; Tier 3: scheduler handles it), verify prefix-cache warmup before returning to full traffic weight.
4. **Bundle transfer** — the §4.2 checklist with sign-off fields.
5. **Incident: latency spike** — first checks: WAITING queue depth (saturation) → prefix hit rate (routing regression) → long-prefill interference (consider P/D split if chronic).

---

## Appendix A — Decision quick-reference

| Question | Default answer |
|---|---|
| Serving engine | vLLM, config from mirrored `vllm-project/recipes` |
| Weight format | NVFP4 (ModelOpt) → FP8 fallback → BF16 baseline only |
| KV cache | FP8, prefix caching on, session-affinity routing |
| Orchestration | Docker Compose + gateway until a §3.2 trigger fires → K8s + llm-d |
| Multi-node MoE | llm-d wide-EP well-lit path; DeepEP HT (prefill) / LL (decode); NIXL for P/D |
| API for agents | `/v1/responses` via agentic-api gateway; `/v1/messages` for Anthropic-protocol clients; chat-completions = legacy passthrough |
| Model naming | Clients call **aliases**, promotion = alias flip |
| Air-gap posture | Harbor by digest + signed bundles + `HF_HUB_OFFLINE=1` + local-path serving |

## Appendix B — Sources / further reading

- vLLM recipes catalog: https://github.com/vllm-project/recipes (mirror internally)
- vLLM OpenAI-compatible & Responses server docs: https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/
- Agentic API gateway: https://github.com/vllm-project/agentic-api
- Expert-parallel deployment: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- llm-d well-lit paths (wide-EP, P/D disaggregation): https://llm-d.ai/
- Wide-EP at scale (vLLM blog): https://vllm.ai/blog/2025-12-17-large-scale-serving
- GB300 + DeepSeek results (vLLM blog): https://vllm.ai/blog/2026-02-13-gb300-deepseek
- Red Hat: scaling MoEs with vLLM + llm-d wide-EP: https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep
- NVIDIA Blackwell Ultra architecture: https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
- NVIDIA NIM air-gap deployment pattern (transferable practices): https://docs.nvidia.com/nim/large-language-models/2.0.0/deployment/air-gap-deployment.html
- GuideLLM benchmarking (incl. air-gapped): https://developers.redhat.com/articles/2025/09/15/benchmarking-guidellm-air-gapped-openshift-clusters
