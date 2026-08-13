# KV Cache: The Real Currency of Agentic Serving

**Deep dive #06 — companion to the [deployment SOP](../SOP-production-ai-model-deployment.md) and [PRIMER](../PRIMER.md).**
**Environment assumed:** on-prem, air-gapped, NVIDIA B200 (192 GB) / B300 (288 GB) GPUs, vLLM (v0.26.x as of August 2026), agentic workloads.
**Freshness note:** everything version-sensitive in this document is tagged with the version or date it was verified against. The KV-cache ecosystem moved fast through 2025–2026; re-verify tagged items when you bump your frozen vLLM image.

---

## Key takeaways

- On an agentic serving pool, most GPU memory is spent remembering conversations (the KV cache), not storing the model. Capacity planning is KV-cache planning.
- Per-token KV size is a closed-form function of the model config: `2 × layers × kv_heads × head_dim × dtype_bytes`. You can compute it from `config.json` before you ever load the model.
- Architecture dominates the math: Llama-3.3-70B (grouped-query attention) needs ~320 KB per token in BF16, Qwen3-235B needs ~188 KB, and DeepSeek-V3/R1 (multi-head latent attention) needs only ~69 KB — a 671B model with cheaper memory-per-token than a 70B model.
- vLLM's PagedAttention stores KV in fixed 16-token blocks addressed through a per-request block table, which eliminates fragmentation and makes block sharing (prefix caching) possible.
- Prefix caching identifies reusable blocks by a hash chain: each block's hash covers its parent's hash plus its own tokens plus "extra keys" (LoRA ID, multimodal input hashes, `cache_salt`). One changed byte early in the prompt invalidates everything after it — this is why a timestamp in a system prompt is poison.
- Measure hit rate as `rate(vllm:prefix_cache_hits) / rate(vllm:prefix_cache_queries)`. Warmed agent loops should sit at 70%+; coding-agent traces measure ~96% token-weighted in published studies. Round-robin routing across replicas can crush this to single digits.
- Cache-aware routing is a tiered decision: consistent-hash session affinity (nginx/envoy/vllm-router) at the Compose tier; llm-d's inference scheduler with prefix-cache scoring — optionally fed by real-time KV events from vLLM — on Kubernetes.
- When sessions outlive HBM (high-bandwidth memory) capacity, offload cold KV instead of recomputing it: vLLM's native OffloadingConnector (CPU, filesystem, object-store tiers, v0.26) or LMCache (CPU/NVMe/remote, richer feature set). Loading KV from CPU beats re-prefill by 2–22× on TTFT in vLLM's published numbers.
- `--kv-cache-dtype fp8` halves KV per token and roughly doubles concurrent long sessions; it composes with everything else (prefix caching, offload, MLA). Validate per model on your evals.
- The cheapest lever is usually at the application layer: byte-stable prompts, truncated tool outputs, and deliberate context compaction do more for capacity than any flag.

---

## 1. What the KV cache actually is

### 1.1 The concept

A transformer generates one token at a time, and every new token must "attend to" every previous token. Attention works on two derived vectors per token per layer: a **key (K)** vector (what this token offers for matching) and a **value (V)** vector (what this token contributes if matched). Recomputing K and V for the whole history on every generation step would be quadratic waste, so the engine computes them once and keeps them in GPU memory. That stored set is the **KV cache** — literally the model's working memory of the conversation.

Two properties make it the central resource of agentic serving:

1. **It grows linearly with context length**, per conversation, per user. A 128k-token agent trace is gigabytes of KV, and agents routinely run many concurrent long sessions.
2. **It is re-derivable but expensive.** If you throw KV away you can always recompute it from the tokens (that is exactly what "prefill" is), but at long context a re-prefill costs tens of seconds of GPU time. Every mechanism in this document — paging, prefix caching, cache-aware routing, offload — exists to avoid paying that recompute.

### 1.2 The sizing formula

For standard attention variants, the per-token KV footprint is:

```
bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes
                  │
                  └── the 2 is "one K vector + one V vector"
```

- `n_layers` — every transformer layer keeps its own K and V (`num_hidden_layers` in `config.json`).
- `n_kv_heads` — the number of **KV heads** (`num_key_value_heads`), which is *not* the same as the number of attention heads (see 1.3).
- `head_dim` — vector width per head (`head_dim`, or `hidden_size / num_attention_heads`).
- `dtype_bytes` — 2 for BF16 (bfloat16, 16-bit), 1 for FP8 (8-bit floating point).

Note what is **absent** from the formula: quantizing the *weights* (NVFP4, FP8-W8A8) does not shrink the KV cache at all. Weights math and KV math are independent budgets; only `--kv-cache-dtype` changes the KV term.

### 1.3 How GQA, MQA, and MLA change the math

- **MHA (multi-head attention)** — the original design: every attention head has its own K and V. `n_kv_heads = n_attention_heads` (e.g. 64). Maximal KV cost; essentially extinct in new frontier models for this reason.
- **GQA (grouped-query attention)** — groups of query heads share one KV head. Llama-3.3-70B has 64 query heads but only 8 KV heads: an 8× KV reduction versus MHA with no formula change, just a smaller `n_kv_heads`.
- **MQA (multi-query attention)** — the extreme: all query heads share a single KV head (`n_kv_heads = 1`). Used by older models like Falcon; maximal savings, measurable quality cost, mostly superseded by GQA.
- **MLA (multi-head latent attention)** — DeepSeek's design (introduced in DeepSeek-V2, used in V3/R1 and successors). Instead of caching K and V at all, the model caches a single compressed **latent vector** per token per layer, from which all heads' K and V are reconstructed on the fly. The formula changes shape:

```
MLA: bytes_per_token = n_layers × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
```

For DeepSeek-V3/R1 that is `kv_lora_rank = 512` plus a small shared rotary-position key of `qk_rope_head_dim = 64`, i.e. **576 numbers per token per layer** — no factor of 2, no per-head multiplier. This is why a 671-billion-parameter model can have a *smaller* KV footprint per token than a 70B GQA model.

One more 2026 wrinkle: hybrid models (sliding-window attention layers, Mamba-style layers mixed with full attention) keep less than the full history for some layers. vLLM v0.26 made sliding-window an explicit attention-backend capability and lets you select a different attention backend per KV-cache group, so the engine handles the bookkeeping — but it means the simple formula above is an upper bound for such models.

### 1.4 Worked examples (verified against each model's `config.json`, August 2026)

| Model | Layers | KV shape per layer | KV/token BF16 | KV/token FP8 |
|---|---|---|---|---|
| Llama-3.3-70B (GQA) | 80 | 8 heads × 128 | 320 KB | 160 KB |
| Qwen3-235B-A22B (MoE, GQA) | 94 | 4 heads × 128 | 188 KB | 94 KB |
| DeepSeek-V3/R1 (MoE, MLA) | 61 | 576-wide latent | 68.6 KB | 34.3 KB |

Sample calculation (Llama-3.3-70B, BF16): `2 × 80 × 8 × 128 × 2 = 327,680 bytes ≈ 320 KB per token`. For DeepSeek: `61 × 576 × 2 = 70,272 bytes ≈ 68.6 KB per token`.

Flip it into "how many tokens does one gigabyte hold":

| Model | Tokens/GB (BF16) | Tokens/GB (FP8 KV) | One 128k session (FP8 KV) |
|---|---|---|---|
| Llama-3.3-70B | ~3,050 | ~6,100 | 21.5 GB |
| Qwen3-235B-A22B | ~5,200 | ~10,400 | 12.6 GB |
| DeepSeek-V3/R1 | ~14,200 | ~28,500 | 4.6 GB |

("128k session" = 131,072 tokens. Caveat: Qwen3-235B-A22B's base config ships `max_position_embeddings = 40,960`; reaching 131,072 requires the YaRN rope-scaling configuration from the model card — check the recipe.)

**How many 128k agent sessions fit on real hardware?** Take the HBM budget (`HBM × --gpu-memory-utilization`, default 0.90 in our profiles), subtract weights and runtime overhead (activations, CUDA graphs — read the true number from vLLM's startup log), and divide. Approximate weights: Llama-70B NVFP4 ≈ 40 GB; Qwen3-235B NVFP4 ≈ 125 GB; DeepSeek-R1 NVFP4 ≈ 400 GB (checkpoint sizes vary a few percent by producer — treat as estimates).

| Deployment (NVFP4 weights, FP8 KV) | KV pool ≈ | 128k sessions ≈ |
|---|---|---|
| Llama-3.3-70B on 1× B200 (192 GB) | ~125 GB | ~6 |
| Llama-3.3-70B on 1× B300 (288 GB) | ~210 GB | ~10 |
| Qwen3-235B on 8× B200 (1,536 GB) | ~1,215 GB | ~96 |
| Qwen3-235B on 8× B300 (2,304 GB) | ~1,910 GB | ~150 |
| DeepSeek-R1 on 8× B300 (2,304 GB) | ~1,620 GB | ~350 |

Three lessons fall out of this table. First, the B300's extra 96 GB per GPU goes almost entirely to KV once weights fit, which is why the SOP routes the agentic pool to B300 (a ~70% session-capacity bump for the Llama case). Second, MLA is a step change: DeepSeek-class models are *architecturally* cheap to serve at long context. Third, these are worst-case numbers — prefix caching means concurrent sessions sharing a system prompt don't each pay full price, and most sessions are far shorter than their maximum.

---

## 2. PagedAttention: block mechanics

### 2.1 The problem it solves

Before vLLM, engines reserved one contiguous KV buffer per request, sized for the maximum possible length. Most of it sat empty (internal fragmentation), and freed buffers of odd sizes couldn't be reused (external fragmentation) — real-world waste was measured at 60–80% of KV memory. **PagedAttention** applies the operating-system idea of paged virtual memory: chop KV storage into small fixed-size **blocks** (pages) and give each request a **block table** (page table) mapping its logical token positions to physical blocks scattered anywhere in GPU memory.

### 2.2 The mechanics

- **Block size:** default **16 tokens** on CUDA (`--block-size`, choices 1–128 in powers of two). One block for Llama-3.3-70B at FP8 KV is `16 × 160 KB = 2.6 MB`. Bigger blocks mean less block-table overhead but coarser sharing granularity; 16 is right for almost everyone — change it only if a recipe says so.
- **Block table:** an append-only list per request. When a request needs space for its next tokens, the KV-cache manager pops a free block and appends it to the table. Waste is bounded to the unused tail of the *last* block (< 16 tokens per sequence), so utilization is effectively ~100%.
- **Sharing and reference counts:** a physical block can appear in many requests' block tables. Each block carries a reference count (`ref_cnt`); a block is only reclaimable when no live request maps it. This is the substrate prefix caching is built on: fifty agent sessions with the same 4,000-token system prompt map the same ~250 physical blocks.
- **Copy-on-write — a historical note:** vLLM's original (v0) engine used copy-on-write when two sequences sharing a block diverged (beam search, `n>1` sampling). The v1 engine dropped it: block tables are append-only, divergent sequences simply get their own new blocks, and short-lived duplicate blocks are reclaimed when a request frees. You will still see copy-on-write described in the PagedAttention paper and older docs; it is not how current vLLM works.

```
Request A: "You are an agent... [tool schemas] ... user: check disk"
Request B: "You are an agent... [tool schemas] ... user: restart svc"

logical blocks      A: [0][1][2][3a]         B: [0][1][2][3b]
                        │  │  │   │              │  │  │   │
physical blocks     ┌───┴──┴──┴───┴──────────────┴──┴──┘   │
(GPU HBM)           │  #12 #40 #7 (shared, ref_cnt=2)      │
                    │  #55 (A only)          #91 (B only) ◄┘
```

### 2.3 Why you care operationally

The block is the unit of *everything* downstream: prefix-cache hashing, eviction, offload transfers, and KV events for routing all speak in blocks. When metrics say "KV cache usage 87%", they mean 87% of physical blocks are mapped.

---

## 3. Prefix caching internals

### 3.1 The mechanism: a hash chain over blocks

**Automatic prefix caching (APC)** is on by default in current vLLM (disable with `--no-enable-prefix-caching`; you shouldn't). The design (vLLM design docs, verified August 2026):

Every *full* block gets a content hash the moment it fills:

```
block_hash = hash(parent_block_hash, block_tokens, extra_keys)
```

- `parent_block_hash` — the hash of the previous block in the sequence. This chaining means a block's identity encodes its *entire prefix*, not just its own 16 tokens. Two conversations with identical text in block 5 but different text in block 2 produce different hashes for block 5 — correctly, because attention makes block 5's KV depend on everything before it.
- `block_tokens` — the token IDs in this block.
- `extra_keys` — anything else that changes the KV or must partition the cache: **LoRA (low-rank adapter) ID** (same text through a different adapter produces different KV), **multimodal input hashes** (images are hashed by content, so the same image bytes reuse cache but a re-encoded/resized variant misses), and the per-request **`cache_salt`**.

Incoming requests are matched block-by-block from the start: tokenize, hash block 0, look it up; on hit, map the existing physical block (bump `ref_cnt`) and continue to block 1; the first miss ends reuse, and prefill runs only from that point. Partial blocks (the trailing < 16 tokens) are never cached.

**What is deliberately *not* in the hash:** sampling parameters. Temperature, top-p, max tokens, logit bias — none of them affect K/V values, so requests with different sampling settings share cache freely. Reuse is exact-match on *content and identity*, independent of *how you sample*.

### 3.2 Eviction: LRU over unreferenced blocks

When a request finishes, its blocks aren't zeroed — they keep their hashes and drop to `ref_cnt = 0`, joining a free queue ordered so that the **least recently used (LRU)** block is evicted first when new allocations need space. Note the elegance: "free" and "cached" are the same state. The entire idle KV pool is a cache, and eviction is invisible until a would-be hit gets evicted before reuse. vLLM (v0.26-era) exposes histograms — block lifetime, idle-time-before-evict, reuse-gap — that tell you whether your reuse window fits in HBM (see §4 and §6).

### 3.3 What invalidates reuse — the poison list

Because of hash chaining, **any byte change invalidates every block from that point onward**. The classic self-inflicted wounds:

- **Timestamps in system prompts.** `"Current time: 2026-08-12T09:14:33Z"` guarantees a 0% hit rate across requests: block 0 never matches, so nothing after it can. If the model needs the date, inject it *late* in the prompt (after the big stable prefix), at coarse granularity (day, not second) — or supply it via a tool.
- **Unstable serialization of tool schemas.** A tool list emitted from an unordered dict can reorder keys between processes. Same semantics, different bytes, zero reuse. Serialize canonically (sorted keys, fixed whitespace) and treat the rendered prompt as a build artifact.
- **Request IDs, trace IDs, user names, random few-shot sampling** anywhere in the early prompt.
- **Chat-template drift.** The template itself can inject dates (some models' default templates do) or change between vLLM versions. After any engine upgrade, diff the rendered prompt bytes (the render endpoint / `--kv-events` tooling in §5 helps here).
- **LoRA and salt partitions.** Different adapter ID → different cache namespace, by design. Likewise `cache_salt`.

`cache_salt` deserves a definition, because it is the *deliberate* version of invalidation: an optional per-request string, folded into block 0's hash, so only requests carrying the same salt can share cache. Its purpose is blocking **timing side-channels** in multi-tenant serving — without it, an attacker who can measure TTFT can probe whether someone else's prompt prefix is cached (added in vLLM PR #17045). Policy for our enclave: salt per tenant/organization boundary if you serve mutually distrusting tenants; do **not** salt per-user or per-request inside a single trusted agent platform, or you will silently destroy cross-session sharing of the system prompt.

```json
{"model": "agent-default", "cache_salt": "team-sigint", "messages": [...]}
```

### 3.4 Why agent loops are the perfect customer

An agent turn appends to an otherwise frozen transcript: system prompt + tool schemas + all prior turns are byte-identical, plus one new tool result. With APC, turn N's prefill cost is only the *new* tokens — the 60,000-token transcript costs nothing to "re-read." Published trace studies (TraceLab, 2026) measure token-weighted prefix hit rates of ~95.7% for coding-agent workloads (tool-result steps ~97.5%); multi-turn conversational workloads measure ~90%. This is the single biggest latency lever in the entire stack, which is why the SOP treats a hit-rate drop as an incident.

---

## 4. Measuring it

vLLM exposes Prometheus counters at `/metrics` (names verified against vLLM main, August 2026; older v1 builds may carry a `gpu_` prefix on the prefix-cache pair, e.g. `vllm:gpu_prefix_cache_queries` — check your pinned image):

- `vllm:prefix_cache_queries` — tokens (cumulative) that *could* have hit cache.
- `vllm:prefix_cache_hits` — tokens that actually did.
- `vllm:kv_cache_usage_perc` — fraction of KV blocks in use (0–1).
- `vllm:kv_block_lifetime_seconds`, `vllm:kv_block_idle_before_evict_seconds`, `vllm:kv_block_reuse_gap_seconds` — eviction-behavior histograms (v0.26-era).

The old hit-rate *gauge* is deprecated; compute rate-windowed hit rate yourself:

```promql
sum(rate(vllm:prefix_cache_hits[5m])) by (model_name)
  /
sum(rate(vllm:prefix_cache_queries[5m])) by (model_name)
```

**What good looks like for agent traffic** (as of mid-2026, from published measurements and our own SLO stance):

| Signal | Healthy | Investigate |
|---|---|---|
| Hit rate, warmed agent pool | ≥ 70% | < 50% |
| Hit rate, coding agents | 85–97% | < 70% |
| `kv_cache_usage_perc` sustained | < 0.90 | ≈ 1.0 + rising reuse-gap |
| Hit rate after router/prompt change | unchanged | any step drop |

Diagnostic logic: **low hit rate + low KV usage** means a content problem (something poisoned the bytes — §3.3) or a routing problem (requests landing on replicas that never saw the prefix — §5). **Low hit rate + pegged KV usage** means an eviction problem (working set exceeds HBM; sessions come back after their blocks died — §6/§7). The `kv_block_reuse_gap_seconds` vs `kv_block_idle_before_evict_seconds` pair distinguishes these directly: if typical reuse gaps are longer than idle-before-evict, you are recomputing exactly what you evicted, and offload will pay for itself.

Also watch preemptions in logs: under memory pressure the v1 scheduler **preempts by recompute** (drops a request's blocks and re-prefills it later). Recurrent preemption warnings are the engine telling you the concurrency/context budget is oversubscribed.

---

## 5. Cache-aware routing: getting requests to where their cache lives

Prefix cache lives in one replica's HBM. The moment you run two replicas behind a load balancer, a coin flip decides whether turn N lands where turns 1..N-1 built cache. Plain round-robin across R replicas caps your effective hit rate near `1/R` of potential — measured brutally in the Mooncake/vLLM agentic study: 1.7% hit rate with cache-blind placement vs 92.2% with cache pooling. Fix it at the tier you actually run.

### 5.1 Compose tier: session affinity by consistent hashing

The 90% solution is **sticky sessions**: hash a stable session key to pick the replica, so a session always returns to the same engine. nginx:

```nginx
upstream vllm_pool {
    hash $http_x_session_id consistent;   # ketama consistent hashing
    server vllm-0:8000;
    server vllm-1:8000;
    server vllm-2:8000;
}
```

Envoy's `ring_hash` policy on a session header is equivalent. Your agent gateway must send the header (one line in the SDK wrapper); `consistent` keeps most sessions in place when a replica joins or leaves. Since December 2025 there is also **`vllm-project/router`** — a Rust load balancer from the vLLM project with consistent-hashing, power-of-two-choices, and P/D-aware policies, explicitly supporting bare-metal (non-Kubernetes) fleets; it is the natural upgrade from nginx at the Compose tier and mirrors into Harbor like any other image.

Limitations of pure affinity: it balances *sessions*, not *load* (one replica can drown while others idle); it has no idea whether the cache actually survived (an eviction or restart turns "sticky" into "sticky to a cold replica"); and cross-session sharing (every session's identical system prompt) is only exploited within each replica.

### 5.2 Kubernetes tier: llm-d's inference scheduler

**llm-d** (the SOP's Tier-3 orchestration layer) routes through the Kubernetes **Gateway API Inference Extension (GAIE)** and its **EPP (endpoint picker / external processing pod)**: for each request, a set of **scorers** rank the candidate vLLM pods and the gateway forwards to the winner. The default profile combines a **`prefix-cache-scorer`** (how much of this request's prefix does each pod already hold?) with a **`queue-scorer`** (how backed up is each pod?), so you get cache locality *and* load balancing per request — strictly better than static affinity.

The prefix-cache scorer has two modes:

- **Approximate** (default): the EPP remembers which pod it sent which prefixes to and assumes the cache is still there. No vLLM configuration needed; accuracy decays under churn and eviction.
- **Precise (KV-events based):** each vLLM replica publishes **KV events** — block-stored/block-removed notifications — over ZeroMQ (ZMQ, a lightweight messaging library), and the EPP's **KV-cache indexer** maintains a live global map of which token blocks sit on which pod:

```bash
vllm serve /models/... \
  --kv-events-config '{"enable_kv_cache_events": true,
                       "publisher": "zmq",
                       "endpoint": "tcp://*:5557",
                       "topic": "kv-events"}'
```

The llm-d team's benchmark ("KV-Cache Wins You Can See", 2026; 150 tenants × 6k-token contexts at 73% of cluster KV capacity) measured P90 TTFT of **0.54 s with precise scheduling vs 31 s approximate vs 92.5 s random**, with roughly double the throughput of cache-blind routing — approximate scoring degrades exactly when the cluster is busiest, because that is when its assumptions about eviction are most wrong. Precise mode is the SOP's recommendation once you are on Kubernetes; the same KV events feed also exists standalone (vLLM `--kv-events-config`) if you ever build custom placement logic.

For our roadmap: nginx/`vllm-project/router` affinity now; llm-d EPP with precise prefix-cache scoring when a §3.2 SOP trigger moves us to Kubernetes. The two need the same client-side discipline (stable session key, byte-stable prompts), so nothing is wasted.

---

## 6. LMCache: the tiered-cache add-on, in depth

**LMCache** is an open-source KV-cache engine that bolts a storage hierarchy onto vLLM: keep hot KV in GPU HBM, spill warm KV to pinned CPU RAM, cold KV to local NVMe (non-volatile memory express — i.e., fast local SSD), and optionally to remote/shared backends (Redis/ValKey, Mooncake, Infinistore; NVIDIA GDS for direct GPU↔storage paths). The pitch for agentic serving: an agent session is *bursty* — seconds of activity, then minutes idle while a tool runs or a human reads. Idle sessions' blocks are exactly what LRU evicts, so without a second tier, every "quiet" session pays a full re-prefill when it wakes. LMCache turns that re-prefill into a memory copy.

### 6.1 Integration and configuration (verified against LMCache docs, August 2026)

LMCache plugs into vLLM's standard **KV connector** interface, in-process:

```bash
# lmcache_config.yaml
#   chunk_size: 256
#   local_cpu: true
#   max_local_cpu_size: 200        # GB of pinned host RAM

LMCACHE_CONFIG_FILE=lmcache_config.yaml \
vllm serve /models/org/model-NVFP4 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

Equivalent environment variables exist (`LMCACHE_LOCAL_CPU=True`, `LMCACHE_MAX_LOCAL_CPU_SIZE=200`, `LMCACHE_CHUNK_SIZE=256`). Operational notes that bite in practice:

- **Chunk size is 256 tokens by default** — LMCache indexes and transfers in 256-token chunks, not vLLM's 16-token blocks. Reuse granularity through LMCache is therefore coarser: a 250-token unique tail can't be stored until the chunk fills. Fine for agents (prefixes are thousands of tokens); worth knowing when interpreting partial-hit behavior.
- **Pinned memory and `RLIMIT_MEMLOCK`:** the CPU tier uses page-locked (pinned) RAM for fast DMA (direct memory access) transfers, and LMCache clamps its allocation to your memlock limit. In containers, set `--ulimit memlock=-1` (Compose: `ulimits: memlock: -1`) or the tier will be silently smaller than configured.
- **Eviction within LMCache is LRU** across its tiers; sizing rule of thumb: make the CPU tier a few multiples of your GPU KV pool (host RAM is cheap next to HBM — a B300 node with 2–4 TB of DDR can back the whole day's session working set).

### 6.2 What it buys, with numbers

- vLLM's own offloading study (Jan 2026 blog): loading KV from CPU instead of recomputing improves TTFT **2–22×** depending on prompt length, and throughput up to **9×** at high hit rates.
- LMCache's multi-turn agentic benchmark (May 2026; MiniMax-M2.5 230 GB FP8 MoE on 2× MI300X, replaying 739 real Claude-Code conversation traces, ~20k→115k tokens per session): under stress (32 concurrent users at 100k context), **average TTFT 3.0× lower** (34.6 s vs 102.2 s) and **2.3× more requests completed** vs HBM-only. Achieved hit rate 72.4% against a theoretical 96.9% — the gap being churn and capacity, which is the honest picture you should expect.
- The remote tier generalizes this across a fleet: the **Mooncake store** integration (vLLM blog, May 2026) pools KV across instances so a session can migrate replicas without losing its cache — hit rate 1.7% → 92.2%, and on replayed Codex traces (12× GB200) 3.8× throughput and 46× lower P50 TTFT. In an enclave, any such backend (Redis, Mooncake master/clients) is another internal standing service to mirror, run, and monitor — adopt only when single-node CPU/NVMe tiers stop being enough.

### 6.3 CacheGen and CacheBlend (verify before relying on them)

Two research-grade LMCache features, both present in current LMCache docs (mid-2026) but worth a fresh look at adoption time:

- **CacheGen** compresses KV into a compact bitstream for cheap storage/transmission — relevant if you ever ship KV across the fabric or persist sessions for days.
- **CacheBlend** relaxes the *prefix-only* rule for RAG (retrieval-augmented generation): it reuses KV for retrieved chunks appearing mid-prompt and recomputes only a small fraction of tokens to stitch attention correctly. Agents that inject variable document sets ahead of a stable suffix are the target workload. It is an approximation — gate it through your eval suite like any quality-affecting change.

---

## 7. vLLM-native KV offload (v0.26)

As of v0.26 (August 2026), vLLM has a built-in answer to the same problem, matured over the v0.11→v0.26 cycle: the **OffloadingConnector**. Default behavior without it is unchanged: full blocks live in HBM until LRU-evicted, and memory pressure causes recompute-preemption.

```bash
vllm serve /models/org/model-NVFP4 \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "CPUOffloadingSpec",
      "cpu_bytes_to_use": 200000000000
    }
  }'
```

Key facts (docs + release notes, verified August 2026):

- `cpu_bytes_to_use` is the host-RAM tier size, **total across all workers** (not per TP rank). Transfers use `cudaMemcpyAsync` DMA and are fully asynchronous — decode does not stall while blocks stream out.
- **Tiered mode** (`"spec_name": "TieringOffloadingSpec"`, new in v0.26) adds secondary tiers behind CPU: **`fs`** (a filesystem/NVMe directory with `root_dir` and read/write thread counts), **`obj`** (S3-compatible object store — in the enclave that means internal MinIO, never an external endpoint), and **`p2p`** (peer transfer over NIXL/UCX RDMA). Eviction policy per tier is `lru` or `arc`. v0.26 also added offloading metrics, split CPU-cache read/write gauges, and sync/async tiering-lookup-delay histograms.
- Per-request control exists via `"kv_transfer_params": {"max_offload_tokens": ...}`.
- A shorthand CLI (`--kv-offloading-backend {native,lmcache}` plus `--kv-offloading-size <GiB>`) appears in the vLLM blog and the RFC for a top-level offloading interface; the main docs still show the `--kv-transfer-config` form. Treat the shorthand as version-dependent and check `vllm serve --help` on your pinned image before writing runbooks against it.
- **`--swap-space` is dead weight in v1** — the old v0 CPU-swap preemption path; in the v1 engine `num_cpu_blocks` is hard-coded to zero and the flag does nothing (vLLM issue #27984). Do not "tune" it; offload is configured only through the connector above. Similarly, the old `vllm:cpu_cache_usage_perc` / `num_requests_swapped` metrics are dead.

**Native offload vs LMCache — the decision:** native keeps your dependency surface at exactly "vLLM" (a real virtue in an air gap), now covers CPU + NVMe + object store, and is where upstream investment is flowing; LMCache adds the remote shared-pool backends, CacheGen/CacheBlend, and a longer production track record. Our default: start with the native `CPUOffloadingSpec` on the long-context/B300 pool when §4's reuse-gap histograms show evictions beating reuse; reach for LMCache when you specifically need cross-instance sharing or its research features. Either way it's one flag change per profile, so the eval/perf gate can arbitrate.

---

## 8. KV cache quantization: `--kv-cache-dtype fp8`

Weights quantization (NVFP4) and KV quantization are separate switches; this one halves every number in §1's tables.

- Options: `fp8` (defaults to the e4m3 variant on our hardware), `fp8_e4m3`, `fp8_e5m2`, or `auto` (match model dtype). e4m3 keeps more mantissa (accuracy), e5m2 more range; use `fp8` and move on.
- By default all quantization scales are 1.0. For accuracy-sensitive models you can bake **calibrated per-tensor or per-attention-head scales** into the checkpoint with `llm-compressor` (per-head requires the Flash Attention backend). With Flash Attention 3 on Blackwell, attention itself runs in the FP8 domain, so you also gain kernel speed, not just capacity.
- `--kv-cache-dtype-skip-layers` can exempt sensitive layers (by index or by type, e.g. `sliding_window`) if evals show localized regression.
- Compatibility: composes with prefix caching (hashes are over token IDs, not stored bytes), with offload (blocks shrink, transfers halve), and with MLA (DeepSeek at FP8 KV is ~34 KB/token — the numbers multiply).

The SOP default stands: **FP8 KV cache on for agentic serving, always validated per model on the eval gate** — a small number of models regress measurably, and the eval gate is where that surfaces, not production. One subtlety for the long-context profile: KV quantization error compounds with very long contexts on some models, so include long-context retrieval probes (needle tests at your real max length) in the gate, not just short-context accuracy.

---

## 9. Long-context strategy for agents: the app layer is the cheap lever

### 9.1 `--max-model-len` is a budget, not a feature flag

`--max-model-len` caps tokens per request (prompt + output). It does **not** change the size of the KV pool — that is fixed by memory after weights — but it changes the worst case any single request can consume, and the scheduler's admission math. At startup vLLM prints the honest capacity line:

```
GPU KV cache size: 1,047,552 tokens
Maximum concurrency for 131,072 tokens per request: 7.99x
```

That second line is your capacity unit in the SOP's sense. Setting `--max-model-len 262144` "because the model supports it" invites a single request to eat 2× the memory and stall the pool (long-prefill interference, preemptions). Profiles pin it deliberately: `interactive-agent` at 128k on B300; anything longer goes to the `long-context` profile with conservative `--max-num-seqs` and offload enabled. Also keep **chunked prefill** defaults (tune `--max-num-batched-tokens`, 8k–32k on Blackwell) so long prefills stream through without spiking other sessions' inter-token latency.

### 9.2 Context management beats context capacity

Every token you *don't* keep in context is ~35–320 KB of HBM you didn't spend (per §1) and prefill you didn't run. The application-layer levers, in order of cheapness:

1. **Truncate tool outputs at the source.** Agents dump 40 KB of raw JSON into the transcript when 2 KB was informative. Cap and summarize tool results *before* appending; this is free capacity.
2. **Append-only transcript discipline.** Never rewrite earlier turns (re-rendering, re-ordering, "cleaning up") — that breaks the byte-stability that prefix caching needs.
3. **Compaction (summarization) at breakpoints.** When a session approaches its budget, replace old turns with a model-written summary. Critical caveat: compaction *rewrites the prefix*, so the next turn is a full cache miss and a one-time re-prefill of the compacted context. Do it at natural pauses (task boundaries), not mid-loop, and compact aggressively when you do (one big miss beats three).
4. **Fresh-context subagents.** Delegate exploratory work (searching, reading long docs) to a subagent whose context is discarded, returning only conclusions to the parent. This keeps the expensive long-lived context small — an architecture change, but the highest-leverage one for very long tasks.

The Responses-API migration (SOP §6) helps here too: once the gateway owns conversation state via `previous_response_id`, prompt rendering happens in *one* place, so byte-stability and compaction policy become platform guarantees instead of per-team folklore.

---

## 10. Capacity worksheet: sizing a pool for N agents of M tokens

A repeatable method — numbers first, then validation. Worked example inline: **target 60 concurrent agent sessions, typical context 60k tokens, p95 100k, Qwen3-235B-A22B NVFP4 on B200 nodes.**

1. **Per-token KV bytes.** From `config.json`: `2 × 94 × 4 × 128 × 1 (FP8) = 96,256 B ≈ 94 KB/token`. (MLA models: `layers × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes`.)
2. **Per-session KV.** Size on p95, not the cap: `100,000 × 96,256 B ≈ 9.6 GB`. Typical session: `60,000 × 96,256 ≈ 5.8 GB`.
3. **Node KV pool.** `HBM_total × gpu_memory_utilization − weights − runtime overhead`. For 8× B200: `1,536 × 0.90 − 125 − ~40 ≈ 1,217 GB`. Don't trust the estimate — read the startup log's `GPU KV cache size: N tokens` from a real boot (and note the logged `--kv-cache-memory` value that reproduces it).
4. **Concurrent-session capacity.** Pool ÷ per-session: `1,217 / 9.6 ≈ 126` sessions at p95 sizing, ~210 at typical sizing. Against a target of 60, one node carries it with ~2× headroom — good, because...
5. **Apply the working-set correction.** N *concurrent sessions* is not N *resident contexts*: idle-but-alive sessions hold blocks until evicted or offloaded. Decide the multiplier from your traces (sessions alive per session actively decoding — often 2–5× for human-in-the-loop agents). If `alive × per-session KV > pool`, configure offload (§6/§7) so idle sessions park in CPU/NVMe instead of dying; size the CPU tier to cover `(alive − resident) × per-session`.
6. **Add the prefix-sharing credit (conservatively).** Shared system prompt + tool schemas of, say, 8k tokens ≈ 0.77 GB, stored once per replica rather than per session — at 60 sessions that returns ~45 GB. Treat as headroom, not plan-of-record.
7. **Reserve for growth during decode.** Reasoning models can emit tens of thousands of thinking tokens; budget output growth into M (this is why the example sizes at p95 total context, not prompt length).
8. **Validate with `guidellm`** at the target concurrency and context distribution; record "max concurrency at SLO" in the recipe file per SOP §7. The math above tells you which node count to *try*; the load test is what you *commit* to.

Sanity crosscheck from the vendor tables in §1.4: if the answer implies more than ~80% sustained `kv_cache_usage_perc`, buy headroom (another node, B300 instead of B200, FP8 KV if not already on, or context-management work from §9.2 — usually in the opposite order of cost).

---

## Common pitfalls checklist

- **A timestamp, request ID, or random element in the first prompt block.** Symptom: hit rate ≈ 0 despite identical-looking prompts. Fix: byte-stable prefix; dynamic data goes late in the prompt or into tool results.
- **Round-robin load balancing across replicas.** Symptom: hit rate ≈ single replica's share. Fix: consistent-hash affinity (Compose tier) or llm-d prefix-cache scoring (K8s tier).
- **Tool schemas serialized from unordered maps.** Symptom: hit rate flaps between deploys of the *client*. Fix: canonical JSON serialization in the gateway/SDK.
- **`cache_salt` sprayed per-request or per-user inside one trusted platform.** Symptom: system-prompt sharing gone. Fix: salt at the tenant boundary only.
- **Tuning `--swap-space`.** It does nothing in the v1 engine; offload is configured via the connector (`--kv-transfer-config` / offloading flags).
- **Assuming weight quantization shrank the KV cache.** It didn't; only `--kv-cache-dtype` does.
- **Setting `--max-model-len` to the model's maximum "just in case."** Invites worst-case admission math, long-prefill interference, and preemption storms. Pin per profile.
- **Forgetting `--ulimit memlock=-1`** on containers running LMCache/pinned-CPU tiers. Symptom: CPU tier silently smaller than configured.
- **Judging offload need without the histograms.** If `kv_block_reuse_gap_seconds` exceeds `kv_block_idle_before_evict_seconds`, you are recomputing what you evicted — that's the offload signal.
- **Re-encoded images in multimodal prompts.** Identical pixels, different bytes → different multimodal hash → cache miss. Pass the same image object/bytes every turn.
- **Compacting context mid-loop.** Every compaction is a deliberate full cache miss; schedule it at task boundaries.
- **Not re-checking hit rate after engine upgrades.** Chat templates and hash-algorithm defaults (`--prefix-caching-hash-algo`, builtin vs sha256) can change between versions; a silent rendered-prompt diff zeroes reuse.

---

## Study questions

1. **Compute the per-token KV cache size, in bytes, for a model with 48 layers, 8 KV heads, head_dim 128, at BF16. How does it change with FP8 KV?**
   Answer: `2 × 48 × 8 × 128 × 2 = 196,608 B ≈ 192 KB` per token at BF16; FP8 halves it to ~96 KB.

2. **Why does DeepSeek-R1 (671B parameters) use less KV memory per token than Llama-3.3-70B?**
   Answer: MLA caches one compressed 576-element latent per layer (61 layers → ~69 KB/token BF16) instead of per-head K and V; Llama's GQA still stores 8 full K and V heads across 80 layers (~320 KB/token).

3. **What exactly goes into a vLLM prefix-cache block hash, and why does the parent hash matter?**
   Answer: hash(parent block hash, the block's token IDs, extra keys: LoRA ID, multimodal content hashes, cache_salt). The parent hash chains in the entire prefix, because attention makes a block's KV depend on everything before it — so any upstream byte change correctly invalidates all later blocks.

4. **Your agent pool's hit rate dropped from 85% to 12% after a "harmless" prompt cleanup. What happened and how do you confirm it?**
   Answer: something changed the early prompt bytes (timestamp, reordered tool schema, template change). Confirm by diffing rendered prompts byte-for-byte between two turns and watching `vllm:prefix_cache_hits/queries` per replica; low hit rate with low KV usage points at content/routing, not eviction.

5. **Why is sampling temperature absent from the block hash?**
   Answer: sampling parameters affect token *selection*, not the K/V values computed from tokens; requests differing only in sampling can safely share cached blocks, so including them would only reduce reuse.

6. **When does session-affinity routing stop being good enough, and what does llm-d's precise mode add?**
   Answer: affinity balances sessions, not load, and goes blind when caches are evicted or replicas restart. llm-d's EPP scores every request across pods (prefix-cache scorer + queue scorer), and precise mode feeds the scorer real-time KV events over ZMQ, so it *knows* which blocks are on which pod — the llm-d benchmark measured P90 TTFT of 0.54 s vs 31 s (approximate) vs 92.5 s (random) under load.

7. **A replica shows `kv_cache_usage_perc ≈ 1.0`, frequent preemptions, and reuse-gap longer than idle-before-evict. What is the cheapest sequence of fixes?**
   Answer: first `--kv-cache-dtype fp8` if not already on (doubles capacity), then app-layer context reduction (tool-output truncation, compaction), then KV offload to CPU/NVMe (OffloadingConnector or LMCache) so idle sessions park instead of dying, then more/bigger HBM (B300) or another node.

8. **What does `--swap-space` do in vLLM v0.26?**
   Answer: nothing — it is the legacy v0 CPU-swap knob; the v1 engine hard-codes CPU blocks to zero and preempts by recompute. Offload is configured through the KV connector instead.

9. **How many concurrent 128k-token sessions of Llama-3.3-70B (NVFP4 weights, FP8 KV) fit on one B300, roughly?**
   Answer: ~210 GB KV pool after weights/overhead at 0.90 utilization; 128k × 160 KB ≈ 21.5 GB per session → about 10 sessions (vs ~6 on a B200).

10. **Why can compaction (summarizing old turns) temporarily *hurt* latency even though it saves memory?**
    Answer: it rewrites the prompt prefix, so the next turn misses the entire prefix cache and pays one full re-prefill of the compacted context; schedule compaction at task boundaries and compact aggressively so the miss is rare.

11. **What is `cache_salt` for, and what is the failure mode of overusing it?**
    Answer: it partitions the prefix cache so only same-salt requests share blocks, closing timing side-channels between distrusting tenants. Salting per-user/per-request inside one trusted platform destroys legitimate sharing (e.g., of the common system prompt) and silently tanks hit rate.

12. **Your capacity worksheet says one node suffices, but the guidellm run misses TTFT SLO at target concurrency. Name two effects the static math ignores.**
    Answer: working-set inflation (idle-but-alive sessions holding or thrashing blocks — the concurrency number understates resident contexts) and long-prefill interference (big prefills stealing batch capacity from decode, which shows up as TTFT/ITL spikes rather than memory exhaustion; mitigated by chunked-prefill tuning or P/D disaggregation).

---

## Sources

Primary documentation and design docs:

- vLLM automatic prefix caching design (block hash, extra keys, LRU/free-queue eviction): https://docs.vllm.ai/en/stable/design/prefix_caching/ (also `docs/design/prefix_caching.md` in the vLLM repo)
- vLLM metrics design (`prefix_cache_queries/hits`, `kv_cache_usage_perc`, KV block histograms, deprecated swap metrics): https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md
- vLLM KV offloading usage guide (OffloadingConnector, `cpu_bytes_to_use`, `TieringOffloadingSpec`, fs/obj/p2p tiers, lru/arc): https://docs.vllm.ai/en/latest/features/kv_offloading_usage/
- vLLM quantized KV cache (`--kv-cache-dtype`, scales, FA3, `--kv-cache-dtype-skip-layers`): https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- vLLM KV events config (`--kv-events-config`, ZMQ publisher): https://docs.vllm.ai/en/stable/api/vllm/config/kv_events/
- vLLM v0.26.0 release notes (tiered offloading, object-store tier, offloading metrics, per-group attention backends): https://github.com/vllm-project/vllm/releases/tag/v0.26.0
- Cache salting PR/RFC (timing side channels, per-request `cache_salt`): https://github.com/vllm-project/vllm/pull/17045 and https://github.com/vllm-project/vllm/issues/16016
- `swap_space` unused in v1 (issue): https://github.com/vllm-project/vllm/issues/27984
- Top-level KV-offloading CLI RFC: https://github.com/vllm-project/vllm/issues/26858

Blogs and benchmarks (dates as published):

- vLLM blog, "Inside vLLM's New KV Offloading Connector" (Jan 2026; TTFT 2–22×, throughput up to 9×): https://vllm.ai/blog/2026-01-08-kv-offloading-connector
- vLLM blog, "Serving Agentic Workloads at Scale with vLLM x Mooncake" (May 2026; 1.7%→92.2% hit rate, Codex-trace results): https://vllm.ai/blog/2026-05-06-mooncake-store
- vLLM blog, vLLM Router release (Dec 2025; consistent hashing, bare-metal): https://vllm.ai/blog/2025-12-13-vllm-router-release
- llm-d, "KV-Cache Wins You Can See" (precise vs approximate vs random scheduling numbers): https://llm-d.ai/blog/kvcache-wins-you-can-see
- llm-d prefix-cache-aware routing architecture (EPP, KV-cache indexer, ZMQ KV events): https://llm-d.ai/docs/architecture/advanced/kv-management/prefix-cache-aware-routing
- Red Hat Developer, "Master KV cache aware routing with llm-d" (Oct 2025): https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference
- LMCache architecture docs (tiers, connector, chunk size, CacheBlend/CacheGen): https://docs.lmcache.ai/developer_guide/architecture.html
- LMCache CPU-offload quickstart (LMCacheConnectorV1 config): https://docs.lmcache.ai/getting_started/quickstart/offload_kv_cache.html
- LMCache blog, multi-turn agentic benchmark on MI300X (May 2026): https://blog.lmcache.ai/en/2026/05/12/benchmarking-lmcache-for-multi-turn-agentic-workloads-on-amd-mi300x/
- TraceLab, coding-agent workload characterization (prefix hit rates ~95.7% token-weighted): https://arxiv.org/pdf/2606.30560

Model configurations used for the worked math:

- Llama-3.3-70B config (80 layers, 8 KV heads, head_dim 128): https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/blob/main/config.json
- Qwen3-235B-A22B config (94 layers, 4 KV heads, head_dim 128): https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
- DeepSeek-V3 config (61 layers, kv_lora_rank 512, qk_rope_head_dim 64): https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
- DeepSeek-V2 paper (MLA design): https://arxiv.org/pdf/2405.04434

Hardware:

- NVIDIA B300 / Blackwell Ultra 288 GB HBM3e specifications: https://www.theregister.com/2025/03/18/nvidia_blackwell_ultra/ and https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
