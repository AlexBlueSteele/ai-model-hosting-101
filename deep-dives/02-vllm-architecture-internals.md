# vLLM Internals: How the Engine Actually Works

**Deep-dive 02 in the on-prem inference series.** Companion to the [deployment SOP](../SOP-production-ai-model-deployment.md) and [PRIMER](../PRIMER.md). Written August 2026 against vLLM v0.26/v0.27 (current stable line at time of writing). vLLM ships a minor release roughly every two weeks, so treat every "as of" marker in this document seriously and re-verify against release notes when you upgrade.

**Who this is for:** an engineer who can run `vllm serve` from a recipe but wants to understand what is actually happening inside the process — so that log lines, flags, and failure modes stop being magic. Everything here is grounded in the primary sources listed at the end.

---

## Key takeaways

- vLLM's founding insight (the PagedAttention paper, SOSP 2023) is that pre-vLLM serving systems wasted **60–80% of GPU memory** on KV-cache fragmentation; paging the cache in small fixed-size blocks, like an operating system pages RAM, recovers almost all of it and directly buys 2–4× throughput.
- Since v0.8.0 (early 2025) vLLM runs only the **V1 engine**: an HTTP frontend process (tokenization, chat templates, streaming) and an **EngineCore** process (scheduler, KV cache manager, GPU execution) that talk over ZeroMQ. V0 was fully removed by v0.11.0 (October 2025).
- The V1 scheduler does **not** distinguish prefill from decode. Every step it hands out a token budget (`--max-num-batched-tokens`) across running requests: all pending decodes first, then chunks of pending prefills. That single mechanism *is* continuous batching plus chunked prefill.
- **Prefix caching** is content-addressed block reuse: each 16-token block gets a hash chained from its parent block. Byte-identical prefixes hit; one changed byte upstream misses everything after it. This is why the SOP bans timestamps in system prompts.
- **Async scheduling** (default since v0.14.0, February 2026) overlaps CPU scheduling for step N+1 with GPU execution of step N, removing the per-step CPU bubble that otherwise caps decode throughput.
- Attention runs through pluggable **backends**: FlashAttention-4 is the default on Blackwell (SM100), FlashInfer wraps NVIDIA TensorRT-LLM kernels, and a portable Triton backend is the always-available fallback. The old `VLLM_ATTENTION_BACKEND` environment variable is deprecated in favor of `--attention-config.backend`.
- vLLM compiles the model with **torch.compile** and captures **CUDA graphs** (piecewise by default) to kill kernel-launch overhead. Compile artifacts cache under `~/.cache/vllm/torch_compile_cache` — ship that directory in your air-gap bundle to avoid a multi-minute cold start on every boot.
- `--gpu-memory-utilization` is measured **once at startup** by a profiling forward pass, applies **per vLLM instance** (not per GPU globally), and everything left over becomes KV-cache blocks — the "GPU KV cache size: N tokens" log line is your real capacity number.
- Unknown architectures fall back to the **Transformers modeling backend** (`--model-impl transformers`). It was historically 2–4× slower; as of July 2026 runtime kernel-fusion work has brought many standard architectures to near-native speed, but verify per model — it is still not the production default.
- The **Rust frontend** (`VLLM_USE_RUST_FRONTEND=1`, merged mid-2026) replaces the Python HTTP frontend with one Rust process (~5× more frontend throughput in vLLM's own benchmark) but is **experimental as of August 2026** — missing TLS, API-key auth, LoRA management, and several endpoints. Not yet for production.

---

## 1. Origin story: the memory problem vLLM was built to solve

### 1.1 Two phases, two bottlenecks

Recall from the PRIMER: serving a request has a **prefill** phase (read the whole prompt in one parallel pass — compute-bound) and a **decode** phase (emit one token at a time — memory-bandwidth-bound). During both phases the model maintains a **KV cache** (key-value cache): per-token intermediate tensors that let the model attend to earlier tokens without recomputing them. The KV cache is the working memory of a conversation, it lives in GPU HBM (high-bandwidth memory), and for long agentic sessions it is *larger than most people's intuition* — we will put numbers on it in §8.

### 1.2 What serving looked like before vLLM

Circa 2022–2023, serving systems (FasterTransformer and friends) allocated each request's KV cache as **one contiguous slab sized for the maximum possible sequence length**. Two kinds of waste followed:

- **Internal fragmentation:** a request that might grow to 2,048 tokens got a 2,048-token slab up front, even if it finished at 80 tokens. The unused tail was dead memory.
- **External fragmentation:** variable-sized contiguous slabs left unusable gaps between allocations, exactly like `malloc` without a good allocator.

The vLLM team at UC Berkeley measured the damage: existing systems wasted **60–80% of KV-cache memory**, meaning only 20–40% of the memory nominally spent on "remembering conversations" held live tokens. Since batch size (and therefore throughput) is capped by how many sequences' caches fit in HBM, this waste directly throttled the GPU.

There was a second, related problem: **static batching**. Early servers batched N requests, ran the *whole* batch to completion, then admitted the next batch. One long-winded response held the entire batch hostage while finished requests' slots sat idle. The Orca system (OSDI 2022) introduced **iteration-level scheduling** — re-form the batch at *every* generation step, retiring finished sequences and admitting new ones immediately. This is what the industry now calls **continuous batching**, and vLLM adopted it from day one.

### 1.3 The PagedAttention insight

The paper — Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 (Best Paper) — asked: operating systems solved fragmentation fifty years ago with virtual memory and paging; why not do the same for the KV cache?

- Chop every sequence's KV cache into fixed-size **blocks** (pages) of a few tokens each.
- Keep a per-sequence **block table** (page table) mapping logical block numbers to physical block locations, which need not be contiguous.
- Write a custom attention kernel that follows the block table during the attention computation.

Result: internal waste shrinks to at most one partially-filled block per sequence (under 4% in the paper's measurements), external fragmentation disappears entirely because all blocks are the same size, and — the bonus that matters most for agents — **blocks become shareable between requests**, which later became automatic prefix caching. The paper reported **2–4× throughput** over the prior state of the art at equal latency, with the biggest wins on long sequences and large batches. That result is the reason vLLM exists as a project.

> **Mental model to keep:** vLLM is an operating system for GPU memory. The scheduler is its CPU scheduler, the KV cache manager is its virtual-memory subsystem, blocks are pages, block tables are page tables, and preemption is swapping. Nearly every internal concept maps onto an OS concept you already know.

---

## 2. The V1 engine: what processes exist and what each does

### 2.1 A brief history: V0 → V1

vLLM's original engine ("V0") accreted features for two years until its Python-heavy inner loop and tangled scheduler became the bottleneck. The team rewrote the core as **V1** (announced January 27, 2025 — a 1.7× speedup from architecture alone, before any new kernels). The timeline that matters to you:

| Version | Date | V0/V1 status |
|---|---|---|
| v0.8.0 | early 2025 | V1 becomes the default engine |
| v0.9.x | mid 2025 | V0 frozen (last supported) |
| v0.10.x | 2025 | V0 code removal begins |
| v0.11.0 | Oct 2025 | V0 fully removed |
| v0.26/v0.27 | Jul–Aug 2026 | V1 only; current stable line |

Anything you read that mentions `VLLM_USE_V1`, "V0 scheduler", `SequenceGroup`, or swap-based preemption describes a dead codebase. Check the date on any vLLM blog post before trusting it.

### 2.2 The process split

A modern `vllm serve` is **at least two operating-system processes**:

1. **API server (frontend) process.** Runs the FastAPI application with the OpenAI-compatible routes. Owns everything that is CPU work on the request path: authentication, request validation, applying the chat template, tokenization, multimodal preprocessing (image decoding etc.), and — on the way out — detokenization and SSE (Server-Sent Events) streaming. In vLLM code this layer is `AsyncLLM` plus its `Processor` (input side) and `OutputProcessor` (output side).
2. **EngineCore process.** The GPU-facing inner loop (`vllm/v1/engine/core.py`). It owns the **scheduler**, the **KV cache manager**, and the **executor** that drives one worker per GPU (so with `--tensor-parallel-size 8` there are 8 worker processes under the EngineCore). At startup it profiles GPU memory (§8), initializes the KV block pool, captures CUDA graphs (§7), then runs a busy loop calling `step()` forever.

The two sides communicate over **ZeroMQ (ZMQ)** sockets — a lightweight message-passing library — using msgpack serialization (`MsgpackEncoder`/`MsgpackDecoder`). The wrapper that exposes an EngineCore over ZMQ is `EngineCoreProc`; the frontend talks to it through an `EngineCoreClient` (async multiprocess client in the server case; there is also an in-process client used by the offline `LLM` class so that `python -c "from vllm import LLM"` batch jobs skip the socket entirely).

**Why split processes at all?** Three reasons. First, isolation: Python's GIL (global interpreter lock) means tokenizing a 100 KB prompt in the same process as the GPU loop would stall token generation for everyone. Second, scaling: the frontend and the engine can be scaled independently — recent vLLM versions can run multiple API server processes (`--api-server-count N`) against one engine, and conversely one frontend can load-balance across multiple EngineCores in data-parallel deployments. Third, replaceability: because the boundary is a ZMQ protocol, the entire Python frontend can be swapped for the Rust frontend (§10) without touching the engine.

### 2.3 Request lifecycle, end to end

```
            CLIENT                    API SERVER PROCESS                 ENGINECORE PROCESS
              |                              |                                  |
  POST /v1/chat/completions                  |                                  |
  {model, messages, tools,                   |                                  |
   stream:true}                              |                                  |
              |------------------------------>                                  |
              |                        auth / validate                          |
              |                        resolve served-model-name                |
              |                        apply chat template (Jinja2)             |
              |                        tokenize -> [token IDs]                  |
              |                        mm preprocess (images -> tensors)        |
              |                              |                                  |
              |                        AsyncLLM.add_request()                   |
              |                              |---- ZMQ (msgpack) --------------->
              |                              |                            WAITING queue
              |                              |                                  |
              |                              |                     +--------- step() loop ------+
              |                              |                     | SCHEDULER:                 |
              |                              |                     |  decodes first, then       |
              |                              |                     |  prefill chunks, up to     |
              |                              |                     |  max-num-batched-tokens    |
              |                              |                     | KV MANAGER:                |
              |                              |                     |  check prefix-cache hits,  |
              |                              |                     |  allocate blocks, update   |
              |                              |                     |  block tables              |
              |                              |                     | EXECUTOR -> GPU workers:   |
              |                              |                     |  one fused forward pass    |
              |                              |                     |  for the whole batch       |
              |                              |                     |  (CUDA graphs / compiled)  |
              |                              |                     | SAMPLER: next token IDs    |
              |                              |                     +----------------------------+
              |                              |<---- ZMQ: EngineCoreOutputs -----|
              |                        OutputProcessor:                         |
              |                        incremental detokenize,                  |
              |                        run tool-call / reasoning                |
              |                        parsers on the text stream               |
              |                              |                                  |
              |<--- SSE: data: {delta:...}   |         (loop: one iteration     |
              |<--- SSE: data: {delta:...}   |          ~= one token per        |
              |<--- SSE: data: [DONE]        |          running request)        |
```

Two things to internalize from this diagram. First, **the engine thinks in token IDs, never text** — templates, tokenization, and detokenization all live in the frontend. If a chat template is wrong, the engine happily serves garbage; nothing GPU-side will error. Second, **one scheduler iteration produces at most one new token for each running decode request**, so your inter-token latency (ITL) is essentially the wall-clock time of one `step()` — everything in §4–§7 is about making that step fast and keeping it full.

---

## 3. PagedAttention mechanics: blocks, tables, and sharing

### 3.1 Blocks and block tables

The KV cache manager carves the KV region of HBM into a single pool of fixed-size **physical blocks**. Each block holds the keys and values for a fixed number of token positions — **16 tokens by default** on CUDA platforms (`--block-size`; some kernels prefer other sizes, and since v0.26 heterogeneous KV-cache groups can even mix layouts). Every request owns a **block table**: an ordered list of physical block IDs such that logical block 0 covers tokens 0–15, logical block 1 covers tokens 16–31, and so on. The attention kernels take block tables as an input and gather K/V through them, which is why arbitrary physical scattering costs almost nothing.

A worked example: a request with a 5,000-token prompt needs ceil(5000/16) = 313 blocks for prefill; the only waste is the unused tail of block 313 (8 token slots). As decode proceeds, the manager appends one fresh block every 16 generated tokens. When the request finishes, its blocks return to the free pool (or linger as reusable cache — next section).

### 3.2 Prefix caching: content-addressed blocks

Automatic prefix caching (on by default; disable with `--no-enable-prefix-caching`) turns the block pool into a **content-addressed cache**. Every *full* block gets a hash computed from: the parent block's hash, the block's 16 token IDs, and "extra keys" when relevant — the LoRA adapter ID, hashes of any multimodal inputs covered by the block, and an optional per-request `cache_salt` (send a salt when you must *prevent* cross-tenant sharing). Since v0.11 the default hash function is SHA-256, chosen to make engineered collisions (one tenant tricking the cache into serving another tenant's KV) computationally infeasible.

The chained-parent construction is the key property: a block's hash commits to the *entire prefix before it*, not just its own 16 tokens. When a new request arrives, the manager hashes its prompt block-by-block and walks the global hash table; every consecutive hit means those tokens **skip prefill entirely**. The first miss ends the matched prefix — and everything after it recomputes, because the chain is broken.

Freed blocks are not zeroed; they go to an LRU (least-recently-used) free list *keeping their hashes*, so a "free" block is really "evictable cache". Eviction happens lazily when the allocator needs blocks. This is why one design doc's phrase matters: there is no tree structure to maintain — blocks are independent, refcounted, and the hash table is the only index.

> **Common pitfalls — prefix caching.**
> 1. A timestamp, request ID, or user name early in the system prompt changes token IDs in block 0, which changes *every* downstream hash. Cache hit rate goes to ~0 and nobody gets an error — you just see TTFT (time to first token) quietly triple. Watch the `vllm:prefix_cache_hits`-family metrics (the SOP treats a hit-rate drop as an incident).
> 2. Hits happen at block granularity. A 10-token common prefix can never hit (it never fills block 0). Put long, stable content (system prompt, tool schemas) first and variable content last.
> 3. The cache lives in **one replica's HBM**. Round-robin load balancing across replicas destroys it; use session-affinity or prefix-aware routing (llm-d does this natively).

### 3.3 Copy-on-write, then and now

The 2023 paper describes **copy-on-write (CoW)** for blocks shared between sequences that then diverge — parallel sampling (`n>1`) and beam search shared prompt blocks with refcount > 1, and the first write to a shared block triggered a private copy, exactly like `fork()` semantics in an OS. That is the mechanism to cite when someone asks "what does copy-on-write mean in PagedAttention?".

In the V1 engine, the same *outcome* is achieved more simply: full blocks are effectively immutable and shared read-only via refcounts through the prefix cache, and each sequence always writes new tokens into its own private tail block. Parallel sampling is implemented as sibling requests whose shared prompt naturally deduplicates through prefix caching, so the explicit CoW machinery of V0 is gone. (Old beam-search-era docs describing CoW block copies describe V0 behavior.)

---

## 4. Continuous batching and chunked prefill: how a step is formed

### 4.1 The unified scheduler

The V0 scheduler had separate code paths and queues for prefill and decode. The V1 scheduler abolishes the distinction: its scheduling decision each step is literally a dictionary `{request_id: num_tokens_to_process_this_step}`, and it tracks per-request progress as `num_computed_tokens` out of `num_tokens`. A decode request contributes 1 token to the step (or 1 + k with speculative decoding); a prefill request can contribute anywhere from 1 token up to its whole remaining prompt. Prefill, chunked prefill, decode, prefix-cache-skips, and speculation all fall out of one representation.

Each step the scheduler spends a **token budget**:

- `--max-num-batched-tokens` — the maximum tokens processed in one forward pass (the budget). Historically defaulted around 2,048–8,192 depending on version and mode; the SOP recommends 8k–32k on Blackwell-class hardware.
- `--max-num-seqs` — the maximum number of *sequences* in one step. Default 1,024 in V1 (up from 256 in V0).

The spending order is **decodes first**: every running request that needs its next token gets 1 token of budget. Whatever budget remains goes to waiting/in-progress prefills — and a prefill larger than the leftover budget is **chunked**, processing (say) 8,000 tokens of a 60,000-token prompt this step and continuing next step. The default queueing policy is FCFS (first-come, first-served); `--scheduling-policy priority` switches to caller-assigned priorities with FCFS tie-breaking.

### 4.2 Why decode-first plus chunking is the whole ballgame for agents

Prefill is compute-bound; decode is bandwidth-bound and leaves compute idle. Mixing them in one batch is close to free real throughput — the prefill chunk soaks up the idle compute *between* the decode tokens' memory stalls. Decode-first ordering plus chunking bounds the damage a monster prompt can do: without chunking, a 200k-token agent-context prefill would occupy the GPU for whole seconds while every other session's ITL spikes; with chunking, that prefill is spread across many steps and each step still serves everyone's next token.

The tuning tradeoff is one knob: a **larger** `--max-num-batched-tokens` finishes prefills in fewer steps (better TTFT, better throughput) but makes each mixed step longer (worse ITL for concurrent decoders). A **smaller** budget smooths ITL at the cost of TTFT. This is exactly the SOP's `interactive-agent` vs `throughput-batch` profile distinction, expressed in scheduler terms. When chronic long-prefill interference persists even after tuning, the structural fix is prefill/decode disaggregation (separate GPU pools, KV shipped over NIXL) — see the SOP §2.5; that is a topology change, not an engine flag.

### 4.3 Preemption

If the block pool runs dry mid-flight (too many long sequences at once), the scheduler **preempts**: the most-recently-scheduled victims release all their blocks and drop back to the waiting queue, to be *recomputed* from scratch later (V1 preemption is recompute-based; recomputation is cheaper than it sounds because prefix caching often restores much of the work). You will see a warning like `...is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space` in the logs.

> **Common pitfall — preemption is a capacity alarm, not noise.** Occasional preemptions under a burst are fine. *Sustained* preemption means your (model length × concurrency) demand exceeds KV capacity: every preempted request burns prefill compute twice and its TTFT includes a requeue. Fixes in order of cheapness: `--kv-cache-dtype fp8` (≈2× KV capacity), lower `--max-num-seqs`, lower `--max-model-len` to what agents actually use, more HBM (B300), or KV offload (LMCache).

---

## 5. Async scheduling: overlapping CPU and GPU

In the naive loop, each iteration is: schedule on CPU → launch GPU work → wait for GPU → process outputs on CPU → repeat. The CPU segments (building the next batch's metadata, sampling bookkeeping, allocating blocks) put a **bubble** between consecutive GPU bursts. At Blackwell speeds a forward step for a modest model can be a few milliseconds, so even 1–2 ms of CPU work per step is a 20–50% throughput tax.

**Async scheduling** makes the scheduler work one step ahead: while the GPU executes step N, the CPU schedules step N+1, betting that each running decode will produce exactly one token (it does not yet know *which* token, and does not need to — block allocation and batch shape don't depend on token identity). When the bet is wrong for a request (it emitted a stop token or hit a sampling constraint), the engine reconciles by discarding that request's pre-scheduled slot. The GPU therefore sees back-to-back steps with no CPU gap.

Status as of August 2026: **on by default since v0.14.0 (February 2026)**; disable with `--async-scheduling=False` if you need to rule it out while debugging. At introduction it was incompatible with some features (structured outputs, speculative decoding, pipeline parallelism); those gaps have been progressively closed through 2026 (structured-output compatibility landed upstream), but the compatibility matrix moves per release — check your version's release notes before assuming a combination works. The SOP's stance stands: leave it on.

---

## 6. Attention backends: who actually computes attention

### 6.1 The cast

The attention operation itself is delegated to a pluggable **backend** — a kernel implementation registered against the engine's paged-KV layout. The ones that matter on NVIDIA hardware in mid-2026:

| Backend | What it is | Where it wins |
|---|---|---|
| FlashAttention (FA2/FA3/FA4) | Tri Dao's fused exact-attention kernels; vLLM ships them via `vllm-flash-attn` | Default choice; FA3 on Hopper, FA4 on Blackwell |
| FlashInfer | Kernel library that also wraps NVIDIA TensorRT-LLM attention kernels for SM100 | Blackwell decode, FP8/FP4 paths, MoE-adjacent kernels |
| Triton unified attention | Backend written entirely in Triton, native to vLLM | Portable fallback (NVIDIA/AMD/Intel); ~parity with FA3 on H100 long-decode per vLLM's March 2026 blog |
| FlashMLA / TRT-LLM MLA | Kernels for MLA (Multi-head Latent Attention, DeepSeek-style compressed KV) | DeepSeek-class models |
| TORCH_SDPA / XFORMERS | PyTorch's built-in scaled-dot-product attention / legacy | Last-resort compatibility |

**FlashAttention-4** deserves a note because it is the reason your B200s got faster in spring 2026: published March 2026 (Dao et al.), a ground-up redesign for Blackwell's SM100 architecture written in CuTe-DSL (CUTLASS' Python kernel DSL), reaching ~1,600 TFLOP/s (~71% utilization) on B200 — roughly 1.5–2× FA3's attention throughput. vLLM auto-selects FA4 on Blackwell since ~v0.17.

### 6.2 How vLLM picks, and how you override

At startup the engine picks a backend by walking an ordered candidate list filtered by: GPU compute capability (SM90 = Hopper, SM100 = Blackwell), model attention type (standard MHA/GQA vs MLA vs sliding-window — since v0.26 sliding-window support is an explicit backend capability, and different KV-cache groups in one hybrid model can even use *different* backends), KV dtype (FP8 KV rules out some kernels), and whether the backend's package imports cleanly. The Triton backend is always present, so selection can never leave you with nothing — just with something slower. The chosen backend is logged at startup (look for a line naming the backend per KV-cache group); **always check it after an engine upgrade**, because a silent fallback from FA4 to Triton is a ~20–40% perf regression that no error message will announce.

Overriding, as of v0.26:

```bash
# Modern form (AttentionConfig):
vllm serve /models/... --attention-config.backend FLASHINFER
# or JSON:                --attention-config '{"backend": "FLASH_ATTN"}'
# Pin the FlashAttention generation if needed:
vllm serve /models/... --attention-config.flash_attn_version 3
```

The historical `VLLM_ATTENTION_BACKEND=FLASHINFER` environment variable — which you will see in almost every pre-2026 tutorial — is **deprecated** (removal targeted for v0.14/v1.0-era cleanup; on current versions it warns or errors). Recognized backend names include `FLASH_ATTN`, `FLASHINFER`, `TRITON_ATTN`, `FLASHMLA`, `TORCH_SDPA`, and platform-specific ROCm/XPU variants.

> **Common pitfall — recipes encode backend choices.** Recipe YAMLs in `vllm-project/recipes` frequently pin a backend and companion env vars (e.g. `VLLM_USE_FLASHINFER_MOE_FP4=1` for the FlashInfer FP4 MoE path on NVFP4 checkpoints, per the SOP default). If you copy the serve command but not the environment block, you can end up NVFP4-on-a-slow-path and wonder why the recipe's throughput numbers don't reproduce.

---

## 7. torch.compile and CUDA graphs: making the step cheap

### 7.1 Why compilation exists

A transformer forward pass is thousands of small GPU kernels. Two overheads follow: Python/PyTorch *dispatch* cost per operation, and CUDA *kernel launch* cost per kernel (microseconds each — irrelevant for big prefills, brutal for decode steps that only compute one token per sequence). vLLM attacks both:

- **torch.compile** traces the model into an FX graph, applies vLLM's custom fusion passes (e.g. fusing RMSNorm with quantization ops), and generates optimized Triton kernels — removing dispatch overhead and specializing kernels to your exact shapes/dtypes/parallelism.
- **CUDA graphs** record the whole sequence of kernel launches for a given batch shape once, then replay it with a single launch call — removing launch overhead.

### 7.2 Piecewise vs full graphs

Attention kernels historically did not sit well inside CUDA graph capture (dynamic shapes, paged-KV pointer chasing). vLLM's solution is **piecewise capture**: `torch.compile` splits the model graph *at every attention op*; the in-between segments (MLPs, norms, projections — the launch-heavy majority) are captured as CUDA graphs, while attention runs eagerly between them. This works with **every** attention backend, which is why it is the safe default. Where the chosen backend does support graph-safe attention, **full** graphs capture the entire step and squeeze out the last overhead — most valuable for small models and pure-decode batches where launch cost dominates.

The single knob is `cudagraph_mode` in the compilation config:

| Mode | Meaning |
|---|---|
| `NONE` | No CUDA graphs (debugging) |
| `PIECEWISE` | Graphs everywhere except attention (long-time default; maximally compatible) |
| `FULL` | One graph including attention (uniform batches only) |
| `FULL_DECODE_ONLY` | Full graphs for pure-decode steps, eager otherwise (nice with P/D disaggregation on the decode pool) |
| `FULL_AND_PIECEWISE` | Full graphs for decode-shaped batches, piecewise for mixed/prefill — best of both; a runtime dispatcher picks per batch |

As of the v0.26-era codebase, a CUDA-graph **dispatcher** selects the right captured graph (or eager fall-through) per batch automatically, and `FULL_AND_PIECEWISE` is the recommended mode where the backend supports it (the docs describe `PIECEWISE` as the "past default"; verify the effective default for your pinned version at startup — it is logged). Configuration:

```bash
# Everything rides on --compilation-config (alias -cc), or the -O shorthand levels:
vllm serve /models/... -O3                                   # full optimizations
vllm serve /models/... --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
vllm serve /models/... -cc.cudagraph_mode FULL_DECODE_ONLY   # dot-notation form
```

The SOP's guidance maps directly: default piecewise behavior is fine; reach for full-graph modes on small models where launch overhead dominates.

### 7.3 Cold start, and why air-gapped ops must care

Compilation is not free: on first boot vLLM traces, compiles, and captures graphs for a ladder of batch sizes — commonly tens of seconds to minutes of extra startup on top of weight loading. Artifacts (FX graphs, generated Triton kernels) are cached in **`~/.cache/vllm/torch_compile_cache`**, keyed by a hash of everything relevant (model, vLLM version, config, parallelism). A published measurement (Route179, June 2026): warm cache cut the torch.compile phase from ~39 s to ~10 s and total cold start from ~221 s to ~167 s for their setup. Two operational corollaries:

- **Ship the compile cache.** The cache directory is copyable. In the enclave, mount a persistent volume at the cache path (or pre-bake the cache in staging on identical GPUs/versions and include it in the release bundle). A cache miss is never *incorrect* — any key change just recompiles — so this is pure win. `VLLM_DISABLE_COMPILE_CACHE=1` exists for debugging suspected cache weirdness.
- **Budget restart time in runbooks.** "Restart the pod" is not a 5-second operation for a compiled 400B model; your node-loss and upgrade runbooks (SOP §8) should assume minutes, or pre-warmed standbys.

---

## 8. Memory accounting: where the HBM actually goes

### 8.1 The startup profiling run

At initialization the EngineCore does, in order: load weights → run a **dummy forward pass at maximum configured batch shape** to measure peak activation ("torch") memory → measure non-torch overhead (NCCL buffers, CUDA context) and CUDA-graph memory → and then compute:

```
kv_cache_memory = total_gpu_memory * gpu_memory_utilization
                  - weights - peak_activation - non_torch - cudagraphs
```

Everything in that remainder is carved into KV blocks. The startup log prints the full breakdown (`total_gpu_memory`, `peak_torch_memory`, `non_torch_memory`, `kv_cache_size`, ...) plus the two lines you should actually read:

- **`GPU KV cache size: N tokens`** — the total token capacity of the block pool. This, not GPU count, is your concurrency ceiling.
- **`Maximum concurrency for L tokens per request: X×`** — N divided by your `--max-model-len` L: how many *worst-case* sequences fit simultaneously. Real concurrency is higher because real sequences are shorter than the max, but this is the honest floor.

### 8.2 What `--gpu-memory-utilization` really means

Three properties people get wrong:

1. **It is a fraction of *total* device memory, evaluated once at startup** — a sizing input to the formula above, not a live cap the engine enforces afterward. Default 0.9.
2. **It is per vLLM instance, not per GPU.** Two engines on one GPU at 0.9 each will try to claim 180% and the second one dies at profiling time. Co-located instances must partition explicitly (e.g. 0.45 + 0.45).
3. **Raising it does not make the model faster; it makes the KV pool bigger.** The SOP's `interactive-agent` profile uses 0.90 to keep headroom for allocator spikes and fragmentation of *non*-KV memory; chasing 0.98 mostly buys OOM crashes at 3 a.m.

Startup also logs the exact byte value that reproduces the current allocation for **`--kv-cache-memory`**; passing it back on the next boot skips the profiling pass and the CUDA-graph memory estimation — another restart-time saver worth putting in frozen production configs (re-derive it whenever the model, vLLM version, or flags change).

### 8.3 Back-of-envelope KV math (do this for every model you host)

Per token, per layer, the cache stores one K and one V vector per KV head:

```
bytes/token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
```

Example — a 70B-class dense model (80 layers, 8 KV heads via GQA, head_dim 128):

- FP16/BF16 KV: 2 × 80 × 8 × 128 × 2 B = **320 KiB per token** → a 128k-token agent session holds **~40 GB** of KV.
- FP8 KV (`--kv-cache-dtype fp8`): **160 KiB per token** → **~20 GB** per 128k session.

Now the SOP's hardware stances become arithmetic rather than opinion: on a B200 (192 GB) with, say, 60 GB left for KV after an NVFP4 model, FP8 KV fits ~3 concurrent worst-case 128k sessions per GPU; B300's extra 96 GB roughly doubles that; and MLA-based models (DeepSeek-style latent compression) cut per-token cost by an order of magnitude, which is why they dominate long-context agent serving.

> **Common pitfall — "CUDA out of memory" at startup vs at runtime mean different things.** At startup (during profiling): your utilization fraction, max batch shape, or another process's residency doesn't fit — lower `--gpu-memory-utilization` or `--max-num-batched-tokens`, or find the squatter with `nvidia-smi`. At runtime, true OOM should be nearly impossible (KV is pre-allocated; overload manifests as *preemption and queueing*, §4.3) — runtime OOM usually implicates non-KV growth such as oversized multimodal inputs or a leak, and is bug-report territory.

---

## 9. How model support works

### 9.1 The registry and native model classes

vLLM ships a re-implementation of every supported architecture (`vllm/model_executor/models/`), registered in a **model registry** mapping the `architectures` field of a checkpoint's `config.json` (e.g. `"Qwen3MoeForCausalLM"`) to a vLLM model class. Native classes are what make vLLM fast: they are written against vLLM's tensor-parallel linear layers, fused MoE kernels, quantization schemes (NVFP4/FP8 via ModelOpt or compressed-tensors checkpoints), and the paged attention interface. Out-of-tree architectures can be added without forking via the plugin system (`ModelRegistry.register_model(...)` in a `vllm.general_plugins` entry point) — relevant if you ever host a truly custom internal model.

### 9.2 The Transformers-backend fallback — and its changing cost

When no native class exists, vLLM can run the model's **Hugging Face Transformers modeling code inside the vLLM engine** — the *Transformers modeling backend* (`--model-impl transformers` to force it; `auto` falls back to it). vLLM injects its paged attention through Transformers' attention-interface hooks, so continuous batching and prefix caching still work even though the layer code is generic.

The performance story has changed and you should hold both facts: **historically** the fallback was substantially slower than native classes (missing fusions and parallel layers — the SOP's "expect ~2–4× slower" intake guidance reflects that era, and remains the right *planning* assumption for a brand-new architecture on day 0). **As of July 2026**, joint Hugging Face/vLLM work applies vLLM's fused kernels to Transformers code at runtime (torch.fx static graph analysis plus AST-level source transforms mapping many-op patterns onto single optimized kernels, including fused MoE and tensor/expert parallelism), and the announcement benchmarks show the backend *matching or beating* native throughput on tested dense 4B/32B and 235B-MoE models. Caveats: coverage is per-architecture (linear-attention models are explicitly unsupported as of the announcement), and exotic layers won't match the fusion patterns. Practical rule for your intake gate: treat Transformers-backend-only as "fine for eval, benchmark before production" — and let `guidellm` on your staging GPUs, not either headline, decide.

### 9.3 Multimodal models in one paragraph

For a VLM (vision-language model), the frontend's multimodal registry runs the model-specific processor: images/video/audio are converted to tensors and the prompt gets **placeholder token ranges** where embeddings will be spliced. On the engine side, the vision encoder runs once per input and the result is held in the GPU-resident **encoder cache** (`EncoderCacheManager`), so a prompt chunked across many prefill steps (§4) reuses the embeddings instead of re-running the encoder per chunk; identical media re-sent across requests can also hit this cache. Multimodal inputs participate in prefix caching via content hashes in the block-hash extra keys (§3.2). Operationally: multimodal preprocessing is real frontend CPU work — budget for it — and encoder-cache memory competes with everything else at profiling time.

### 9.4 LoRA adapters at serve time

LoRA (Low-Rank Adaptation) fine-tunes ship as small delta matrices atop a frozen base model. vLLM serves **many LoRAs on one base simultaneously**, batching requests for different adapters into the same forward pass via specialized multi-LoRA kernels:

```bash
vllm serve /models/base --enable-lora \
  --lora-modules support-bot=/models/loras/support-v3 triage=/models/loras/triage-v1 \
  --max-loras 4 --max-lora-rank 64
```

Each adapter appears as its own model name in `/v1/models`, and clients select it via the `model` field. Runtime add/remove exists — `POST /v1/load_lora_adapter` / `/v1/unload_lora_adapter` — but only when `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, and the docs carry an explicit warning that runtime loading is a security risk outside fully trusted environments (it makes "load arbitrary tensors from a path" an API call). In the enclave, prefer startup-time `--lora-modules` from the signed model store; treat the runtime endpoints as disabled-by-default. Note also from §3.2: the LoRA ID is part of the block hash, so different adapters never share cached KV even on identical prompts — many-adapter fleets dilute prefix-cache efficiency.

---

## 10. The Rust frontend (experimental as of Aug 2026)

Everything in §2's *frontend* box — HTTP handling, validation, templating, tokenization, streaming — is CPU-bound Python, and at high request rates it, not the GPU, becomes the bottleneck; the stopgap was running many Python API-server processes (`--api-server-count`). The **Rust frontend** (RFC #40846, merged mid-2026) replaces that whole process with a single Rust binary speaking the same ZMQ/msgpack protocol to the same EngineCore — the process-split architecture paying off exactly as designed. vLLM's own preprocess-heavy benchmark: **~837 req/s from one Rust process vs ~162 req/s for the default Python setup (~5×)**, with one process beating 32 Python workers in the headline comparison.

Enablement is one env var on a stock install:

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve /models/...
```

**Maturity, verified against the feature-parity roadmap (issue #44280) as of August 2026:** working — chat/completions (streaming and not), tool-call and reasoning parsing for key model families, image multimodal, metrics/health/admin routes, multi-engine load balancing; v0.26 added video/audio multimodal, a Seed-OSS tool parser, and a native `vllm-bench` port. **Missing** — Anthropic `/v1/messages`, embeddings, audio-transcription and tokenize/detokenize endpoints, `n>1`, TLS, API-key auth, CORS, LoRA loading/management, and logging parity. The roadmap's own words: *experimental and not yet feature-complete*, with no committed GA date.

**Recommendation for our environment:** do not deploy it in the enclave yet — the missing TLS/auth alone disqualifies it under the SOP's internal-PKI requirements, and `/v1/messages` absence breaks the agentic-gateway story. Track it per release; when parity lands it should slot in with zero client-visible change (same routes, same EngineCore). It is fine to benchmark in staging today.

---

## 11. API server anatomy: routes, tokenizers, templates

### 11.1 The routes you actually serve

One `vllm serve` (Python frontend) exposes, as of v0.26 — availability of some depends on model type and flags:

| Route | Purpose |
|---|---|
| `/v1/chat/completions`, `/v1/completions` | Classic OpenAI text APIs (chat and raw-prompt) |
| `/v1/responses` | Stateful agentic API (SOP §6 migration target; engine-local state — use the agentic-api gateway for durable state) |
| `/v1/embeddings`, `/v1/score`, `/rerank` | Pooling-model tasks |
| `/v1/audio/transcriptions`, `/v1/audio/translations` | Whisper-style audio |
| `/tokenize`, `/detokenize` | Direct tokenizer access (token counting without a client-side tokenizer copy) |
| `/v1/models`, `/health`, `/load`, `/version`, `/metrics` | Discovery, liveness, load report, Prometheus metrics |
| `/v1/load_lora_adapter`, `/v1/unload_lora_adapter` | Runtime LoRA (gated, §9.4) |

Useful serving flags at this layer: `--api-key` (single static key — real authn belongs in your gateway), `--served-model-name` (the public name(s) clients use — the SOP's alias discipline starts here), `--middleware`, and `--allowed-origins` for CORS. `/metrics` is per-replica; scrape every one (SOP §7).

### 11.2 Tokenizer and chat template handling

The frontend loads the model's tokenizer (from the same local path as the weights — with `HF_HUB_OFFLINE=1` nothing may reference the Hub). For `/v1/chat/completions`, the `messages` array is rendered to a single token sequence by the **chat template**: a Jinja2 template that ships in the checkpoint's `tokenizer_config.json` (`chat_template` field) and encodes the model's role markers, system-prompt convention, and — critically for agents — how tool schemas and tool results are formatted into the prompt. Override with `--chat-template /path/to/template.jinja` when the shipped one is missing or broken.

On the output side, the frontend's `OutputProcessor` detokenizes incrementally and runs the **tool-call parser** (`--tool-call-parser`, with `--enable-auto-tool-choice`) and **reasoning parser** (`--reasoning-parser`) over the text stream to produce structured `tool_calls` and separated reasoning content. Note the symmetry: the chat template is the model-specific *encoder* of the conversation, the parsers are the model-specific *decoders* of the response — and both live in the frontend, invisible to the engine. This is why the SOP's Phase-1 intake gate hammers on template sanity and 20 canned tool-call prompts: a wrong template or parser produces *silently* degraded agents on a perfectly healthy engine, and neither error rates nor GPU metrics will tell you.

---

## 12. Releases, reading the notes, and the recipes repo

### 12.1 Cadence and versioning

vLLM ships a **minor release roughly every two weeks** (recent line: v0.25.0 on Jul 11 2026, v0.26.0 on Jul 27 2026, v0.27.0 on Aug 10 2026), with occasional patch releases (v0.25.1, v0.27.1) carrying targeted fixes. A minor release is large — v0.26.0 was 411 commits from 212 contributors; v0.27.0 was 561 commits — so "we're only two versions behind" can mean a thousand commits of drift. Each release publishes wheels and Docker images; the enclave consumes the image by digest per the SOP.

Reading release notes for operational risk, in priority order: (1) **breaking/deprecation section** — flag renames like the `VLLM_ATTENTION_BACKEND` → `--attention-config` migration land here first; (2) **default-behavior changes** — a new default backend, cudagraph mode, or scheduler default changes your perf profile with zero config change on your side; (3) **tool-call / reasoning parser and chat-template changes** — the SOP's engine-upgrade runbook re-runs the tool-call test battery for exactly this reason; (4) day-0 model support, which tells you whether the *next* model drop needs an engine refresh. The monthly-refresh rule in SOP Phase 0 exists because the most common model-drop blocker is a frozen engine that predates the architecture.

### 12.2 The recipes repo as an operational contract

`vllm-project/recipes` (rendered at recipes.vllm.ai) is the project's catalog of **validated launch configurations**: one YAML per model at `models/<hf_org>/<hf_repo>.yaml`, mirroring the Hugging Face path exactly. Each recipe enumerates hardware targets, model variants (including quantizations such as NVFP4), and serving strategies, and the site renders it as a command builder — pick hardware + variant + strategy, copy the exact `vllm serve` command including environment variables. The same data is exported as static JSON for tooling to consume, and the repo's build validates every YAML (`node scripts/build-recipes-api.mjs`).

Why this matters beyond convenience: a recipe is an assertion that *someone ran this exact configuration on this exact hardware and it worked* — flags, env vars, parallelism, and quantization as one tested unit. That is the SOP's "recipes-first, never from scratch" rule in mechanical form, and why the enclave keeps a git mirror of the repo in every release bundle: your deployment configs should be diffs against a recipe, so that when something misbehaves you can bisect *your* deltas instead of debugging an unexplored flag combination. When a recipe for a new model doesn't exist yet, its absence is itself intake signal (SOP Phase 1, step 1).

---

## Study questions

1. **Why did pre-vLLM serving systems waste 60–80% of KV-cache memory, and what two kinds of fragmentation did PagedAttention eliminate?**
   Answer: They allocated each request's KV cache as one contiguous max-length slab, causing internal fragmentation (unused tail of each slab) and external fragmentation (gaps between variable-sized slabs). Fixed-size paged blocks with per-request block tables eliminate both, leaving at most one partially-filled block per sequence.

2. **What are the two processes in a V1 `vllm serve` deployment, how do they communicate, and which one owns tokenization?**
   Answer: The API server (frontend) process and the EngineCore process, communicating over ZeroMQ with msgpack serialization. Tokenization, chat templating, and detokenization all live in the frontend; the EngineCore only ever sees token IDs.

3. **How does the V1 scheduler represent its per-step decision, and why does that unify continuous batching and chunked prefill?**
   Answer: As a dictionary `{request_id: num_tokens_this_step}` spent against a token budget (`--max-num-batched-tokens`), with no prefill/decode distinction. Decodes contribute 1 token each and are scheduled first; remaining budget goes to prefills, which are chunked automatically when they exceed it — so mixed batches and chunking fall out of one mechanism.

4. **A colleague adds `Generated at 2026-08-12T09:14:03Z` to the top of the agent system prompt. What breaks, and how would you detect it?**
   Answer: Prefix-cache hits collapse to near zero — the changed bytes alter block 0's token IDs, and because each block's hash chains from its parent, every downstream block hash changes too. Detect via the prefix-cache hit-rate metric dropping (the SOP treats this as an incident) and a corresponding TTFT increase; nothing errors.

5. **What does copy-on-write mean in the PagedAttention paper, and what happened to it in V1?**
   Answer: In the paper (and V0), sequences sharing a block (parallel sampling, beam search) held refcounts, and the first write to a shared block triggered a private copy. V1 achieves the same outcome without explicit CoW: full blocks are shared read-only via prefix-cache refcounts, and each sequence always writes into its own tail block.

6. **What does the log line `Maximum concurrency for 131072 tokens per request: 3.2x` tell you, and what is the cheapest flag to roughly double it?**
   Answer: The KV block pool can hold about 3.2 simultaneous worst-case 131k-token sequences (pool token capacity ÷ max-model-len). `--kv-cache-dtype fp8` halves per-token KV bytes and roughly doubles it — validated on your evals, since a few models regress.

7. **Async scheduling overlaps what with what, and what bet does it make?**
   Answer: It overlaps CPU scheduling of step N+1 with GPU execution of step N. The bet is that each running decode produces exactly one token; when a request instead stops, the engine reconciles by discarding its pre-scheduled slot. Default on since v0.14.0 (Feb 2026).

8. **After an engine upgrade, throughput drops ~30% with no errors. Name two silent-fallback mechanisms from this document you would check first.**
   Answer: (1) Attention backend selection — check the startup log for which backend was chosen; an import failure or renamed flag (e.g. relying on deprecated `VLLM_ATTENTION_BACKEND`) can silently drop you from FA4/FlashInfer to the Triton fallback. (2) Model implementation — the model may have fallen back to the Transformers modeling backend, or a recipe env var like `VLLM_USE_FLASHINFER_MOE_FP4=1` was lost, disabling the fused FP4 MoE path.

9. **Why does `--gpu-memory-utilization 0.9` on two vLLM instances sharing one GPU fail, and when is the value even enforced?**
   Answer: It is a per-instance fraction of *total* device memory evaluated once during the startup profiling pass, not a live global cap — two instances at 0.9 attempt 180% and the second fails at profiling. Co-located instances must explicitly split (e.g. 0.45 each).

10. **What is piecewise CUDA graph capture and why is it the compatible default?**
    Answer: torch.compile splits the model graph at every attention op; the non-attention segments are captured as CUDA graphs (killing launch overhead) while attention runs eagerly. Because attention kernels are the graph-capture-hostile part, piecewise works with every attention backend; full-graph modes (`FULL_AND_PIECEWISE` etc.) add attention into the graph only where the backend supports it.

11. **What operational steps shrink vLLM cold-start time in an air-gapped deployment?**
    Answer: Persist/ship `~/.cache/vllm/torch_compile_cache` (pre-baked in staging on identical GPU/version — a stale key is only a cache miss, never a correctness failure), and pass the startup-logged `--kv-cache-memory` value to skip the memory-profiling and CUDA-graph estimation pass. Budget minutes, not seconds, in restart runbooks regardless.

12. **Why is the Rust frontend not yet appropriate for our enclave despite its ~5× frontend throughput, and what makes it a drop-in later?**
    Answer: As of Aug 2026 it is explicitly experimental and missing TLS, API-key auth, LoRA management, `/v1/messages`, embeddings, and tokenize endpoints — disqualifying under SOP PKI and gateway requirements. Because it speaks the same ZMQ/msgpack protocol to the same EngineCore and serves the same routes, reaching parity makes adoption a one-env-var change (`VLLM_USE_RUST_FRONTEND=1`) with no client impact.

---

## Sources

Primary sources consulted August 2026. Fast-moving items are marked with the version/date they describe.

- vLLM architecture overview (V1 process split, EngineCore, ZMQ): https://docs.vllm.ai/en/latest/design/arch_overview/
- vLLM V1 alpha announcement (Jan 27, 2025): https://vllm.ai/blog/2025-01-27-v1-alpha-release
- V0 deprecation RFC (#18571) and removal completed in v0.11.0: https://github.com/vllm-project/vllm/issues/18571 , https://github.com/vllm-project/vllm/releases/tag/v0.11.0
- PagedAttention paper — Kwon et al., SOSP 2023: https://arxiv.org/abs/2309.06180 ; background summary: https://en.wikipedia.org/wiki/PagedAttention
- Automatic prefix caching design (block hashing, SHA-256 default since v0.11, cache_salt): https://docs.vllm.ai/en/stable/design/prefix_caching/
- Optimization and tuning (chunked prefill, decode-first scheduling, gpu-memory-utilization, --kv-cache-memory, preemption): https://docs.vllm.ai/en/stable/configuration/optimization/
- Scheduler config reference (async_scheduling, policies): https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
- Async scheduling default in v0.14.0 (Feb 2026): https://agentnativedev.medium.com/vllm-v0-14-0-async-scheduling-grpc-and-deployability-b042bbe40312 ; plan issue: https://github.com/vllm-project/vllm/issues/27679
- Attention backend feature support / selection: https://docs.vllm.ai/en/latest/design/attention_backends/
- Triton attention backend deep dive (vLLM blog, Mar 4, 2026): https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive
- FlashAttention-4 (Dao et al., Mar 2026; B200 numbers): https://tridao.me/blog/2026/flash4/
- VLLM_ATTENTION_BACKEND deprecation → --attention-config: https://docs.vllm.ai/en/latest/api/vllm/config/attention/
- CUDA graphs design (cudagraph_mode, dispatcher): https://docs.vllm.ai/en/stable/design/cuda_graphs/
- torch.compile integration + compile cache: https://docs.vllm.ai/en/latest/design/torch_compile/ ; intro blog (Aug 2025): https://blog.vllm.ai/2025/08/20/torch-compile.html
- Cold-start measurement with compile caching (Jun 2026): https://route179.dev/2026/06/30/optimizing-vllm-cold-start-with-model-streaming-and-compile-caching/
- Transformers modeling backend at native speed (HF blog, Jul 8, 2026): https://huggingface.co/blog/native-speed-vllm-transformers-backend ; original integration blog (Apr 2025): https://blog.vllm.ai/2025/04/11/transformers-backend.html
- Multimodal V1 design (encoder cache): https://developers.redhat.com/articles/2025/02/27/vllm-v1-accelerating-multimodal-inference-large-language-models
- LoRA adapters (startup + runtime loading, security note): https://docs.vllm.ai/en/latest/features/lora/
- Rust frontend RFC / merge / parity roadmap: https://github.com/vllm-project/vllm/issues/40846 , https://github.com/vllm-project/vllm/pull/40848 , https://github.com/vllm-project/vllm/issues/44280
- OpenAI-compatible server (routes, chat templates): https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- Releases (cadence; v0.26.0 Jul 27 2026, v0.27.0 Aug 10 2026): https://github.com/vllm-project/vllm/releases , https://github.com/vllm-project/vllm/releases/tag/v0.26.0
- Recipes repo structure and validation: https://github.com/vllm-project/recipes , https://github.com/vllm-project/recipes/blob/main/CONTRIBUTING.md
