# Primer: Concepts Behind the AI Deployment SOP

**Read this first.** The [deployment SOP](./SOP-production-ai-model-deployment.md) assumes you already know the vocabulary of LLM inference. This document builds that vocabulary from the ground floor, in the order the ideas stack on each other, and ends with a glossary of every acronym the SOP uses.

---

## 1. What happens when a model serves a request

When an LLM answers a prompt, there are two very different phases:

- **Prefill** — the model reads your entire prompt in one big parallel pass. This is compute-heavy, like skimming a 50-page briefing. The time it takes is the delay before the first word appears: **TTFT (time to first token)**.
- **Decode** — the model then generates the response one token at a time, each token depending on everything before it. This phase is limited by memory bandwidth, not compute. Its pace is the **inter-token latency (ITL)** — how fast words stream out.

While reading the prompt, the model builds working memory about it called the **KV cache** (key-value cache) — think of it as the model's notes on the conversation so far. It lives in GPU memory and it's *big*: a long conversation can occupy gigabytes per user.

> **The fact that drives half the SOP:** GPU memory is mostly spent on *remembering conversations*, not on the model itself.

## 2. What vLLM is and why we need it

A GPU serving one user at a time would sit idle 95% of the time. **vLLM is a serving engine** — a program that takes raw model weights and turns them into an efficient web service. It juggles hundreds of simultaneous conversations on the same GPUs by:

- **batching** many users' work into each GPU pass,
- managing all those KV caches without wasting memory (its founding innovation, **PagedAttention**, manages GPU memory the way an operating system manages RAM), and
- exposing an **OpenAI-compatible API**, so any tool built for ChatGPT-style APIs can point at your own hardware instead.

You don't tune it from scratch. The community maintains a **recipes repo** ([vllm-project/recipes](https://github.com/vllm-project/recipes)) — tested "exact launch command for model X on hardware Y" files. That's why the SOP's rule is *start from the recipe*.

## 3. Quantization — why "NVFP4" keeps coming up

Model weights are billions of numbers, normally stored at 16 bits each. **Quantization** stores them in fewer bits — 8 (**FP8**) or 4 (**FP4**) — like compressing a photo: slightly less precise, half or a quarter the size.

The key hardware fact: B200/B300 GPUs have **circuits that do math directly on 4-bit numbers** (**NVFP4** is NVIDIA's 4-bit format). So a quantized model isn't just smaller — it's *faster*. A model needing two servers at full precision fits on half of one in NVFP4. That's why the SOP makes NVFP4 the production default: the same hardware serves roughly 2–4× more, usually with negligible quality loss — which is exactly what the eval gate exists to verify.

The same trick applies to conversation memory: `--kv-cache-dtype fp8` compresses the KV cache, letting each server hold about twice as many concurrent conversations.

## 4. Parallelism — splitting a model across GPUs

Big models don't fit on one GPU, so they get split:

- **Tensor parallelism (TP):** slice every layer across the 8 GPUs inside one server, all working in lockstep. The default. It needs the ultra-fast **NVLink** wiring *inside* a server and doesn't stretch well *between* servers.
- **Pipeline parallelism (PP):** put different layers on different GPUs, like an assembly line. A fallback when TP alone can't fit the model; adds latency.
- **Expert parallelism (EP):** some models are **Mixture-of-Experts (MoE)** — instead of one giant brain, they contain many smaller "expert" sub-networks, and each token activates only a few (DeepSeek-style models work this way). EP places different experts on different GPUs. **Wide-EP** does that across *multiple servers* — the highest-throughput setup and also the hardest to operate, because it needs a validated high-speed network (InfiniBand/RoCE with RDMA) between servers.

The SOP's stance: don't go multi-node until a single server genuinely can't keep up.

## 5. The agent-specific tricks

### Prefix caching (the big one)

An AI agent works in a loop: think → call a tool → read the result → think again. Every turn, it re-sends the *entire* conversation — system prompt, tool definitions, every previous step — plus one new bit. Without caching, the model re-reads all of that from scratch each turn (a full prefill, every time). **Prefix caching** says: "I already have notes (KV cache) on this exact text — skip straight to the new part." For agents this can cut response delay 5–10×.

Two practical consequences in the SOP:

1. **Keep prompts byte-identical between turns.** A timestamp embedded in your system prompt silently destroys caching.
2. **Route a session back to the same server replica each turn** ("session-affinity" / "prefix-aware" routing) — the cache lives in one specific machine's GPU memory. Round-robin load balancing throws the benefit away.

### Speculative decoding

Decode is slow because it's one token at a time. The trick: a tiny fast "draft" predictor guesses the next several tokens, and the big model verifies all the guesses in a single pass. When guesses are right — often, especially for predictable output like tool-call JSON — generation runs 1.5–2.5× faster. (**MTP** and **EAGLE** in the SOP are two flavors of draft predictor; "n-gram" is a zero-cost one that just looks for repeated text.)

### Tool-call parsers

When a model wants to call a tool, it writes that request in its own text format — and every model family formats it differently. vLLM needs a small **parser** for that exact format to convert the model's text into the structured JSON your agent framework expects. A missing or mismatched parser means the model "works" but every tool call comes out as garbage text. This is the most common way a new model silently breaks agents — which is why the SOP makes "parser exists and passes 20 test prompts" a hard gate. **Reasoning parsers** are the same idea for separating a model's internal "thinking" from its final answer.

### Structured output (guided decoding)

The server can *force* the model's output to match a JSON schema — it literally blocks tokens that would break the format. Turn this on for anything a machine parses; it turns "usually valid JSON" into "always valid JSON."

### Prefill/decode disaggregation (P/D)

At high scale, one user's huge prompt (long prefill) can stall token generation for everyone else sharing the GPU. The fix is to run **separate GPU pools**: some servers only do prefill, others only decode, with the KV cache handed off over the network (via a transfer library called **NIXL**). Powerful, complex — the SOP says adopt it only when latency targets demand it, and via an orchestration layer (llm-d), not hand-rolled.

## 6. Docker vs Kubernetes, in plain terms

**Docker** packages a program with everything it needs into a **container** — a sealed box that runs identically anywhere. You'll use Docker regardless; that part isn't a choice. **Docker Compose** is a small tool that starts a fixed set of containers together on one machine from a config file.

**Kubernetes (K8s)** is a *manager for fleets of containers across many servers*. You declare "I want 3 copies of this model running at all times," and it decides which machines run them, restarts crashes, and shifts traffic during upgrades. The tradeoff: Kubernetes is itself a complex system needing care and feeding — and in an air-gapped environment, feeding it (its own upgrades and images) is real ongoing work.

Rule of thumb: with 1–3 servers and a handful of models, **a human can be the scheduler** — Docker plus a small routing gateway is simpler and just as reliable. Kubernetes earns its cost when the decisions outgrow humans: many models coming and going, one model spanning multiple servers, auto-scaling, or hard zero-downtime requirements. The SOP's stance: start simple, but **write down in advance the trigger that makes you switch**, so the migration is a planned project instead of a panic.

Names that ride along with Kubernetes in the SOP: **GPU Operator** (makes GPUs usable inside K8s), **llm-d** (Kubernetes-native orchestration built specifically for vLLM — smart routing, P/D, wide-EP), **KServe** (a general model-serving layer), **Kueue** (GPU quota/queueing between teams), **Dynamo** (NVIDIA's non-Kubernetes alternative orchestrator).

## 7. Air-gapped operations — the bundle idea

Production has no internet, so the SOP splits the world in two:

- A **connected staging zone**, where downloading is allowed: model weights, container images, the recipes repo. Also where you verify, test, and sign things.
- The **enclave** (production), which only ever receives sealed, checksummed, signed **bundles**, carried across the gap on physical media or through a one-way transfer device.

Inside the enclave, everything a deployment references must already exist locally:

- **Harbor** — a local container image registry (the enclave's private Docker Hub). Every manifest points at it; nothing points at the internet.
- **Model store** — weights are big plain files on shared storage (NFS or a Kubernetes volume), *not* baked into container images, so models and software can be updated independently.
- `HF_HUB_OFFLINE=1` — the setting that tells vLLM "never try to phone the internet; everything is a local file."

The discipline that makes this work: **everything version-pinned, checksummed, and signed** — because inside the enclave there is no "pip install will fix it later."

## 8. The API migration — stateless vs stateful

`/v1/chat/completions` (the classic OpenAI API everything uses today) is **stateless**: the server remembers nothing between calls. Your agent re-sends the whole conversation every turn, and your client code owns the entire agent loop.

The **Responses API** (`/v1/responses`) is **stateful**: after each turn the server stores the conversation and hands back a receipt ID. Next turn, the client sends only the new input plus `previous_response_id: "resp_abc123"`, and the server picks up where it left off. Why it's better for agents:

- requests shrink dramatically (no more re-sending everything),
- the server can preserve the model's internal *reasoning* between turns — reasoning models genuinely perform better this way,
- tools can optionally execute **server-side**, so every team stops reimplementing the agent loop differently.

The migration plan in one breath: put a **gateway** in front of the models (one front door — the `agentic-api` project is a purpose-built one), have it speak *both* APIs during the transition, migrate agent clients one at a time, and keep the old API alive for boring tooling that doesn't need state. One new responsibility: the server now stores conversation state, so that database needs backups and a retention policy like anything else in production.

## 9. How it all connects

Reading the SOP back with these concepts: **vLLM turns GPUs into an API** → **quantization and KV tricks** decide how much fits per server → **prefix caching + parsers** make *agents* specifically work well → **Docker-then-Kubernetes** is how you run it → **bundles** are how anything enters the building → the **Responses API** is where the client side is heading. The "fast model drop" section is all of the above pre-staged, so a new model release only fills in blanks.

---

## Glossary

### Performance & serving

| Term | Meaning |
|---|---|
| **TTFT** | Time to first token — delay before the response starts. Dominated by prefill. |
| **ITL / TPOT** | Inter-token latency / time per output token — how fast words stream once started. |
| **p50 / p99** | Median / 99th-percentile of a latency distribution. SLOs are usually set on p99 (the worst realistic case). |
| **SLO** | Service level objective — the latency/quality target you commit to (e.g. "TTFT p99 ≤ 2 s"). |
| **Goodput** | Throughput that *meets the SLO* — requests served fast enough to count. |
| **Prefill / decode** | The two phases of inference: read the prompt (parallel, compute-bound) / generate tokens one-by-one (memory-bound). |
| **KV cache** | The model's per-conversation working memory, stored in GPU memory. The main consumer of GPU RAM at scale. |
| **PagedAttention** | vLLM's technique for managing KV cache memory in small pages, like an OS manages RAM. |
| **Prefix caching** | Reusing the KV cache for a prompt prefix the server has already seen. The #1 agent-latency win. |
| **Chunked prefill** | Slicing a long prompt-read into pieces so it doesn't stall other users' token generation. |
| **Speculative decoding** | A fast draft predictor guesses several tokens; the big model verifies them in one pass. 1.5–2.5× faster decode. |
| **MTP / EAGLE / n-gram** | Flavors of speculative-decoding draft predictor (built into the model / a trained add-on head / simple text-repetition lookup). |
| **CUDA graphs** | Pre-recording GPU work sequences to cut per-step launch overhead. |
| **Structured / guided decoding** | Forcing output to match a JSON schema by blocking invalid tokens (backed by the **xgrammar** library). |
| **Tool-call parser** | vLLM component that converts a model's own tool-call text format into standard structured JSON. Model-family-specific. |
| **Reasoning parser** | Same idea, for separating a model's "thinking" from its final answer. |
| **guidellm** | Load-testing tool for LLM servers — used for the performance gate. |

### Hardware

| Term | Meaning |
|---|---|
| **B200 / B300** | NVIDIA Blackwell / Blackwell Ultra data-center GPUs (192 GB / 288 GB memory). |
| **HBM** | High-bandwidth memory — the GPU's onboard RAM. |
| **HGX** | NVIDIA's standard 8-GPU server board design. |
| **NVL72 / GB300** | Rack-scale systems wiring 72 GPUs (with Grace CPUs) into one NVLink domain. |
| **NVLink** | Ultra-fast GPU-to-GPU wiring inside a server/rack. |
| **InfiniBand / RoCE** | High-speed networking between servers (dedicated fabric / Ethernet-based equivalent). |
| **RDMA / IBGDA / NVSHMEM** | Tech letting GPUs read each other's memory across the network directly, without CPU involvement. Required for wide-EP. |
| **NVFP4 / FP8 / BF16** | Number formats: NVIDIA 4-bit (production default on Blackwell) / 8-bit / 16-bit (full-quality baseline). |
| **DCGM** | NVIDIA's GPU health-monitoring toolkit. |

### Model architecture & parallelism

| Term | Meaning |
|---|---|
| **MoE** | Mixture-of-Experts — a model made of many small expert sub-networks, few active per token. |
| **TP / PP / EP / DP** | Ways to split work across GPUs: tensor (slice each layer) / pipeline (assembly line of layers) / expert (spread MoE experts) / data (duplicate the model, split the users). |
| **Wide-EP** | Expert parallelism spanning multiple servers. Highest throughput, highest complexity. |
| **EPLB** | Expert-parallel load balancer — keeps popular experts from hot-spotting one GPU. |
| **DeepEP** | Optimized communication kernels for MoE traffic between GPUs (high-throughput and low-latency modes). |
| **NIXL** | NVIDIA's library for shipping KV cache between servers — the plumbing of P/D disaggregation. |
| **P/D disaggregation** | Running prefill and decode on separate GPU pools so long prompts don't stall token generation. |
| **ModelOpt / llm-compressor** | NVIDIA's / the open-source toolkit for producing quantized checkpoints. |

### Deployment & operations

| Term | Meaning |
|---|---|
| **Container / Docker** | A sealed package of a program plus everything it needs, running identically anywhere. |
| **Docker Compose** | Config-file tool to start a fixed set of containers together on one machine. |
| **Kubernetes (K8s)** | Fleet manager for containers across many servers: placement, restarts, rollouts, scaling. |
| **GPU Operator** | NVIDIA add-on that makes GPUs usable inside Kubernetes. |
| **llm-d** | Kubernetes-native orchestration built for vLLM: prefix-aware routing, P/D, wide-EP. CNCF Sandbox project. |
| **KServe / Kueue** | Kubernetes model-serving layer / GPU quota-and-queueing between teams. |
| **Dynamo** | NVIDIA's orchestration layer for disaggregated serving that runs outside Kubernetes. |
| **Gateway API Inference Extension** | Kubernetes standard for LLM-aware traffic routing (what llm-d's scheduler plugs into). |
| **CNCF** | Cloud Native Computing Foundation — vendor-neutral home for Kubernetes-ecosystem projects. |
| **LiteLLM** | Popular open-source API gateway/router for LLM endpoints. |
| **agentic-api** | vLLM's gateway providing the stateful Responses API (and Anthropic's protocol) in front of vLLM. |
| **Canary** | Sending a small % of real traffic to a new version before full rollout. |
| **Blue/green** | Running old and new versions side by side and flipping traffic between them. |
| **Alias** | A stable name clients call (`agent-default`) that ops re-points at whichever model is current. Makes promotion/rollback a config flip. |
| **PVC / NFS** | Kubernetes persistent storage volume / classic shared network file storage. |

### Air-gap & security

| Term | Meaning |
|---|---|
| **Air gap / enclave / disconnected** | A network with no internet connection, by policy. |
| **CDS / data diode** | Approved one-way devices for moving data into a secured network. |
| **Harbor** | Self-hosted container image registry — the enclave's private Docker Hub. |
| **skopeo / ORAS** | Tools for copying container images / arbitrary artifacts between registries and archives. |
| **cosign** | Tool for cryptographically signing container images. |
| **Digest pinning** | Referencing an image by its content hash (`@sha256:…`) instead of a movable tag — guarantees you run exactly what was tested. |
| **SHA-256 / checksum** | Content fingerprint used to verify files weren't corrupted or tampered with in transfer. |
| **devpi** | Self-hosted Python package (PyPI) mirror. |
| **HF_HUB_OFFLINE=1** | Environment setting telling Hugging Face libraries (and vLLM) to never attempt internet access. |
| **PKI / CA** | Your internal certificate authority — issues the TLS certificates services use inside the enclave. |
| **Prometheus / Grafana / Alertmanager / Loki** | The standard self-hosted monitoring stack: metrics collection / dashboards / alerts / logs. |

### APIs

| Term | Meaning |
|---|---|
| **/v1/chat/completions** | The classic stateless OpenAI API — client re-sends the full conversation every turn. Legacy for agents. |
| **/v1/responses (Responses API)** | The stateful successor — server stores the conversation; client continues it via `previous_response_id`. |
| **previous_response_id** | The "receipt" a client sends to continue a stored conversation server-side. |
| **/v1/messages** | Anthropic's API protocol — supported by the gateway for Claude-Code-style clients. |
| **SSE / WebSocket** | Two ways of streaming tokens over a connection (one-way stream / two-way channel). |
