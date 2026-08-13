# LLM Gateways and Routers: LiteLLM vs the Field (and What to Actually Run)

**Deep-dive 09 — companion to the [deployment SOP](../SOP-production-ai-model-deployment.md) and [PRIMER](../PRIMER.md).**
**Written August 2026. Version-sensitive facts are dated inline; re-verify quarterly — this corner of the ecosystem changes monthly.**

The question this document answers: **"Is LiteLLM the best option?"** — asked for a specific environment: on-prem, air-gapped, NVIDIA B200/B300 GPUs, vLLM as the only serving engine, agentic (tool-calling loop) workloads, and a planned migration to the OpenAI Responses API (`/v1/responses`).

Short version of the verdict (argued in full in §5): **No — not as the centerpiece.** LiteLLM is a genuinely good product whose headline value — translating one API to 100+ cloud providers — is worth nothing behind an air gap where every backend is your own vLLM. What remains (virtual keys, budgets, an admin UI) is real but narrow, and it comes bundled with a Python data plane whose throughput ceiling is documented by its own maintainers. The right answer here is layered: the `agentic-api` gateway for stateful agent traffic, a thin high-performance proxy for the front door, and — only if you need per-team key and budget accounting *today* — LiteLLM as a sidecar control plane, positioned so it is easy to remove later.

---

## Key takeaways

- **"Gateway" names at least five different layers.** Client SDK shims, API gateways (auth/keys/budgets), inference load balancers (cache-aware routing across replicas), semantic routers (model selection by prompt), and stateful agentic gateways (Responses API state) solve different problems and mostly stack rather than compete. Comparing LiteLLM to llm-d is a category error.
- **LiteLLM's core value proposition — uniform access to 100+ heterogeneous providers — evaporates in an air-gapped enclave** where every backend is a vLLM replica that already speaks the same OpenAI-compatible API. What survives is key/budget/team management and an admin UI.
- **LiteLLM's Python data plane has a documented throughput ceiling.** Its own benchmarks show ~2 ms median / ~13 ms p99 added overhead on a tuned 4-instance deployment (v1.79.1); community reports show a single instance cutting vLLM throughput from ~16 to ~9 requests per second at 500-way concurrency (issue #21046, Feb 2026). LiteLLM itself launched a Rust gateway rewrite in July 2026 — the clearest possible admission of the ceiling.
- **LiteLLM proxy features (keys, budgets, teams) require PostgreSQL**, and several "enterprise" features live in a commercially-licensed `enterprise/` directory inside the otherwise-MIT repo; the boundary has been a source of community confusion.
- **The stateful `/v1/responses` layer is a separate product category.** vllm-project's `agentic-api` (Rust, Apache-2.0) owns response persistence, tool-ownership routing, and the Anthropic `/v1/messages` protocol; no general-purpose gateway replaces it. LiteLLM's Responses API support is a translation/affinity layer, not a state store designed for self-hosted vLLM fleets.
- **Envoy AI Gateway hit v1.0 (GA) in June 2026** with token-based (not request-based) rate limiting, stable Kubernetes CRDs, and Gateway API Inference Extension (InferencePool) support — making "Envoy front door + GAIE/llm-d routing underneath" the standard Kubernetes-tier answer.
- **The routing layer under the gateway matters more than the gateway for agent latency.** Prefix-cache-aware routing (llm-d's inference scheduler / GAIE's Endpoint Picker) preserves the prefix-cache hits that dominate agent TTFT; a round-robin gateway silently throws them away.
- **Every hosted gateway (OpenRouter, Portkey Cloud, Kong Konnect, Vercel AI Gateway) is not applicable** in a disconnected enclave; evaluate only what you can run entirely from your own Harbor registry.
- **Recommended stack:** Compose tier — `agentic-api` + thin nginx/Envoy front door (LiteLLM only if per-team budgets are needed now). Kubernetes tier — Envoy AI Gateway + llm-d/GAIE routing + `agentic-api`. All layers speak OpenAI-compatible APIs, so switching costs are dominated by key migration, not client rewrites.
- **Benchmark gateways against your real vLLM backend, never a mock upstream.** Most published gateway numbers measure forwarding against a stub server, which deletes the variable that matters (behavior under slow, streaming, long-lived connections).

---

## 1. Taxonomy: five layers that all get called "a gateway"

The LLM networking ecosystem grew fast and the word "gateway" got stretched over five genuinely different jobs. Before comparing products, place them. An analogy: an airport has check-in desks (identity, tickets, baggage rules), a departures board (which gate for which flight), gate agents (boarding the right people onto the right plane), and air-traffic control (which runway, in what order). Calling all of them "the airport" is true but useless when you are hiring.

### 1.1 The five layers

**Layer A — Client-side SDK shim.** A library inside your application that presents one function-call interface over many providers. LiteLLM's *SDK* (`litellm.completion(...)`), the OpenAI client pointed at a custom `base_url`, and LangChain's model wrappers live here. No server, no network hop, nothing to deploy. In your environment the OpenAI SDK with `base_url` set to your internal endpoint already does this job, because every backend speaks the OpenAI dialect.

**Layer B — API gateway / proxy.** A server between clients and models that owns *access*: authentication, API key issuance ("virtual keys"), per-team budgets and rate limits, provider abstraction, retries/fallbacks, request logging, and usually an admin UI. LiteLLM *proxy*, Envoy AI Gateway, Kong AI Gateway, Apache APISIX with AI plugins, Portkey's gateway, and Higress live here. This is the layer the "Is LiteLLM best?" question is really about.

**Layer C — Inference load balancer / router.** A traffic-steering layer that decides *which replica* of a model pool serves each request, using engine-level signals: KV-cache (key-value cache — the model's per-conversation working memory) utilization, queue depth, and crucially **prefix-cache state** (which replica already holds this session's cached prompt). This is the llm-d inference scheduler and the Kubernetes **Gateway API Inference Extension (GAIE)** with its Endpoint Picker. It is not a competitor to Layer B — it runs *underneath* whatever API gateway you choose. Round-robin at this layer destroys agent prefix-cache hit rates (SOP §2.3).

**Layer D — Semantic router.** A classifier that picks *which model* should answer, based on the prompt itself — send easy questions to the small cheap model, hard ones to the frontier model, and optionally toggle reasoning modes. The vllm-project **Semantic Router** (an Envoy external processor using ModernBERT-class classifiers) is the reference implementation. Optional; valuable only once you serve multiple model tiers behind one alias.

**Layer E — Stateful agentic gateway.** A server that owns *conversation state* for the Responses API: it persists each response, hydrates `previous_response_id` on the next turn, streams over SSE (Server-Sent Events) or WebSocket, and executes gateway-owned tools server-side. vllm-project's **`agentic-api`** is the purpose-built implementation and the SOP §6 recommendation. General API gateways at Layer B either lack this or approximate it with session-affinity tricks.

### 1.2 Layered reference architecture

Where each product sits in *your* target stack:

```
            Agent clients (OpenAI SDK / Anthropic-protocol clients)
                                 |
   ┌─────────────────────────────▼──────────────────────────────┐
   │ LAYER B: API gateway  — TLS, authn, virtual keys, budgets, │
   │ rate limits, audit logging                                 │
   │   candidates: Envoy AI Gateway | LiteLLM | APISIX | nginx  │
   └─────────────────────────────┬──────────────────────────────┘
                                 |
   ┌─────────────────────────────▼──────────────────────────────┐
   │ LAYER E: agentic gateway (vllm-project/agentic-api)        │
   │   /v1/responses  /v1/messages  /v1/chat/completions        │
   │   response state DB, tool-ownership routing, SSE/WebSocket │
   └─────────────────────────────┬──────────────────────────────┘
                                 |
   ┌─────────────────────────────▼──────────────────────────────┐
   │ LAYER D (optional): semantic router — model tier selection │
   │   candidate: vLLM Semantic Router (Envoy ext-proc)         │
   └─────────────────────────────┬──────────────────────────────┘
                                 |
   ┌─────────────────────────────▼──────────────────────────────┐
   │ LAYER C: inference load balancer — per-replica routing     │
   │   K8s: llm-d inference scheduler / GAIE Endpoint Picker    │
   │   Compose: consistent-hash session affinity (nginx/HAProxy)│
   └───────┬──────────────────┬───────────────────┬─────────────┘
           |                  |                   |
      ┌────▼─────┐       ┌────▼─────┐        ┌────▼─────┐
      │ vLLM     │       │ vLLM     │        │ vLLM     │
      │ replica 1│       │ replica 2│        │ replica 3│
      └──────────┘       └──────────┘        └──────────┘
        (per-model pools, SOP §2 serving profiles, B200/B300)
```

Two placement notes. First, Layers B and E can be collapsed at small scale — `agentic-api` can *be* your front door on the Compose tier if your tenancy needs are modest. Second, Layer C is invisible to clients; it is pure operations. The layers are composable precisely because everything speaks the same OpenAI-compatible wire protocol.

> **Common pitfall:** teams evaluate "LiteLLM vs llm-d" or "Kong vs vLLM Semantic Router" as either/or choices. These pairs are different layers. The real decisions are one per layer: *what is my API gateway* (B), *what routes across replicas* (C), *do I need model-tier selection* (D), and *what owns Responses state* (E).

---

## 2. LiteLLM deep dive

### 2.1 What it is

LiteLLM (BerriAI) is two products sharing one repo:

1. **The SDK** — a Python library (`pip install litellm`) that lets `litellm.completion(model="anthropic/...", ...)` call 100+ providers through one OpenAI-shaped interface.
2. **The proxy** (marketed as "LiteLLM AI Gateway") — a Python server (FastAPI-based) that exposes an OpenAI-compatible endpoint and adds the Layer B feature set: virtual keys, teams, budgets, rate limits, routing/fallbacks, caching, guardrail hooks, logging integrations, and an admin web UI.

It is one of the most-deployed pieces of LLM infrastructure in existence, moves extremely fast (weekly minor releases; v1.93.0 shipped July 19, 2026, with nightlies in the v1.9x range as of mid-August 2026), and has the largest provider-translation matrix in the field.

### 2.2 Feature inventory (verified against docs, August 2026)

- **Provider translation:** 100+ providers normalized to the OpenAI format — OpenAI, Anthropic, Bedrock, Vertex, Azure, local engines (vLLM, Ollama), and everything in between. *This is the flagship feature.*
- **Virtual keys:** issued via `POST /key/generate` against the proxy, authorized by a master key. **Requires PostgreSQL** (`DATABASE_URL`); keys, teams, users, and spend live in the database.
- **Budgets and rate limits:** per-key, per-team, per-user spend caps with daily/weekly/monthly/yearly durations; requests-per-minute (RPM) and tokens-per-minute (TPM) limits; tag-based budgets for cost-center accounting.
- **Routing and reliability:** multiple deployments per model name with load-balancing strategies, automatic retries, provider fallback chains, cooldowns for failing deployments.
- **Caching:** exact-match response caching backed by Redis (also in-memory/S3/disk options).
- **Guardrail hooks:** a plugin interface for pre/post-call checks (PII masking, moderation). The fuller guardrails suite is enterprise-tier.
- **Admin UI:** key/team management, spend dashboards, model management — this is the feature non-platform stakeholders actually see.
- **Prometheus metrics:** a `/metrics` endpoint. As of the August 2026 docs this is **free/OSS** (with at least one enterprise-only metric family, for managed batches). Historically it was enterprise-gated, and third-party writeups still describe it that way — check the current docs page, not blog posts, before you plan around it.
- **Responses API:** `/v1/responses` is supported and *bridged* — for providers without a native Responses implementation, LiteLLM translates to `/chat/completions` under the hood. It maintains session continuity by pinning `previous_response_id` follow-ups to the originating deployment, and can persist prompts to cold storage (S3-compatible) when configured. This is affinity plus translation, not a purpose-built self-hosted state store.
- **Config model:** a single `config.yaml` (`model_list`, `router_settings`, `general_settings`, `litellm_settings`) plus environment variables plus database-resident state. At fleet scale this file famously grows into hundreds of lines with per-deployment keys, fallbacks, and callback settings — versionable, but sprawling.

**Licensing:** the repo is MIT **except** the `enterprise/` directory, which carries a commercial license; enterprise features (SSO/SAML, SCIM, audit logs, full guardrails, JWT auth, and support SLAs) activate via a license key on the same Docker image (Basic tier ~$250/month, Premium ~$30k/year per third-party pricing writeups). A 2026 GitHub issue (#34241) documents community confusion about where the MIT/enterprise line actually falls in the code — some gated features are technically MIT-licensed code behind a flag. For an enclave, the operational takeaway is: assume anything labeled "enterprise" in the docs needs a paid key, and verify the specific features you need against the current feature-comparison page before committing.

### 2.3 Strengths

- **Fastest path to multi-provider anything.** If you had ten providers, LiteLLM would be the obvious default; nothing matches the translation matrix.
- **The tenancy feature set is complete and battle-tested.** Virtual keys + teams + budgets + a UI your finance and security people can use is a real, hard-to-replicate product. Most alternatives make you assemble this from parts.
- **Deploys air-gapped without drama.** It is a Docker image plus PostgreSQL plus (optionally) Redis; no phone-home requirement for the OSS core. Mirror the image into Harbor, point `config.yaml` at internal vLLM endpoints, done.
- **Enormous community and documentation surface.** Whatever you hit, someone hit first.

### 2.4 Weaknesses (documented, not vibes)

**Throughput ceiling of the Python data plane.** Every request traverses a Python async server, with Python's GIL (Global Interpreter Lock — the interpreter's one-thread-at-a-time execution lock) capping per-process parallelism; scaling means many workers and many replicas, each with its own connection pools to PostgreSQL/Redis. Concrete numbers:

- *LiteLLM's own published benchmark* (v1.79.1-stable, 4 instances of 4 vCPU/8 GB, Locust with 1,000 simulated users): median added overhead ~2 ms, p95 ~8 ms, p99 ~13 ms at ~1,170 RPS aggregate. Note that is a **tuned, four-instance** deployment; their earlier two-instance figures showed ~12 ms median.
- *Community production report* (GitHub issue #21046, February 12, 2026, v1.80.15): vLLM serving gpt-oss-120B delivered ~16 requests/s direct but ~9 requests/s through one LiteLLM proxy (4 vCPU/8 GB, 4 workers, pgbouncer-pooled Postgres) at 500-way concurrency — a 1.7–4× throughput reduction depending on load, persisting after disabling spend logs and applying the documented production settings.
- *The vendor's own verdict:* in July 2026 LiteLLM announced a **Rust rewrite of the gateway** (early beta, July 22, 2026), publishing benchmarks that frame the Python proxy at ~257.7 ms p99 added overhead under their stress test versus ~0.7 ms for Rust, and ~330 MB versus ~22 MB memory. Treat the exact numbers as vendor-published marketing measured against a mock upstream with logging/spend-tracking/persistence disabled — but the strategic message is unambiguous: LiteLLM knows the Python data plane is the bottleneck. The Rust gateway is **not feature-complete** as of August 2026 (streaming and the full feature surface "still landing"), so you cannot deploy it as a LiteLLM replacement yet.

**Correctness-under-load complaints.** Community posts through 2025–2026 (including a widely-shared "avoid LiteLLM" writeup and a "why teams are migrating away" survey on dev.to) allege TPM/RPM limiter confusion, token counts diverging from provider billing, and budget enforcement bugs at scale. These are anecdotal and partially version-specific — LiteLLM fixes fast — but the *pattern* (a huge, weekly-changing codebase where money-adjacent features regress) is consistent enough to plan around: pin versions, upgrade deliberately, and re-run your gate tests after every bump. In an air gap you will pin anyway; the cost is that you inherit whichever bugs your pinned version has, with no hotfix-by-pip.

**Breaking-change velocity.** Weekly releases and a version scheme that only recently standardized (their own blog announced "cleaner release versions" in 2026). For a bundle-transfer enclave, every LiteLLM upgrade is a cross-gap event with re-testing; a slower-moving component would cost less to own.

**Config sprawl and a stateful footprint.** PostgreSQL is mandatory for the features you would deploy it for; Redis appears once you want distributed rate-limit state or caching. That is two stateful services to back up, monitor, and include in DR drills — for a Layer B component. Compare: nginx's footprint for the same TLS+auth front-door job is one process and one config file.

### 2.5 When LiteLLM is the right tool vs the wrong tool

| Situation | LiteLLM? | Why |
|---|---|---|
| Many heterogeneous cloud providers | Yes | Translation matrix is unmatched |
| Per-team budgets/keys needed today, small team | Yes | Complete tenancy product with UI |
| Single homogeneous vLLM fleet, high RPS | No | Translation is dead weight; Python ceiling costs GPUs' worth of goodput |
| Stateful Responses API for self-hosted agents | No | Not a state store; use agentic-api |
| Kubernetes fleet with cache-aware routing needs | No (as router) | Layer C belongs to llm-d/GAIE |

### 2.6 The air-gap accounting

Line up LiteLLM's feature list against your environment and mark what survives:

- 100+ provider translation → **dead weight** (every backend is vLLM speaking OpenAI already).
- Cost tracking against provider billing → **mostly moot** (no per-token invoices in an enclave; "spend" becomes synthetic internal accounting — still useful for chargeback, but it is your own made-up prices).
- Fallback across providers → **shrinks** to fallback across your own replicas, which Layer C does better with engine signals.
- Response caching (Redis, exact-match) → **marginal**; vLLM's prefix caching gives the big win, and exact-match response caching rarely hits for agent traffic where every turn differs.
- Virtual keys, teams, budgets, rate limits, admin UI → **survives intact.** This is the entire remaining case for LiteLLM here.
- Guardrail hooks, audit logging → **survives**, partially enterprise-gated.

So the honest question is not "is LiteLLM good" (it is) but: **is per-team key/budget management with a UI worth a Python data-plane hop, a PostgreSQL dependency, and a fast-moving component in your bundle pipeline?** If you have multiple internal teams sharing the fleet and need that accounting *now*, plausibly yes — deployed as a thin layer doing nothing else. If you have one or two consuming teams, no: static keys at an nginx/Envoy front door cover it, and you re-evaluate when tenancy actually hurts.

---

## 3. The field

Each candidate below is scored on the axes that matter in your environment: deployment model, performance, license, air-gap fit, and Responses API support. Versions are as of August 2026.

### 3.1 vllm-project/agentic-api — the stateful agentic gateway (Layer E)

- **What:** a Rust (1.85+, Axum-based) gateway in front of vLLM that owns the stateful agent surface: `POST /v1/responses` with full state and streaming, WebSocket transport, `/v1/conversations`, and the Anthropic `/v1/messages` protocol (in progress as of August 2026) for Claude-Code-style clients. Apache-2.0.
- **Launch modes:** bundled (`vllm serve <model> --agentic-api --gateway-port 9000`) or standalone against an existing fleet (`agentic-api --llm-api-base http://vllm-pool:8000`).
- **State:** SQLite response store today. The SOP's guidance to move to a Postgres-class backend at fleet scale is a forward-looking expectation — the README documents **SQLite only** right now, so at fleet scale plan for either careful SQLite ops (single-writer, backed-up volume) or gateway sharding per model pool until a heavier backend lands.
- **Tool-ownership model:** each tool is *gateway-owned* (executed server-side — in your enclave, only internal tools; the built-in web-search integration activates via environment variables like `YOU_API_KEY` and must stay unset), *client-owned* (returned to the client to execute), or *provider-owned* (passed to vLLM).
- **Air-gap fit:** excellent — one small container, no external dependencies.
- **Maturity caveat:** this is a young project (double-digit GitHub stars, Messages API still landing). It is also the *only* purpose-built self-hosted Responses gateway for vLLM, and it is where the vLLM project itself is investing. Pin versions, test hard, and keep the passthrough `/v1/chat/completions` path as the fallback during migration.
- **Verdict-relevant:** nothing else in this list replaces it. Every other product is Layer B/C/D; agentic-api is the answer to a different question that your roadmap (SOP §6) already committed to.

### 3.2 Envoy AI Gateway — the high-performance API gateway (Layer B)

- **What:** an open-source (Apache-2.0) AI gateway built on Envoy Proxy and Envoy Gateway, under the Envoy project umbrella (CNCF ecosystem). **v1.0.0 GA shipped June 23, 2026**, declaring its Kubernetes CRDs (AIGatewayRoute, AIServiceBackend, BackendSecurityPolicy, GatewayConfig, MCPRoute) stable at v1beta1 with a semantic-versioning guarantee.
- **Signature features:** **token-based rate limiting** — it extracts real token usage from LLM responses into Envoy dynamic metadata and rate-limits/budgets on tokens rather than request counts, with separate cost attribution for input, output, cached, and reasoning tokens; quota-aware routing (QuotaPolicy); provider translation for 16 providers; an MCP (Model Context Protocol — the standard for agent-to-tool connections) gateway with OAuth; multi-tenant routing.
- **Data plane:** Envoy (C++), the same proxy that fronts a large share of the internet. Per-hop overhead is sub-millisecond-class; the AI-specific logic runs in Go external processors. This is the performance answer to LiteLLM's Python ceiling.
- **Deployment:** Kubernetes-native (its control plane is an operator over Envoy Gateway), **plus** a standalone `aigw run` CLI mode with an official container image and a reference Docker Compose file — usable for the Compose tier, though the K8s deployment is the paved road.
- **GAIE/InferencePool support** landed in v0.4.0 (November 2025), so it integrates directly with the Layer C routing you will run under llm-d.
- **What it lacks:** the LiteLLM-style self-serve tenancy product. Keys/limits are Kubernetes resources managed by platform engineers, not a web UI with `/key/generate` for team leads. Budget UX is your Grafana dashboard over its metrics.
- **Air-gap fit:** good — standard OCI images, no SaaS tether. You mirror a handful of images (controller, Envoy, ext-procs).

### 3.3 Kong AI Gateway (Layer B)

- **What:** AI plugins on the Kong API gateway (Lua/OpenResty on nginx). Mature general-purpose gateway; AI features arrived from v3.8 (semantic caching) through 3.14 (April 2026: "Agent Gateway" governing LLM, MCP, and A2A — agent-to-agent — traffic).
- **The catch:** the open-source core includes only the basic `ai-proxy`. **`ai-proxy-advanced` (multi-provider load balancing, semantic routing), semantic caching, prompt guard, PII sanitization, and LLM analytics are Enterprise/Konnect-gated**, and Konnect is a SaaS control plane — not applicable in an enclave. Air-gapped Kong Enterprise exists but is a commercial procurement.
- **Fit here:** weak. You would pay enterprise pricing for cloud-provider-oriented features while still needing agentic-api and llm-d anyway. Choose Kong only if it is *already* your organization's standard API gateway and the platform team insists on one vendor.

### 3.4 Apache APISIX + AI plugins (Layer B)

- **What:** Apache-2.0 API gateway (OpenResty/nginx + etcd) with genuinely open-source AI plugins: `ai-proxy` and `ai-proxy-multi` (provider translation, load balancing, retries, fallbacks, health checks), token-based rate limiting, prompt policy, and token-usage logging.
- **Notable:** unlike Kong, the AI capabilities are in the OSS tier. Performance is nginx-class. Air-gap deployable (mind the etcd control-plane dependency).
- **Fit here:** a respectable "thin front door with some AI awareness" option for teams already fluent in APISIX. For a vLLM-only fleet its provider translation is as much dead weight as LiteLLM's; you would use it for TLS/auth/rate-limits — a job plain nginx also does with less machinery.

### 3.5 Portkey gateway (Layer B)

- **What:** an MIT-licensed gateway (TypeScript/Node data plane) with fallbacks, retries, load balancing, and basic guardrails in the OSS core. The catch: **the observability control plane — logs, costs, traces, dashboards — plus semantic caching, prompt management, and RBAC live in the hosted SaaS** (with an enterprise on-prem option reported as of April 2026). Reports circulated in 2026 of an acquisition by Palo Alto Networks (unverified from primary sources at time of writing — treat as rumor and re-check before depending on the project's trajectory).
- **Fit here:** the OSS core alone is a competent but unremarkable proxy once the SaaS features are subtracted; in an air gap you get less than LiteLLM OSS gives you, on a smaller community. Pass.

### 3.6 Higress (Layer B)

- **What:** Alibaba's open-source (Apache-2.0) "AI-native" gateway built on Istio + Envoy with WebAssembly (WASM) plugins (Go/Rust/JS): provider translation, token-based rate limiting, response caching, MCP server hosting, AI observability. Deploys on Kubernetes or standalone Docker. Listed as a CNCF Sandbox project in its README (as of mid-2026).
- **Fit here:** technically credible (Envoy data plane, real token-rate-limiting) and fully self-hostable. Its center of gravity — community, docs, provider integrations — is the China cloud ecosystem, and for a US-style enclave Envoy AI Gateway covers the same ground with a more direct GAIE/llm-d story. Keep on the watch list; do not build on it here.

### 3.7 vLLM Semantic Router (Layer D)

- **What:** vllm-project/semantic-router (Apache-2.0), v0.3 "Themis" released June 5, 2026. A Rust (Candle-based) Envoy external processor that classifies each request with ModernBERT-class models — intent, complexity, jailbreak probability — and routes to the best model tier, toggles reasoning modes, and provides semantic caching. It is a "Mixture-of-Models" router: cheap model for easy prompts, big model for hard ones.
- **Air-gap fit:** fine — the classifier models are local files you bundle like any other weights.
- **Fit here:** not yet. It earns its place when you serve **multiple model tiers** behind one alias and want automatic cost/quality arbitrage. Your SOP's alias scheme (`agent-default`, `agent-long-context`) is manual model selection, which is the right starting point. Revisit when GPU contention makes "small model for small jobs" an SLO strategy.

### 3.8 Plain nginx / HAProxy + your PKI (Layer B, minimal)

- **What:** the null hypothesis. nginx or HAProxy terminates TLS with certs from your internal CA, enforces static bearer tokens or mTLS, applies coarse rate limits, and proxies to agentic-api/vLLM. Consistent hashing on a session header gives crude session affinity for prefix-cache friendliness on the Compose tier.
- **Streaming note:** you must disable response buffering for SSE (`proxy_buffering off;` in nginx, or per-location via the `X-Accel-Buffering: no` response header) or you will destroy TTFB (time to first byte) for streamed tokens.
- **What you give up:** self-serve keys, budgets, per-token accounting, admin UI, LLM-aware anything.
- **Fit here:** strong as the Tier 2 front door when tenancy is simple. It is boring, sub-millisecond, air-gap-native, and everyone on the team already knows it.

### 3.9 llm-d / Gateway API Inference Extension (Layer C — runs under all of the above)

- **What:** GAIE is a kubernetes-sigs project (introduced June 2025; **InferencePool CRD graduated to v1/GA**, with ecosystem conformance following through Istio 1.29 in February 2026). An InferencePool wraps a set of model-server pods; its **Endpoint Picker (EPP)** watches engine metrics — KV-cache utilization, queue depth, LoRA adapters, prefix-cache state — and picks the best replica per request, including prefix-cache-aware routing that estimates which replica already holds a session's prefix. **llm-d** (v0.8 as of mid-2026) packages this as its inference scheduler plus "well-lit paths" for prefix-cache-aware routing, tiered KV caching, P/D disaggregation, and wide expert parallelism.
- **The point for this document:** this layer is where agent latency is won or lost, and it is orthogonal to the gateway choice. Envoy AI Gateway, Kong, and LiteLLM can all sit above an InferencePool. When you reach the Kubernetes tier, GAIE/llm-d routing is non-negotiable per the SOP; the only question is which Layer B gateway fronts it — and Envoy AI Gateway is the one designed alongside it.

### 3.10 Not applicable: the SaaS tier

OpenRouter, Portkey Cloud, Kong Konnect, Vercel AI Gateway, Cloudflare AI Gateway, and every "AI gateway as a service" require internet egress and route to cloud providers. In a disconnected enclave they are not options, and comparison articles ranking them are noise for your decision. (Related: newer "agent gateway" projects such as agentgateway.dev exist in the Envoy/kgateway orbit and integrate with the Semantic Router; treat them as watch-list items, not candidates, as of August 2026.)

---

## 4. Comparison matrix

Ratings are for **your environment specifically** (air-gapped, vLLM-only, agentic). "Ops weight" counts stateful dependencies, upgrade cadence, and Kubernetes coupling. Tables are split to stay print-friendly.

### 4.1 Deployability and performance

| Product | Air-gap fit | Added latency class | Ops weight |
|---|---|---|---|
| LiteLLM proxy | Good (image + Postgres) | ms-class; p99 grows under load (Python) | Medium-high: Postgres, Redis, weekly releases |
| agentic-api | Excellent (one container) | Sub-ms-class (Rust); unpublished — measure | Low: SQLite file, young project churn |
| Envoy AI Gateway | Good (few images; K8s paved road) | Sub-ms proxy + ext-proc overhead | Medium: K8s operator (or standalone CLI) |
| Kong AI (OSS) | OK, but AI features enterprise/SaaS | Sub-ms (nginx-class) | Medium; enterprise procurement for AI |
| APISIX + AI plugins | Good (mind etcd) | Sub-ms (nginx-class) | Medium: etcd control plane |
| Portkey (OSS core) | OK; control plane is SaaS | ms-class (Node) | Low-medium |
| Higress | Good (K8s or Docker) | Sub-ms (Envoy) | Medium: Istio-based stack |
| nginx / HAProxy | Excellent | Sub-ms | Very low |

### 4.2 Feature coverage for this use case

| Product | Keys/budgets/multi-tenancy | Responses API | Streaming fidelity |
|---|---|---|---|
| LiteLLM proxy | Best-in-class, with UI | Bridged + session affinity; not a vLLM state store | Good SSE; verify per version |
| agentic-api | None (bring your own front door) | Native, stateful — the reference | SSE + WebSocket, purpose-built |
| Envoy AI Gateway | Token-based limits/quotas via CRDs; no self-serve UI | Passthrough to backend | Excellent (Envoy) |
| Kong AI (OSS) | Kong auth/limits; AI analytics gated | Passthrough | Excellent (nginx) |
| APISIX + AI plugins | Gateway auth + token limits; no LLM tenancy UI | Passthrough | Excellent (nginx) |
| Portkey (OSS core) | Minimal without SaaS | Partial translation | Good |
| Higress | Token limits, provider keys | Passthrough | Excellent (Envoy) |
| nginx / HAProxy | Static tokens/mTLS only | Passthrough | Excellent if buffering disabled |

### 4.3 Observability and license

| Product | Observability | License |
|---|---|---|
| LiteLLM proxy | Prometheus `/metrics` (OSS as of Aug 2026 docs), spend dashboards | MIT core + commercial `enterprise/` dir |
| agentic-api | Logs; metrics surface still maturing — front with a proxy that measures | Apache-2.0 |
| Envoy AI Gateway | First-class: Envoy stats, token metrics, OpenTelemetry | Apache-2.0 |
| Kong AI | Strong, but LLM analytics enterprise-gated | Apache-2.0 core + commercial |
| APISIX | Prometheus, token usage in access logs | Apache-2.0 |
| Portkey OSS | Thin without SaaS control plane | MIT (gateway) |
| Higress | Envoy stats + AI plugins | Apache-2.0 |
| nginx / HAProxy | Standard exporters; no token awareness | OSS |

*(GAIE/llm-d is deliberately absent from these tables: it is Layer C and stacks under any row above.)*

---

## 5. Verdict: what to actually run

### 5.1 The direct answer

**Is LiteLLM the best option? No — not for this environment as the primary gateway.** Three reasons, in order of weight:

1. **Its flagship value is dead weight here.** Behind an air gap with a homogeneous vLLM fleet, provider translation — the reason LiteLLM exists — translates nothing. You would be running the largest, fastest-moving codebase in the category to use its smallest subsystem.
2. **It cannot carry the layer your roadmap actually needs.** The SOP's endgame is stateful `/v1/responses` with server-side tool ownership via agentic-api. LiteLLM does not replace that; at best it sits in front of it, adding a Python hop to every agent turn on a path where you own the entire latency budget.
3. **The performance ceiling is real and admitted.** Self-published overhead of ~2 ms median/~13 ms p99 requires a tuned multi-instance deployment; community reports show single instances halving vLLM throughput at high concurrency; and LiteLLM's own July 2026 Rust rewrite is the vendor conceding the point. On B200/B300-class hardware where each node's goodput is precious, spending it on gateway CPU overhead is the wrong trade. (When the Rust gateway reaches feature parity, this calculus is worth revisiting — as of August 2026 it is an early beta.)

**The one scenario where LiteLLM earns a place:** you need per-team virtual keys, budget accounting, and a self-serve admin UI *now*, across several internal teams, and nobody wants to build even a thin key service. Then run LiteLLM as a **narrow tenancy layer** — keys, budgets, UI, nothing else: caching off, translation unused, minimal callbacks — in front of agentic-api, with the explicit plan that it is removable. Do not let it become the routing brain.

### 5.2 Compose tier (SOP Tier 2) — the stack for now

```
clients ──► nginx (TLS from internal CA, static bearer keys, rate limits,
            proxy_buffering off)
        ──► agentic-api  (/v1/responses, /v1/messages, chat-completions
            passthrough; SQLite state on a backed-up volume)
        ──► vLLM containers (per-model pools, SOP §2 profiles)

[Only if multi-team budgets are needed today]
clients ──► LiteLLM proxy (virtual keys, teams, budgets, UI; + Postgres)
        ──► agentic-api ──► vLLM
```

Reasoning: at one-to-a-few nodes, your tenancy is probably a handful of internal services. Static keys in nginx (or Envoy via `aigw run` standalone if you want token-aware limits early) cost near-zero latency and near-zero ops. agentic-api is the component doing the irreplaceable work. Adding LiteLLM here is a *business* decision (chargeback and self-serve keys), not a technical one — make it consciously and keep it thin.

### 5.3 Kubernetes tier (SOP Tier 3) — the stack for later

```
clients ──► Envoy AI Gateway (v1.0+: authn, token-based rate limits,
            quotas, MCP gateway, OpenTelemetry)
        ──► agentic-api (Responses/Messages state; per-model-pool instances)
        ──► GAIE InferencePool / llm-d inference scheduler
            (prefix-cache-aware + load-aware Endpoint Picker)
        ──► vLLM replicas
```

Reasoning: Envoy AI Gateway is Apache-2.0, GA since June 2026, built on the same Envoy/Gateway-API substrate as GAIE/llm-d, and rate-limits on *tokens* — the unit that actually measures LLM load. Its keys/limits are CRDs in Git, which matches enclave change control better than UI-driven state. llm-d's scheduler under it preserves the prefix-cache hit rates your agent TTFT depends on. agentic-api remains the state layer in both tiers, which is exactly what makes the Tier 2 → Tier 3 migration boring.

### 5.4 Switching costs (why this decision is low-regret)

Every layer speaks OpenAI-compatible HTTP, so moving between gateways does not touch client code — clients call aliases (SOP §5) against a stable base URL. The sticky parts are:

- **API keys:** migrating LiteLLM virtual keys (Postgres rows) to Envoy CRDs or nginx tokens means reissuing keys. Mitigation: distribute keys to services via your secrets manager from day one, so rotation is a config push.
- **Dashboards and alerts:** metric names differ per gateway. Keep SLO definitions (TTFT p99, ITL p99, parse-failure rate) gateway-agnostic in the config repo; re-map exporters when you switch.
- **Response state:** agentic-api's SQLite/state DB carries across tiers untouched — that is deliberate; the layer you cannot cheaply switch is the one we standardized once.

---

## 6. Sidebar: how to benchmark a gateway honestly

Most published gateway benchmarks (including vendors' own) measure proxy forwarding against a **mock upstream** — a stub that answers instantly. That deletes the dominant real-world variable: thousands of concurrent, slow, streaming connections held open for many seconds each, which is precisely what an LLM workload is and what breaks Python/Node data planes first. Method:

1. **Baseline direct.** Run `guidellm` (or `genai-perf`) straight at a vLLM replica at your production concurrency and context-length mix. Record TTFT p50/p99, ITL (inter-token latency) p99, and max RPS at SLO.
2. **Same test through the gateway.** Identical load, identical replica, one added hop. The gateway's cost is the *delta*, reported as: added p50/p99 latency at target RPS, and change in max-RPS-at-SLO.
3. **Streaming fidelity separately.** For SSE: measure time-to-first-byte through vs around the gateway (buffering bugs show up here), inter-chunk jitter, and behavior at 1k+ concurrent open streams. A gateway that adds 2 ms per request but buffers SSE flushes has failed the agent use case.
4. **Load the control plane too.** Run key-lookup/budget-check paths hot (thousands of distinct keys), with the real Postgres/Redis attached — LiteLLM's cheerful numbers were measured with persistence features off.
5. **Find the knee.** Ramp RPS until gateway CPU, not GPU, is the bottleneck; that RPS is your scale-out trigger. Record gateway CPU/memory per 100 RPS as a capacity planning unit alongside the SOP §7 "max concurrency at SLO" number.
6. **Re-run on every gateway version bump** — it is a bundle-crossing event in the enclave, so fold this into the same canary gate as engine upgrades (SOP §8, runbook 1).

> **Common pitfall:** benchmarking with short prompts and `stream: false`. Agentic traffic is long-prefix, streaming, high-concurrency, and bursty. A gateway ranking produced under curl-style load can invert under real agent load.

---

## Study questions

1. **Why is "LiteLLM vs llm-d" a category error?**
   Answer: They live at different layers — LiteLLM is a Layer B API gateway (auth, keys, budgets, provider abstraction); llm-d/GAIE is a Layer C inference load balancer choosing which replica serves each request using engine signals. They stack; they do not compete.

2. **What is LiteLLM's flagship feature, and why does it lose value in this enclave?**
   Answer: Translating 100+ heterogeneous provider APIs into one OpenAI-shaped interface. In an air-gapped fleet every backend is vLLM, which already speaks the OpenAI dialect — there is nothing to translate.

3. **What infrastructure does LiteLLM's virtual-key feature require?**
   Answer: A PostgreSQL database (`DATABASE_URL`) plus a master key; keys are minted via `POST /key/generate`. Distributed rate-limit state and caching additionally want Redis.

4. **Cite two concrete pieces of evidence for LiteLLM's Python throughput ceiling.**
   Answer: GitHub issue #21046 (Feb 2026, v1.80.15): vLLM throughput fell from ~16 to ~9 req/s through one proxy instance at 500-way concurrency; and LiteLLM's own July 2026 launch of a Rust gateway rewrite, whose announcement benchmarks frame the Python proxy at hundreds of ms p99 overhead under stress.

5. **What does the `agentic-api` gateway do that no Layer B gateway does?**
   Answer: It owns Responses API state — persisting responses, hydrating `previous_response_id`, WebSocket/SSE streaming, and a tool-ownership model (gateway-owned / client-owned / provider-owned) — plus the Anthropic `/v1/messages` protocol, all designed for self-hosted vLLM.

6. **What makes Envoy AI Gateway's rate limiting "AI-aware"?**
   Answer: It extracts actual token usage from LLM responses into Envoy dynamic metadata and enforces limits/budgets on tokens (with separate input/output/cached/reasoning costing) rather than raw request counts — requests vary by orders of magnitude in cost, so token-based limits track real load.

7. **Why does the routing layer under the gateway matter so much for agentic workloads?**
   Answer: Agents resend a nearly identical prefix each turn; prefix-cache hits skip that prefill. Only prefix-cache-aware routing (llm-d scheduler / GAIE Endpoint Picker) reliably sends a session to the replica holding its cache — round-robin discards most of the benefit and inflates TTFT.

8. **What is the trap in most published gateway latency benchmarks?**
   Answer: They measure forwarding against a mock upstream that responds instantly, hiding how the gateway behaves with thousands of slow, long-lived streaming connections — the actual LLM workload shape. Benchmark through to a real vLLM backend and measure the delta.

9. **When would LiteLLM still deserve a slot in this architecture?**
   Answer: When several internal teams need self-serve virtual keys, budget accounting, and an admin UI immediately — run it as a thin tenancy layer in front of agentic-api, with caching/translation unused and a plan to remove it.

10. **Which candidate gateways gate their AI features behind SaaS or enterprise licensing, and why does that matter here?**
    Answer: Kong (ai-proxy-advanced, semantic caching, LLM analytics are Enterprise/Konnect) and Portkey (observability control plane, semantic caching are hosted SaaS). In a disconnected enclave, SaaS control planes are unusable and enterprise licensing is a procurement dependency — so their effective OSS feature set is much smaller than marketing suggests.

11. **What is the vLLM Semantic Router and when should this team adopt it?**
    Answer: A Rust Envoy external processor (vllm-project, Apache-2.0, v0.3 June 2026) that classifies prompts with ModernBERT-class models and routes easy queries to small models and hard ones to big/reasoning models. Adopt only after serving multiple model tiers behind one alias, when automatic cost/quality arbitrage beats manual alias selection.

12. **What actually carries over when you switch Layer B gateways, and what does not?**
    Answer: Client code carries over (everything is OpenAI-compatible HTTP against stable aliases) and agentic-api's state DB carries over. API keys must be reissued and dashboards/alerts re-mapped — so distribute keys via a secrets manager and keep SLO definitions gateway-agnostic from day one.

---

## Sources

Primary sources fetched and read directly (August 2026):

- vllm-project/agentic-api README: https://github.com/vllm-project/agentic-api
- LiteLLM Prometheus metrics docs: https://docs.litellm.ai/docs/proxy/prometheus
- LiteLLM benchmarks docs: https://docs.litellm.ai/docs/benchmarks
- LiteLLM virtual keys docs: https://docs.litellm.ai/docs/proxy/virtual_keys
- LiteLLM Responses API docs: https://docs.litellm.ai/docs/response_api
- LiteLLM Rust gateway benchmark announcement (July 22, 2026): https://docs.litellm.ai/blog/rust-ai-gateway-benchmarks
- LiteLLM production overhead issue #21046: https://github.com/BerriAI/litellm/issues/21046
- LiteLLM license-boundary issue #34241: https://github.com/BerriAI/litellm/issues/34241
- Envoy AI Gateway release notes (v1.0.0, June 23, 2026): https://aigateway.envoyproxy.io/release-notes/
- Envoy AI Gateway standalone `aigw run`: https://aigateway.envoyproxy.io/docs/cli/aigwrun/
- llm-d project site (v0.8, well-lit paths): https://llm-d.ai/
- Gateway API Inference Extension (InferencePool v1): https://gateway-api-inference-extension.sigs.k8s.io/ and https://github.com/kubernetes-sigs/gateway-api-inference-extension
- Kubernetes blog — Introducing Gateway API Inference Extension (June 2025): https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/
- vLLM Semantic Router: https://github.com/vllm-project/semantic-router and https://blog.vllm.ai/2025/09/11/semantic-router.html
- Higress README: https://github.com/alibaba/higress
- Apache APISIX ai-proxy plugin docs: https://apisix.apache.org/docs/apisix/plugins/ai-proxy/ and https://apisix.apache.org/ai-gateway/
- Kong AI Gateway docs and ai-proxy-advanced plugin: https://developer.konghq.com/ai-gateway/ and https://developer.konghq.com/plugins/ai-proxy-advanced/
- Kong AI Gateway 3.8 announcement (semantic caching): https://konghq.com/blog/product-releases/ai-gateway-3-8
- Portkey feature comparison (OSS vs hosted): https://portkey.ai/docs/product/product-feature-comparison
- DeepInspect — AI gateway latency benchmarks and the mock-upstream problem: https://www.deepinspect.ai/blog/ai-gateway-latency-benchmarks
- Secondary/community context (treated as anecdotal): https://dev.to/debmckinney/why-production-teams-are-migrating-away-from-litellm-and-where-theyre-going-3dc2 and TrueFoundry LiteLLM pricing/enterprise writeups: https://www.truefoundry.com/blog/litellm-pricing-guide
