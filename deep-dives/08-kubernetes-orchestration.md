# Kubernetes for GPU Inference: GPU Operator, llm-d, KServe, Kueue, and Dynamo

**Deep-dive 08 · Written August 2026 · Companion to the [deployment SOP](../SOP-production-ai-model-deployment.md) §3 and the [primer](../PRIMER.md) §6.**

This document teaches the Kubernetes (K8s) layer of the stack: what Kubernetes actually is when your workload is "a vLLM process that owns 8 GPUs for 20 minutes of model loading and then must never be interrupted," what the NVIDIA GPU Operator installs on every node, how LLM-aware routing works (Gateway API Inference Extension), and the four orchestration options above vLLM — llm-d, KServe, NVIDIA Dynamo, and Ray Serve — plus the plain-K8s baseline you should be able to write by hand. Everything is framed for our environment: on-prem, air-gapped, B200/B300 nodes, agentic workloads.

Versions move fast in this space. Anything version-stamped below is "as of August 2026" unless noted, and fast-moving claims are flagged.

---

## Key takeaways

- Kubernetes is a *declarative fleet manager*: you describe desired state (3 replicas of this pod, with 8 GPUs each) and controllers continuously reconcile reality toward it. For LLM serving its value is placement, restarts, rollouts, and smart routing — not magic performance.
- The NVIDIA GPU Operator is one Helm chart that installs the entire GPU enablement stack per node (driver, container toolkit, device plugin, DCGM exporter, MIG manager, feature discovery). Without it, `nvidia.com/gpu` does not exist as a schedulable resource.
- GPUs are requested as extended resources (`resources.limits: nvidia.com/gpu: 8`) — integers only, no fractions. Dynamic Resource Allocation (DRA) is the successor API (core GA since Kubernetes v1.34); as of mid-2026 the device plugin is still the default production path, with the NVIDIA DRA driver integrated into GPU Operator and worth planning for.
- LLM traffic breaks classic load balancing: requests have wildly unequal cost, and prefix-cache locality means *where* a request lands changes its latency 5–10×. The Gateway API Inference Extension (GAIE) standardizes LLM-aware routing via an Endpoint Picker (EPP) and the `InferencePool` / `InferenceObjective` CRDs.
- llm-d (CNCF Sandbox, March 2026) is the Kubernetes-native orchestration layer for vLLM: prefix-cache-aware scheduling, prefill/decode disaggregation, and wide expert parallelism packaged as benchmarked "well-lit paths." It is the SOP's default Tier-3 answer.
- KServe's new `LLMInferenceService` CRD (KServe v0.16+) wraps llm-d patterns in a single manageable resource — attractive if you want a platform abstraction; skippable if you are comfortable with llm-d's charts directly.
- NVIDIA Dynamo (1.0 GA, March 2026) is the framework-agnostic alternative: a Rust-based distributed serving OS above vLLM/TensorRT-LLM/SGLang with an SLA-driven Planner, KV-aware routing, and KVBM cache tiering. Choose it for TensorRT-LLM, rack-scale NVL72 topologies, or NVIDIA-supported stacks; choose llm-d for standard-K8s-native vLLM fleets.
- Kueue adds what vanilla K8s lacks for shared GPU pools: team quotas, queueing, all-or-nothing (gang) admission, and preemption — the difference between "GPUs sit 30% utilized behind team silos" and a genuinely shared pool.
- Multi-node inference needs the *second* network: pod networking (CNI) carries API traffic fine, but NCCL/RDMA traffic needs hostNetwork or Multus+SR-IOV with RDMA device plugins — and the #1 failure mode is silent fallback to TCP that "works" at one-tenth the bandwidth.
- Air-gapped K8s is a standing cost, not a one-time install: control-plane image mirroring, quarterly upgrade bundles, and etcd care. RKE2's tarball air-gap flow and Zarf packaging are the mature patterns. This cost is exactly why the SOP says "don't pay for K8s before Tier 2 hurts."

---

## 1. Kubernetes crash course, scoped to model serving

### 1.1 The mental model

Kubernetes is a control loop machine. You submit YAML documents describing *desired state* to the **API server**; a set of **controllers** continuously compares desired state to actual state and takes actions to converge them; the **scheduler** decides which node each pod lands on; the **kubelet** (the agent on every node) actually starts containers via a container runtime (containerd); and **etcd** is the database where all of this state lives. That is the whole architecture: a database of intent, plus loops that make intent real.

The objects you need for serving, in dependency order:

| Object | What it is | Serving-specific note |
|---|---|---|
| **Pod** | One or more containers scheduled together on one node; the atom of scheduling | One vLLM engine = one pod. A TP=8 model is one pod requesting 8 GPUs, not 8 pods |
| **Deployment** | "Keep N identical pods running," with rolling update logic | Right shape for single-node model replicas |
| **StatefulSet / LeaderWorkerSet** | Pods with stable identity; LWS groups a leader pod with worker pods | Multi-node inference (one model spanning nodes) uses LeaderWorkerSet (LWS), a SIG-sponsored API |
| **Service** | A stable virtual IP + DNS name load-balancing across matching pods | Fine for dumb routing; LLM-aware routing replaces this (see §5) |
| **PVC** | PersistentVolumeClaim — a request for storage that pods mount | How pods see the model store (NFS-backed, ReadOnlyMany) |

Plus the plumbing objects: **ConfigMap** and **Secret** carry configuration and credentials into pods; **Namespace** partitions the cluster (per-team, per-environment); **Ingress** is the legacy HTTP entry-point API, now superseded by the **Gateway API** (`Gateway` + `HTTPRoute` objects) — which matters here because LLM routing is built on Gateway API, not Ingress (§5).

### 1.2 CRDs and operators — the extension mechanism everything below uses

A **CRD (Custom Resource Definition)** teaches the API server a new object type. An **operator** is a controller that watches those custom objects and reconciles them, encoding operational knowledge in software ("when someone creates an `LLMInferenceService`, create these deployments, this routing config, these monitors"). Every project in this document — GPU Operator, llm-d, KServe, Kueue, Dynamo — is delivered as CRDs plus operators. When you evaluate any of them, the question is always the same: *what objects does it add, and what does its controller do that I would otherwise script by hand?*

### 1.3 Why Kubernetes at all (and why not yet)

The SOP §3.2 defines the triggers, restated here in K8s vocabulary:

1. **Multi-node serving** — wide expert parallelism (wide-EP) or prefill/decode (P/D) disaggregation needs coordinated placement of pod groups across nodes with RDMA-attached networking. Humans can do this once; they cannot do it during a 2 a.m. node failure.
2. **Model fleet with churn** — beyond ~5 models with frequent adds/retires, humans-as-scheduler produces placement drift and stranded GPU capacity.
3. **Shared pools with quotas** — two-plus teams sharing GPUs need admission control (Kueue), not a spreadsheet.
4. **Zero-downtime as a hard SLA** — rolling updates, canary weighting, and automated rollback as platform primitives.
5. **Smart routing at scale** — prefix-cache-aware routing across many replicas (llm-d) measurably beats round-robin for agent traffic; it only exists in the K8s ecosystem.

The counterweight, which is doubled in an air gap: Kubernetes is itself a distributed system you must feed. Control-plane images to mirror, three-releases-per-year upgrade cadence, etcd backups, CNI and operator version matrices. The SOP's stance stands: run the compose tier until a trigger fires, and treat the K8s migration as a planned project (§13 below gives the plan).

---

## 2. NVIDIA GPU Operator: making GPUs exist in Kubernetes

Kubernetes knows nothing about GPUs natively. The **NVIDIA GPU Operator** (Helm-installed, current release line 26.x as of mid-2026; 26.3.0 is a recent reference point) deploys and lifecycle-manages every piece of the GPU enablement stack as pods on each GPU node, so nodes become interchangeable and OS images stay generic.

What a default install runs, in dependency order (each component's init container waits for the previous one):

| Component | Job |
|---|---|
| **Node Feature Discovery (NFD)** | Labels nodes with hardware facts (PCI IDs, kernel, CPU features) so later components and your pods can target them |
| **Driver container** | Runs the NVIDIA kernel driver *as a container*, keeping the host OS clean; alternatively detect a host-preinstalled driver (`driver.enabled=false`) |
| **NVIDIA Container Toolkit** | Configures containerd so containers can see GPU devices and driver libraries |
| **Device plugin** | Advertises `nvidia.com/gpu` to the kubelet as a schedulable resource |
| **GPU Feature Discovery (GFD)** | Adds GPU-specific labels, e.g. `nvidia.com/gpu.product=NVIDIA-B200`, memory size, MIG capability |

Branching off that chain: the **DCGM exporter** (DCGM = Data Center GPU Manager, NVIDIA's health/telemetry stack) exposes per-GPU Prometheus metrics — this is where the SOP's GPU utilization and HBM headroom dashboards come from; the **MIG manager** applies Multi-Instance GPU partitioning profiles (§3); and a **validator** runs a CUDA smoke test on each node so a broken node fails loudly at admission, not at model load. Optional extras you may enable later: GPUDirect Storage (GDS), the DRA driver (§3.3), and vGPU components (irrelevant on-prem here).

**Air-gap notes for the operator itself:** every one of these images must be mirrored into Harbor and the chart's `values.yaml` pointed at your registry (the operator supports overriding registry/repository per component). The driver container is the sharp edge: it must match your exact kernel version. In a disconnected environment the two safe patterns are (a) precompiled driver containers matched to your frozen OS image, or (b) preinstall the driver in the node OS image and set `driver.enabled=false` — pattern (b) is common in air gaps because it removes the kernel-matching dance from the cluster entirely, at the cost of coupling driver upgrades to node reimaging.

### 2.1 Requesting GPUs in a pod spec

GPUs are **extended resources**: integer-only, requested in `limits` (requests are implied equal), never oversubscribed by the scheduler.

```yaml
spec:
  containers:
    - name: vllm
      resources:
        limits:
          nvidia.com/gpu: 8        # whole GPUs only; no 0.5
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-B200    # GFD label; pin pools by GPU type
```

Rules worth memorizing: you cannot request a fraction of a GPU through this API; a pod requesting 8 GPUs will only schedule on a node with 8 *free* GPUs (bin-packing whole nodes is the norm for TP=8 models); and GPUs are not shared between pods unless you explicitly configure time-slicing or MIG.

---

## 3. Sharing GPUs: time-slicing vs MIG, and where DRA fits

For our primary workload — one vLLM engine owning all 8 GPUs of a node — sharing is irrelevant. It matters for the *long tail*: embedding models, rerankers, small utility models, and dev/test, where dedicating a 192 GB B200 to a 2 GB model is absurd.

| | Time-slicing | MIG (Multi-Instance GPU) |
|---|---|---|
| Mechanism | Device plugin advertises N virtual replicas of each GPU; contexts share the GPU by scheduling turns | Hardware partitions one GPU into up to 7 isolated instances, each with dedicated memory slice and compute |
| Isolation | **None** — shared memory, one OOM or crash can take down neighbors | Hard isolation of memory and faults |
| Config | ConfigMap consumed by the device plugin (`replicas: 4`) | MIG manager applies profiles per node label; GPU Operator 26.3.0 added dynamic per-node default MIG config generation |
| Good for | Trusted bursty dev workloads | Multi-tenant small-model serving on shared nodes |
| Bad for | Anything production or latency-sensitive | Big-model serving (a partition is a smaller GPU; vLLM TP across MIG instances is not a thing) |

Practical guidance for our fleet: keep production inference nodes un-partitioned (whole GPUs to vLLM); if you need a "small models" pool, MIG one or two nodes rather than time-slicing, so a runaway embedding job cannot corrupt a neighbor.

### 3.3 Dynamic Resource Allocation (DRA) — the successor API, status as of Aug 2026

**DRA (Dynamic Resource Allocation)** replaces the crude integer counter with a real allocation API: workloads declare *claims* against *device classes* with attribute selectors ("one GPU with ≥180 GiB memory," "two GPUs on the same NVLink domain"), and the scheduler allocates concrete devices. Objects: `DeviceClass`, `ResourceClaim`, `ResourceClaimTemplate`.

Verified status: DRA's core graduated to **GA in Kubernetes v1.34** (kubernetes.io blog, Sept 2025); the feature gate is locked on in v1.35, and further DRA enhancements graduated in v1.36 (2026). On the NVIDIA side, the DRA driver first shipped **ComputeDomains** (the abstraction for secure multi-node NVLink on GB200/NVL72-class systems — directly relevant if you ever field GB300 racks), with official *GPU allocation* support arriving in the 25.8.0 driver release, integrated into the GPU Operator rather than a standalone chart.

What this means operationally in mid-2026: the device plugin path (`nvidia.com/gpu`) remains the boring default that every chart and guide in this document assumes; DRA is the direction of travel and becomes load-bearing the moment you need multi-node NVLink domains (ComputeDomains) or expressive GPU selection. Track it, don't build on it yet, and expect llm-d/KServe/Dynamo charts to adopt it beneath you. Flagged as fast-moving — re-verify quarterly.

---

## 4. Networking for multi-node inference

Single-node serving needs nothing special: the pod network (provided by a **CNI** — Container Network Interface — plugin like Cilium or Calico) carries HTTP in and out, and NVLink carries all GPU-to-GPU traffic inside the node. The moment one *model* spans nodes (wide-EP, P/D disaggregation, or PP across nodes), you need the second network: **RDMA** (Remote Direct Memory Access — NICs reading/writing memory across the fabric without CPU involvement) over InfiniBand or RoCEv2, ideally with **GPUDirect RDMA** so the NIC's DMA engine reads GPU HBM directly, bypassing system RAM entirely.

Kubernetes pod networking cannot carry this. Two patterns exist:

- **`hostNetwork: true`** — the pod uses the node's network namespace directly and sees the real InfiniBand/RoCE interfaces. Simple, performant, and what many wide-EP guides assume. Costs: no network isolation, port collisions between pods on one node, and it bypasses NetworkPolicy. Acceptable when inference nodes are single-tenant (ours are).
- **Multus + SR-IOV** — **Multus** is a meta-CNI that attaches *additional* network interfaces to a pod beyond the default; **SR-IOV** (Single Root I/O Virtualization) carves a physical NIC into virtual functions (VFs) that are passed into pods as first-class devices via the SR-IOV device plugin (requested like GPUs, e.g. `nvidia.com/ibnic: 1`). The pod keeps a normal isolated pod network for API traffic and gets a real RDMA-capable interface for NCCL/NVSHMEM traffic. The **NVIDIA Network Operator** automates the host stack (DOCA-OFED drivers, RDMA shared device plugin, SR-IOV operator wiring) and pairs with the GPU Operator.

Either way, the pod also needs `IPC_LOCK` capability (RDMA pins memory) and generous or unlimited memlock ulimits, and `/dev/infiniband/*` devices exposed (the RDMA device plugin handles this).

### 4.1 Gotchas (each has burned real deployments)

- **Silent TCP fallback.** NCCL (NVIDIA's collective communication library) will quietly fall back to TCP-over-pod-network if it can't use the IB devices — everything *works*, at roughly one-tenth the bandwidth. Always validate with `NCCL_DEBUG=INFO` and confirm `NET/IB` (not `NET/Socket`) in logs, and run `nccl-tests` all-reduce/all-to-all as part of node-pair acceptance, exactly like the SOP's fabric validation rule (§1).
- **GPU–NIC topology alignment.** GPUDirect RDMA performs when GPU and NIC share a PCIe switch/NUMA node. Enable the kubelet Topology Manager and make sure the device plugins report topology, or a "working" setup loses half its bandwidth to cross-socket hops.
- **RoCE specifics.** Lossless Ethernet (PFC/ECN) must be configured on the switches; wrong GID index or MTU mismatches produce mysterious hangs rather than clean errors.
- **Wide-EP raises the bar further.** DeepEP kernels use NVSHMEM with **IBGDA** (GPU-initiated RDMA), which is pickier than NCCL about the fabric; the SOP's rule — validate the fabric before you need multi-node — exists because this is a weeks-long debugging swamp if discovered late.
- **Order matters.** A working RDMA cluster is roughly twenty things configured correctly in sequence (firmware, OFED, VFs, device plugin, Multus attachment, NCCL env). Automate node acceptance; never hand-configure.

---

## 5. Gateway API Inference Extension (GAIE): LLM-aware routing as a standard

### 5.1 The problem

A classic Kubernetes Service load-balances by connection count or round-robin, which assumes requests are cheap and interchangeable. LLM requests are neither: one request may cost 500× another (context length, output length), replicas differ enormously in *effective* cost for a given request depending on whose KV cache they hold (a prefix-cache hit skips prefill entirely — the primer's #1 agent win), and a saturated replica should shed load long before TCP tells you anything. Round-robin across vLLM replicas throws away most of the prefix-caching benefit for agent traffic.

### 5.2 The design

The **Gateway API Inference Extension** (GAIE, a Kubernetes SIG project; v1.x line current, with releases through v1.3/v1.4 as of mid-2026) extends the standard Gateway API rather than inventing a new proxy. The pieces:

- **InferencePool** (CRD, `inference.networking.k8s.io/v1`) — the LLM-aware replacement for a Service: a set of model-server pods *plus a reference to an Endpoint Picker* responsible for choosing among them. A Gateway's `HTTPRoute` sends traffic to an InferencePool instead of a Service.
- **InferenceObjective** (CRD) — the per-model/per-workload policy object: maps a served model name to backends with criticality/priority. **Naming caution:** this was called `InferenceModel` before the v1.0 GA release; the API group also graduated from `inference.networking.x-k8s.io` to `inference.networking.k8s.io`. Older blog posts and charts still say `InferenceModel` — check your CRD versions before copying YAML.
- **Endpoint Picker (EPP)** — a separate service the proxy consults per request via Envoy's external-processing (ext-proc) callout: "here is a request for model X; which pod should get it?" The EPP tracks live metrics from each vLLM replica — queue depth, KV-cache utilization, prefix-cache state, loaded LoRA adapters — and returns the best endpoint. The reference EPP is intentionally basic; production deployments use a real one, and **llm-d's inference scheduler is the flagship production EPP**.

```
 client ──► Gateway (Envoy / kgateway / Istio)
               │  ext-proc callout: "which endpoint?"
               ▼
             EPP  ── scrapes ──►  vLLM pod A  (queue: 2, prefix hit likely)
               │                  vLLM pod B  (queue: 9)
               │                  vLLM pod C  (queue: 3)
               └─ "send it to A"
```

Because it rides the standard Gateway API, any conformant gateway can be the data plane: **kgateway** (CNCF's Envoy-based gateway, formerly Gloo), **Istio**, **Envoy Gateway** / Envoy AI Gateway, GKE Gateway in cloud contexts, and NGINX Gateway Fabric all ship GAIE support. This is the layer where the SOP's session-affinity/prefix-aware routing requirement becomes an actual API rather than a vendor feature — and it is the foundation llm-d builds on.

---

## 6. llm-d deep dive: Kubernetes-native vLLM orchestration

### 6.1 What it is and where it came from

**llm-d** is a distributed inference serving stack for Kubernetes built specifically around vLLM (with SGLang also supported as a model server). It was launched in May 2025 by **Red Hat, Google Cloud, IBM Research, CoreWeave, and NVIDIA**, with additional backers including AMD, Intel, Hugging Face, Mistral AI, Lambda, and UC Berkeley, and was accepted as a **CNCF Sandbox project on March 24, 2026**. Release cadence has been steady: v0.7 shipped May 2026, and the project docs reference **v0.8 as the latest release** at the time of writing (Aug 2026). Sandbox status signals community governance and momentum, not production maturity — treat llm-d as early-production/staging-grade and gate it with your own evals, same as any model.

The pitch in one sentence: vLLM makes one replica fast; llm-d makes a *fleet* of replicas behave like one efficient system, using standard Kubernetes machinery (Gateway API, GAIE, Helm/kustomize) rather than a parallel universe of custom infra.

### 6.2 Architecture

Three core elements, then optional layers:

1. **Router = proxy + EPP.** An L7 proxy (typically Envoy via kgateway or Istio) fronts the fleet; the **llm-d inference scheduler** is the EPP (exactly the GAIE mechanism from §5). Its scoring is multi-objective: prefix-cache hit likelihood, in-flight request count, queue depth — and as of the 2026 releases, **predicted-latency-based scheduling** (an XGBoost model predicting each candidate pod's latency for this request) has graduated to GA within the project.
2. **InferencePool** — the GAIE object grouping model-server pods, with optional *variants* distinguishing roles (e.g., prefill pods vs decode pods).
3. **Model servers** — vLLM pods deployed from the project's charts (the `modelservice` Helm chart is the deployment unit for a model + its pools).

Optional layers, each a separately adoptable component (they live as separate repos in the `llm-d` GitHub org — inference scheduler, KV-cache manager, routing sidecar):

- **KV-cache management** — a global KV index fed by **KV-events** streamed from each vLLM replica (vLLM emits events when cache blocks are created/evicted), enabling *precise* prefix-cache routing (the EPP knows exactly who holds which prefix, not just a probabilistic guess) and tiered offload of cold KV to CPU/disk (LMCache territory, matching SOP §2.3).
- **P/D disaggregation** — prefill pods and decode pods as separate scalable groups, KV handed off over **NIXL** (NVIDIA's KV-transfer library), with a routing sidecar steering the two-phase flow. This is the SOP §2.5 "adopt via llm-d, not hand-rolled" path.
- **Wide-EP** — multi-node expert parallelism for DeepSeek-class MoE models using LeaderWorkerSet, DeepEP kernels, and expert load balancing — the highest-throughput big-MoE path per SOP §2.2.
- **Operational layer** — flow control for multi-tenant fairness, SLO-aware autoscaling (HPA/KEDA plus a workload-variant autoscaler).

### 6.3 "Well-lit paths"

llm-d's central operating idea, and the reason the SOP endorses it: rather than a toolbox of a thousand knobs, the project maintains **benchmarked, tested, documented recipes** — living guides with kustomize configs and Helm charts — for the configurations that are known to work. Current guide catalog (from the repo, Aug 2026): *optimized baseline* (prefix-cache + load-aware routing — start here), *predicted-latency routing*, *precise prefix-cache routing*, *tiered prefix cache*, *P/D disaggregation*, *wide expert-parallelism*, *flow control*, *workload autoscaling*, *fast model actuation*, and workload guides for **agentic serving**, multimodal, and RL; experimental paths include batch gateway and encode disaggregation. This is the same philosophy as `vllm-project/recipes` one level up the stack: start from the recipe, never from scratch.

### 6.4 Install shape (and air-gap implications)

Prerequisites: a working cluster with GPU Operator, Gateway API CRDs, GAIE CRDs, and a gateway control plane (kgateway or Istio). Installation is per-guide: each well-lit path ships kustomize overlays plus Helm charts pulled from OCI registries, parameterized by environment files. There is no single monolithic "llm-d operator" — you compose the guides. For the enclave this is actually convenient: charts are OCI artifacts, so they mirror into Harbor with `oras`/`helm push` exactly like images; the kustomize/values files live in your config repo per SOP §4. Expect to mirror: gateway images (Envoy/kgateway or Istio), EPP/scheduler images, routing sidecar, KV-cache-manager, and your vLLM images.

### 6.5 Relationship to KServe and OpenShift AI

llm-d is deliberately a *stack of components*, not a platform. Two platforms package it: **KServe**, whose `LLMInferenceService` CRD (§7) is explicitly built on llm-d's architecture — the two communities co-announced this "best of both worlds" integration in a joint blog (March 2026); and **Red Hat OpenShift AI**, which productizes llm-d as its supported LLM serving layer. If you buy OpenShift for the enclave, you are getting llm-d with vendor support; if you run vanilla Kubernetes, you use llm-d's charts directly or through KServe.

---

## 7. KServe: the model-serving platform layer

**KServe** (CNCF incubating since November 2025) is the general-purpose model-serving platform for Kubernetes, predating the LLM era: its classic **InferenceService** CRD gives you a one-object deployment for a model — autoscaling (including scale-to-zero via Knative), canary traffic splitting, and a standard prediction protocol — designed for the world of many small predictive models (fraud scorers, classifiers).

**When that classic abstraction helps:** heterogeneous fleets (LLMs *plus* embedding models, rerankers, classic ML), platform teams offering serving-as-a-service to many internal teams, and organizations that want one CRD and one dashboard per model regardless of what's inside.

**When it hurts:** the classic InferenceService assumptions — request-scoped statelessness, scale-to-zero, quick pod startup — are wrong for LLMs (20-minute model loads make scale-to-zero a trap; KV-cache locality makes replicas non-interchangeable). Wrapping vLLM in the predictive-era abstraction adds indirection without adding intelligence.

KServe's answer is a dual-track strategy: keep InferenceService for predictive AI, and introduce **LLMInferenceService** (landed in KServe v0.16; current docs line v0.19 as of Aug 2026) for generative AI, *built on llm-d*. One `LLMInferenceService` object declares: model source (`spec.model` — PVC/S3/HF URI), workload shape (`spec.template` / `spec.worker` / `spec.prefill` — single-node, multi-node via LeaderWorkerSet, or disaggregated prefill/decode), parallelism (`spec.parallelism` — TP/DP/EP), and routing (`spec.router` — Gateway API + the GAIE scheduler). The KServe controller then materializes the llm-d deployment for you.

Decision for our environment: KServe+llm-d is worth it if you want the single-CRD management plane and expect a mixed model fleet; plain llm-d is fewer moving parts (no KServe controller, no Knative option to reason about) if the fleet is all-vLLM. Both end in the same data plane. Verify the LLMInferenceService maturity against the KServe release notes when you get there — it is new in 2026 and evolving fast (flagged).

---

## 8. Kueue: quotas, queueing, and gang admission for shared GPU pools

Vanilla Kubernetes scheduling is first-come-first-served with no concept of "team A gets 24 GPUs, team B gets 8, and batch jobs may borrow idle capacity." **Kueue** (a Kubernetes SIG project) adds job-level admission control *in front of* the scheduler:

- **ResourceFlavor** — a labeled capacity type ("B200 nodes," "B300 nodes").
- **ClusterQueue** — a quota of flavors ("team-agents may hold 32 B300 GPUs"), organized into *cohorts* whose members can borrow each other's idle quota, with preemption to reclaim it.
- **LocalQueue** — the per-namespace mailbox teams submit to; a workload waits in queue until its *entire* resource ask fits, then is admitted whole.

That "admitted whole" property is the practical **gang scheduling**: a 2-node wide-EP deployment needing 16 GPUs either gets all 16 or stays queued — no half-started multi-node model deadlocking the pool while it waits for stragglers. (Kueue does admission/quota; some shops layer Volcano or the KAI scheduler beneath it for finer-grained gang placement and fairness — layered, not either/or.) Kueue also understands serving workloads, not just Jobs: Deployments, StatefulSets, and LeaderWorkerSets can be queued, and recent releases add topology-aware scheduling so multi-node groups land on adjacent fabric.

Where it fits for us: the day the cluster serves two masters — the interactive agent pool *and* nightly batch/eval jobs (SOP profile `throughput-batch`) — Kueue is how night-batch borrows the interactive pool's idle GPUs and gets preempted at 7 a.m. without a human moving anything. Industry write-ups consistently credit this pattern with lifting shared-cluster GPU utilization from the ~30% range to 60%+; treat exact numbers as workload-dependent marketing, but the direction is real.

---

## 9. NVIDIA Dynamo deep dive: the framework-agnostic alternative

### 9.1 What it is

**NVIDIA Dynamo** is a datacenter-scale distributed inference framework — announced at GTC in March 2025 as the successor to the Triton-for-LLMs role, and **1.0 GA on March 16, 2026**, positioned by NVIDIA as "the inference operating system for AI factories." Core is written in **Rust** for performance with Python bindings for extensibility; Apache 2.0 licensed; and — the key philosophical difference from llm-d — **engine-agnostic**: it orchestrates **vLLM, TensorRT-LLM, and SGLang** workers behind one front end.

### 9.2 Components

- **Frontend + Router** — OpenAI-compatible HTTP entry point with **KV-aware routing**: requests are routed to the worker whose cached KV blocks overlap the request most (same goal as llm-d's precise prefix routing, different implementation), balanced against worker load. Discovery and messaging run on an etcd + NATS control plane (as of the 1.0-era architecture).
- **Planner** — an **SLA-driven autoscaler**: you state latency targets (TTFT/ITL), it profiles the workload and continuously right-sizes prefill vs decode pools to meet the SLO at minimum GPU spend. This is the most differentiated piece — llm-d's autoscaling is comparatively earlier-stage.
- **KVBM (KV Block Manager)** — tiered KV-cache memory management across GPU HBM → CPU RAM → local SSD → remote storage, extending effective cache capacity for exactly our workload (long-lived agent sessions).
- **NIXL (NVIDIA Inference Xfer Library)** — the transfer layer moving KV blocks between workers/tiers over NVLink, InfiniBand RDMA, or TCP fallback. Note NIXL is *shared* infrastructure: llm-d's P/D path uses NIXL too. ModelExpress rides it for streaming weights to cold-starting workers.
- **Disaggregated serving** — P/D split as a first-class deployment mode, independently scalable pools.

### 9.3 Kubernetes story: operator, DGD, and Grove

Dynamo predates its Kubernetes maturity — early versions felt like a standalone system that happened to run in pods — but as of 1.0 it ships a **Kubernetes operator with CRDs**: a `DynamoGraphDeployment` (DGD) describes the whole serving graph (frontend, router, prefill workers, decode workers), and a beta `DGDR` (DynamoGraphDeploymentRequest) path generates zero-config deployments. **Grove** is the companion Kubernetes-native orchestrator handling what vanilla K8s can't: topology-aware **gang scheduling** of multi-component, multi-node serving groups (role-aware startup ordering, placing tightly-communicating pods physically close), designed for rack-scale systems like GB200/GB300 NVL72. Grove's scheduling objects are young — verify current maturity before depending on them (flagged).

### 9.4 Dynamo vs llm-d: how to choose

| Dimension | llm-d | Dynamo |
|---|---|---|
| Governance | CNCF Sandbox, multi-vendor community | NVIDIA-led open source (Apache 2.0) |
| Engines | vLLM (primary), SGLang | vLLM, TensorRT-LLM, SGLang |
| K8s integration | Native: standard Gateway API/GAIE CRDs, Helm/kustomize | Own operator + DGD CRDs + Grove; K8s-capable rather than K8s-idiomatic |
| Autoscaling | HPA/KEDA + variant autoscaler (earlier-stage) | SLA-driven Planner (most mature piece) |
| Sweet spot | vLLM fleets on standard Kubernetes | TRT-LLM, NVL72 rack-scale, NVIDIA-supported stacks |

The SOP's stance holds after this research: **on Kubernetes with a vLLM-standard fleet, prefer llm-d** — it composes with the gateway/routing standards everything else uses, and its objects are ordinary K8s objects your team can reason about. **Choose Dynamo when** any of these are true: you adopt TensorRT-LLM for specific models (Dynamo is the only orchestrator treating it as first-class), you field GB300 NVL72 racks where Grove's topology-aware gang scheduling and NVIDIA's own tuning matter, you want the SLA Planner rather than building autoscaling policy yourself, or you want a single-vendor supported inference stack end-to-end. One caveat the SOP phrases slightly differently: the SOP describes Dynamo as running "outside Kubernetes" — as of 1.0 that is out of date; Dynamo has a real K8s operator story. The honest distinction is *K8s-idiomatic (llm-d) vs K8s-hosted (Dynamo)*.

---

## 10. Ray Serve LLM: the alternative for Ray-centric shops

**Ray** is a general distributed-computing framework (Python-native tasks/actors across a cluster); **Ray Serve** is its serving layer; **KubeRay** is the operator running Ray clusters on Kubernetes (`RayCluster`/`RayService` CRDs). Ray Serve's LLM module (`ray.serve.llm`, stable through the Ray 2.5x line in 2026) wraps vLLM engines as Serve deployments: an `LLMConfig` declares model + engine args, and a builder exposes an OpenAI-compatible app with Serve's autoscaling, multi-model composition, and zero-downtime upgrades.

It has real LLM-awareness now: the **PrefixCacheAffinityRouter** checks replica load balance first, and when balanced, uses a prefix tree over recent prompts to route requests sharing prefixes to the same replica — the same cache-locality goal as llm-d's EPP, implemented inside Ray's routing layer rather than at a gateway.

When it's the right answer: **you are already a Ray shop** — data pipelines, batch inference, RL fine-tuning, or training on Ray — and want serving inside the same cluster, resource pool, and Python codebase. The agent-loop-plus-tools pattern also sometimes pulls teams here, since tool execution can be Ray tasks colocated with serving. When it isn't: adopting Ray *only* to serve vLLM buys you a second distributed system to operate (head-node HA, Ray version matrices, object store tuning) in exchange for features llm-d gives you on standard K8s primitives. In an air gap, that is a whole additional platform's images and upgrade cadence. For our environment: not the default; revisit only if a Ray-based batch/RL workload arrives first.

---

## 11. The plain-K8s baseline: one vLLM model, by hand

Before adopting any orchestration layer, you should be able to write the baseline: one model, one Deployment, one Service, model weights from the NFS-backed store. This is also genuinely sufficient for Tier-3-entry scale (a few single-node models with dumb routing), and every layer above (§6–§9) is "this, plus smarter routing and placement."

Worked example — annotations inline. Everything references the enclave Harbor by digest and local model paths, per SOP §4.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-store
  namespace: serving
spec:
  accessModes: ["ReadOnlyMany"]        # many pods read the same weights
  storageClassName: nfs-models          # your NFS-backed class
  resources:
    requests:
      storage: 2Ti
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-default-vllm
  namespace: serving
  labels: { app: agent-default-vllm }
spec:
  replicas: 1
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 0          # no spare node? old pod must die before new one starts
      maxUnavailable: 1    # (with a spare node, flip these for zero-downtime)
  selector:
    matchLabels: { app: agent-default-vllm }
  template:
    metadata:
      labels: { app: agent-default-vllm }
      annotations:                       # simplest Prometheus discovery
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-B200   # GFD label from GPU Operator
      terminationGracePeriodSeconds: 180      # let in-flight generations drain
      volumes:
        - name: models
          persistentVolumeClaim: { claimName: model-store, readOnly: true }
        - name: shm                             # TP workers use /dev/shm;
          emptyDir: { medium: Memory, sizeLimit: 32Gi }   # default 64MB will crash TP=8
      containers:
        - name: vllm
          image: harbor.enclave.local/serving/vllm-openai@sha256:<digest>
          args:
            - "--model=/models/org/model-NVFP4"
            - "--served-model-name=prod-agent-v1"
            - "--tensor-parallel-size=8"
            - "--kv-cache-dtype=fp8"
            - "--max-model-len=131072"
            - "--gpu-memory-utilization=0.90"
            - "--enable-auto-tool-choice"
            - "--tool-call-parser=<parser>"     # from the mirrored recipe
            - "--port=8000"
          env:
            - { name: HF_HUB_OFFLINE, value: "1" }        # never phone home
            - { name: VLLM_NO_USAGE_STATS, value: "1" }
            - { name: DO_NOT_TRACK, value: "1" }
          ports:
            - { containerPort: 8000, name: http }
          resources:
            limits:
              nvidia.com/gpu: 8
              memory: 512Gi
              cpu: "64"
          volumeMounts:
            - { name: models, mountPath: /models, readOnly: true }
            - { name: shm, mountPath: /dev/shm }
          startupProbe:                 # THE critical probe: loading a 400B
            httpGet: { path: /health, port: 8000 }   # NVFP4 model + CUDA-graph
            periodSeconds: 15           # capture can take 10-20 minutes.
            failureThreshold: 80        # 80 x 15s = 20 min before "failed"
          readinessProbe:               # gates Service traffic
            httpGet: { path: /health, port: 8000 }
            periodSeconds: 10
          livenessProbe:                # restarts a hung engine
            httpGet: { path: /health, port: 8000 }
            periodSeconds: 30
            failureThreshold: 5
---
apiVersion: v1
kind: Service
metadata:
  name: agent-default-vllm
  namespace: serving
spec:
  selector: { app: agent-default-vllm }
  ports:
    - { name: http, port: 8000, targetPort: 8000 }
```

**Common pitfalls box:**

- **No startup probe → restart loop of death.** Without `startupProbe`, the liveness probe starts failing during the 15-minute model load, K8s kills the pod, and it loops forever. This is the single most common first-day failure.
- **Default `/dev/shm` (64 MB) breaks tensor parallelism.** vLLM's TP workers exchange data through shared memory; mount a memory-backed `emptyDir` as above.
- **`maxSurge: 1` with no free node = deadlock.** A rolling update tries to start the new 8-GPU pod before killing the old one; with no idle 8-GPU node it waits forever. Either keep a spare node (true zero-downtime) or accept `maxSurge: 0` (brief gap, alias-level blue/green covers you).
- **Weights baked into images** — violates SOP §4.1; keep weights on the PVC.
- **Forgetting `terminationGracePeriodSeconds`** — default 30 s truncates in-flight agent generations on every deploy.

What this baseline lacks — and the exact gaps §5–§9 fill: the Service round-robins across replicas (no prefix-cache affinity), there is no request-level load shedding, no P/D or multi-node option, and promotion/canary is manual. Keep the gateway (LiteLLM or agentic-api) in front of these Services doing alias routing; that layer is unchanged from Tier 2.

---

## 12. Air-gapped Kubernetes specifics

Everything in SOP §4 (Harbor, digests, signed bundles) applies; K8s adds its own layer of "things that silently assume internet."

**Control-plane images.** Kubernetes itself is containers: API server, controller-manager, scheduler, etcd, CoreDNS, kube-proxy, plus your CNI. All must live in Harbor. With **kubeadm**, `kubeadm config images list` enumerates them and `--image-repository harbor.enclave.local/k8s` pulls from the mirror. Every add-on chart (GPU Operator, gateway, llm-d, monitoring) needs its image list mirrored and values overridden — build the habit of generating the full image manifest per release bundle.

**Distribution choice.** **RKE2** (Rancher's hardened Kubernetes) has the most mature first-class air-gap flow — documented tarball installs (drop `rke2-images.linux-amd64.tar.zst` into `/var/lib/rancher/rke2/agent/images/` and the runtime loads them locally), plus `system-default-registry` to point all system images at Harbor, and a `registries.yaml` mirror config for containerd. It is common in government/regulated enclaves for this reason and is a sensible default here. **kubeadm** works fine with more assembly. **k3s** shares RKE2's air-gap machinery and suits edge/small footprints more than 8-GPU HGX fleets.

**Zarf.** [Zarf](https://zarf.dev) (open source, defense-origin) is the de facto packaging tool for moving whole Kubernetes workloads across an air gap: a declarative `zarf.yaml` lists charts, images, manifests, and files; `zarf package create` on the connected side pulls everything into one signed tarball; `zarf package deploy` on the inside pushes images into the enclave registry and installs the charts. It effectively industrializes SOP §4.2's bundle format for the K8s layers of the stack — worth adopting the day the bundle contents become "twelve Helm charts" instead of "three images."

**The upgrade burden — budget for it.** Upstream Kubernetes ships three minor releases a year, each supported ~14 months; version-skew rules limit how far kubelets may lag the control plane, so you cannot simply stop upgrading. Realistic enclave cadence: a quarterly "platform bundle" (control plane + CNI + GPU Operator + gateway + llm-d + monitoring images and charts, tested together on a staging cluster on the connected side) crossing the gap alongside model bundles. Plus standing care: scheduled etcd snapshots (and a tested restore), certificate rotation awareness, and a staging cluster inside or outside the enclave to rehearse upgrades. This recurring cost is the concrete content of the SOP's warning that K8s in an air gap is "a significant standing cost" — roughly, one platform engineer's recurring attention, forever.

---

## 13. Choosing, and migrating from the compose tier

### 13.1 Comparison matrix

| Option | Routing smarts | Multi-node paths | Engines | Choose when |
|---|---|---|---|---|
| **Plain K8s** (Deploy+Service) | None (round-robin) | DIY (LWS by hand) | any | Entry point; few single-node models, gateway does aliasing |
| **llm-d** | Prefix/KV/latency-aware EPP via GAIE | P/D + wide-EP as well-lit paths | vLLM, SGLang | Default: vLLM fleet on standard K8s (SOP stance) |
| **KServe + llm-d** | Same data plane as llm-d | Same, via `LLMInferenceService` | vLLM + classic ML runtimes | Platform team, mixed predictive+LLM fleet, one-CRD management |
| **Dynamo** | KV-aware router + SLA Planner | P/D first-class; Grove gang scheduling | vLLM, TRT-LLM, SGLang | TRT-LLM in the mix, NVL72 racks, NVIDIA-supported stack |
| **Ray Serve LLM** | PrefixCacheAffinityRouter | Ray placement groups | vLLM | Already a Ray shop (data/RL/batch on Ray) |

Kueue is orthogonal — add it to any of these when the GPU pool is shared across teams or day-interactive/night-batch. GAIE is likewise not a competitor: it is the standard llm-d and KServe build on, and the safest single thing to standardize on early.

### 13.2 Migration plan: Tier 2 (compose) → Tier 3 (K8s + llm-d), preserving the gateway/alias layer

The SOP's design intent is that the registry, model store, profiles, and gateway carry over unchanged. The migration that honors that:

1. **Stand up the cluster beside production.** RKE2 (or kubeadm) on 1–2 nodes withdrawn from the compose pool; GPU Operator; NFS model store mounted as the same paths (`/models/...`) via PV/PVC; monitoring stack federated into the existing enclave Prometheus. Nothing user-facing changes.
2. **Port one model as the plain-K8s baseline (§11).** Same image digest, same recipe-derived flags, same profile name. Validate with the same eval + guidellm gates as any model promotion.
3. **Point the existing gateway at it.** The Tier-2 gateway (LiteLLM / agentic-api) simply gains a new backend URL — the K8s Service (exposed via the gateway's network or a NodePort/LoadBalancer). Promote via alias weight (5% → 25% → 100%), exactly like a model promotion. **This is the load-bearing trick: because clients only ever call aliases, replatforming is invisible to them.**
4. **Migrate models one at a time,** draining the compose stack as capacity frees up. Keep compose runnable as the rollback tier until the last model moves.
5. **Adopt llm-d when replica count makes routing matter.** Install Gateway API + GAIE CRDs + kgateway/Istio, deploy the optimized-baseline well-lit path, and move multi-replica models behind InferencePools. The agentic gateway stays in front — it speaks `/v1/responses` and owns aliases/state; llm-d's inference gateway sits behind it doing endpoint picking. Two layers, two jobs.
6. **Then, and only as triggers fire:** Kueue when the pool is shared; P/D disaggregation when long-prefill interference is chronic (SOP runbook 5); wide-EP when a single node can no longer serve the flagship MoE.

---

## Study questions

1. **Why can't a standard Kubernetes Service load-balance LLM traffic well, and what replaces it?**
   Answer: Services round-robin connections, but LLM requests have wildly unequal cost and strong cache locality — a replica holding a session's KV prefix serves it 5–10× faster. The Gateway API Inference Extension replaces the Service with an InferencePool whose Endpoint Picker (EPP) routes per-request using live queue depth, KV utilization, and prefix-cache state.

2. **List the components the NVIDIA GPU Operator installs and the one-line job of each.**
   Answer: NFD (label node hardware), driver container (kernel driver as a container), container toolkit (containerd GPU wiring), device plugin (advertise `nvidia.com/gpu`), GFD (GPU-specific labels), DCGM exporter (Prometheus GPU metrics), MIG manager (partition profiles), validator (smoke test).

3. **Your vLLM pod crash-loops on deploy; logs show the model was still loading when the container restarted. What's missing?**
   Answer: A startupProbe. Liveness probing began during the 10–20 minute model load and killed the pod. Add a startupProbe with failureThreshold × periodSeconds comfortably exceeding worst-case load time; liveness only takes over after startup succeeds.

4. **Time-slicing vs MIG: which is safe for multi-tenant small-model serving and why?**
   Answer: MIG — it partitions the GPU in hardware with isolated memory and fault domains (up to 7 instances). Time-slicing merely interleaves contexts with no memory isolation, so one tenant's OOM or crash can take down neighbors; it's acceptable only for trusted dev/burst workloads.

5. **What is DRA and should you build on it today (Aug 2026)?**
   Answer: Dynamic Resource Allocation — the claims-based successor to the device-plugin integer counter, letting workloads request devices by attributes. Core is GA since K8s v1.34 and the NVIDIA driver is now GPU-Operator-integrated, but the device plugin remains the default production path; adopt DRA when you need ComputeDomains (multi-node NVLink) or attribute-based selection, and expect the orchestration layers to adopt it beneath you.

6. **Why does multi-node inference need Multus or hostNetwork, and what's the classic silent failure?**
   Answer: The pod network (CNI) can't carry RDMA; NCCL/NVSHMEM traffic needs the real InfiniBand/RoCE interfaces, via hostNetwork or a Multus-attached SR-IOV VF plus RDMA device plugin. The classic failure is NCCL silently falling back to TCP — everything works at ~10% bandwidth; catch it with NCCL_DEBUG=INFO (look for NET/IB) and nccl-tests at node acceptance.

7. **What was InferenceModel renamed to, and why does it matter operationally?**
   Answer: InferenceObjective, at the GAIE v1.0 GA (API group also moved from inference.networking.x-k8s.io to inference.networking.k8s.io). Pre-2026 tutorials and charts use the old names, so copied YAML can target CRDs your cluster doesn't have — check API versions first.

8. **What are llm-d's "well-lit paths," and which one do you deploy first?**
   Answer: Benchmarked, maintained deployment recipes (kustomize + Helm) for known-good configurations — the fleet-level analog of vllm-project/recipes. Start with the "optimized baseline" (prefix-cache + load-aware routing); adopt P/D disaggregation and wide-EP paths only when their SOP triggers fire.

9. **When would you pick Dynamo over llm-d?**
   Answer: TensorRT-LLM as a first-class engine, GB200/GB300 NVL72 rack-scale topologies where Grove's topology-aware gang scheduling matters, wanting the SLA-driven Planner autoscaler, or a single-vendor NVIDIA-supported stack. On a standard-K8s vLLM fleet, llm-d is the more idiomatic choice (SOP default).

10. **What does Kueue add that the default scheduler lacks, and what is gang admission?**
    Answer: Team quotas (ClusterQueue/ResourceFlavor), queueing with borrowing and preemption across cohorts, and all-or-nothing admission — a workload needing 16 GPUs across 2 nodes is admitted only when everything fits, preventing half-started multi-node deployments from deadlocking the pool.

11. **Name three K8s-specific air-gap costs that don't exist at the compose tier.**
    Answer: Mirroring the control plane's own images (etcd, API server, CoreDNS, CNI) plus every operator chart's images; a forced upgrade treadmill (3 minor releases/year, ~14-month support, kubelet skew limits) requiring quarterly platform bundles; and etcd care — scheduled snapshots and tested restores.

12. **In the Tier 2 → Tier 3 migration, what makes replatforming invisible to agent clients?**
    Answer: The alias layer. Clients call gateway aliases (agent-default), never model names or backend URLs, so moving a model from a compose container to a K8s Service (and later behind llm-d) is just repointing the gateway's backend and shifting alias traffic weights — the same procedure as a model promotion.

---

## Sources

Primary project sources (fetched):

- llm-d repository and docs: https://github.com/llm-d/llm-d and https://llm-d.ai/docs/architecture (v0.7 May 2026; v0.8 referenced in docs, Aug 2026)
- llm-d guides / well-lit paths: https://github.com/llm-d/llm-d/tree/main/guides
- CNCF: Welcome llm-d (Sandbox, Mar 24, 2026): https://www.cncf.io/blog/2026/03/24/welcome-llm-d-to-the-cncf-evolving-kubernetes-into-sota-ai-infrastructure/
- Gateway API Inference Extension docs: https://gateway-api-inference-extension.sigs.k8s.io/ and releases: https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases
- Kubernetes blog — v1.34 DRA GA: https://kubernetes.io/blog/2025/09/01/kubernetes-v1-34-dra-updates/
- NVIDIA GPU Operator docs: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html and release notes (26.3): https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/26.3/release-notes.html
- NVIDIA DRA driver for GPUs: https://github.com/NVIDIA/k8s-dra-driver-gpu and https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/dra-intro-install.html
- NVIDIA Dynamo: https://github.com/ai-dynamo/dynamo, https://docs.nvidia.com/dynamo/latest/getting-started/introduction, and Dynamo 1.0 announcement: https://nvidianews.nvidia.com/news/dynamo-1-0
- KServe LLMInferenceService overview: https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview and KServe+llm-d blog: https://kserve.github.io/website/blog/cloud-native-ai-inference-kserve-llm-d
- CNCF: KServe becomes incubating (Nov 2025): https://www.cncf.io/blog/2025/11/11/kserve-becomes-a-cncf-incubating-project/
- Ray Serve prefix-aware routing: https://docs.ray.io/en/latest/serve/llm/user-guides/prefix-aware-routing.html
- RKE2 air-gap install: https://docs.rke2.io/install/airgap
- Zarf: https://zarf.dev

Secondary (context/verification):

- Red Hat: GPUDirect RDMA on OpenShift AI: https://developers.redhat.com/articles/2025/04/29/accelerate-model-training-openshift-ai-nvidia-gpudirect-rdma
- Red Hat: KServe + llm-d for gen-AI inference (Apr 2026): https://developers.redhat.com/articles/2026/04/21/kserve-llm-d-optimized-gen-ai-inference
- CoreWeave: Kueue for AI workloads: https://www.coreweave.com/blog/kueue-a-kubernetes-native-system-for-ai-training-workloads
- Kubernetes blog: Introducing Gateway API Inference Extension: https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/
