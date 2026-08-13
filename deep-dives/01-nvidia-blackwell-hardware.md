# NVIDIA Blackwell Hardware for Inference: B200, B300, and the Systems Around Them

**Deep-dive 01 in the on-prem inference series.** Companion to the [deployment SOP](../SOP-production-ai-model-deployment.md) and [PRIMER](../PRIMER.md). Written August 2026; hardware facts verified against NVIDIA documentation current as of mid-2026. Anything fast-moving carries a date or version.

**Audience:** an engineer who knows servers and Linux but is new to GPU inference hardware. Every acronym is defined at first use.

---

## Key takeaways

- LLM inference has two phases with opposite hardware appetites: **prefill** (reading the prompt) is compute-bound and eats floating-point throughput; **decode** (generating tokens) is memory-bandwidth-bound and eats HBM bandwidth. You size and pick GPUs for both, not one.
- A Blackwell GPU is **two dies fused into one** by a 10 TB/s die-to-die link (NV-HBI). Software sees a single GPU; you never manage the dies separately.
- The headline inference feature of this generation is **hardware FP4 (NVFP4)**: 4-bit tensor-core math with per-block scaling. It roughly halves weight bytes versus FP8 and roughly doubles peak math versus FP8 — which is why the SOP makes NVFP4 the default production weight format.
- **B300 (Blackwell Ultra) is an inference-specialist refresh of B200**: 288 GB HBM3e instead of 192 GB, ~1.5× the *dense* FP4 math, and ~2× faster softmax/attention instructions. FP8 and memory bandwidth are essentially unchanged.
- Exact TFLOPS and memory numbers **vary by SKU and power limit**: the same "B200" is 9 PFLOPS dense FP4 at 1,000 W in an HGX board but 10 PFLOPS at 1,200 W in a GB200 rack, and shipping HGX/DGX B200 systems expose ~180 GB usable per GPU, not the marketing 192 GB. Always read the datasheet for *your* SKU.
- Form factors are a ladder: **HGX** (8-GPU baseboard inside a vendor server) → **DGX** (NVIDIA's own server built on HGX/MGX) → **NVL72** (a full liquid-cooled rack where 72 GPUs + 36 Grace CPUs share one NVLink domain). For most enterprise on-prem inference, HGX nodes are the right unit; NVL72 is for frontier-scale multi-node models.
- Inside a node or NVL72 rack, GPUs talk over **NVLink 5** (1.8 TB/s per GPU) through NVSwitch. Between nodes you choose **Quantum-X800 InfiniBand** (lowest latency, best collectives) or **Spectrum-X Ethernet with RoCEv2** (fits existing Ethernet operations). Validate whichever you buy *before* you need multi-node serving.
- Power is a facility problem now: ~1,000–1,400 W per GPU, ~14 kW per 8-GPU server, ~120–150 kW per NVL72 rack (liquid cooling mandatory at rack scale).
- Blackwell's software floor is **CUDA 12.8+ with an R570-or-newer driver** (compute capability 10.0, `sm_100`), plus Fabric Manager on any NVSwitch system, and NCCL for multi-GPU communication.
- Node acceptance is not optional: `dcgmi diag -r 3` (and `-r 4` for burn-in), `nccl-tests` bandwidth checks against a recorded golden number, and ECC/row-remap inspection catch most bad nodes before they corrupt a production rollout.

---

## 1. GPU inference fundamentals: what actually limits speed

### 1.1 Two budgets: math and bytes

Every GPU has two headline budgets:

- **Compute:** how many floating-point operations per second (FLOPS) its tensor cores can execute. Measured in TFLOPS (teraFLOPS, 10^12) or PFLOPS (petaFLOPS, 10^15).
- **Memory bandwidth:** how many bytes per second it can move between its on-package memory (HBM — high-bandwidth memory, the GPU's RAM) and its compute units. Measured in TB/s.

A workload is **compute-bound** when the math runs out first (the memory system could feed it faster than it can chew) and **memory-bandwidth-bound** when the bytes run out first (the tensor cores sit idle waiting for data).

### 1.2 Roofline intuition in one paragraph

The **roofline model** is the standard way to reason about this. For any kernel, compute its **arithmetic intensity**: FLOPs performed per byte moved from memory. Compare that to the GPU's ratio of peak FLOPS to peak bandwidth (the "ridge point"). If your intensity is below the ridge, you are bandwidth-bound and the only cures are moving fewer bytes (quantization, caching) or buying more bandwidth. If above, you are compute-bound and the cures are faster math (lower precision, more FLOPS) or less math.

Concrete ridge points (dense math, HGX-class B200):

```
B200 ridge (FP8):  4,500 TFLOPS / 8 TB/s  ≈ 560 FLOPs per byte
B200 ridge (BF16): 2,250 TFLOPS / 8 TB/s  ≈ 280 FLOPs per byte
```

So on a B200, a kernel must do hundreds of floating-point operations for every byte it reads to keep the tensor cores busy. Very few inference kernels get there without batching.

### 1.3 Why prefill and decode stress different limits

- **Prefill** processes the whole prompt in one parallel pass. Each weight loaded from HBM is reused across every token in the prompt, so arithmetic intensity is high (roughly proportional to prompt length × batch). Long agent prompts push prefill firmly into **compute-bound** territory — this is where FP4/FP8 TFLOPS numbers matter, and it determines **TTFT** (time to first token).
- **Decode** generates one token at a time per sequence. To produce a single token, the GPU must stream essentially **all model weights plus the KV cache** (key-value cache — the model's stored per-conversation attention state) through from HBM, and each byte is used in only a couple of operations. Intensity at batch size 1 is on the order of 1–2 FLOPs/byte — a hundred times below the ridge. Decode is **memory-bandwidth-bound**, and it determines **ITL** (inter-token latency).

A useful back-of-envelope: single-stream decode speed is capped near `bandwidth ÷ bytes-read-per-token`. For a 70-billion-parameter dense model:

```
FP8 weights  (~70 GB):  8 TB/s ÷ 70 GB ≈ 114 tokens/s ceiling per sequence
NVFP4 weights (~35 GB): 8 TB/s ÷ 35 GB ≈ 228 tokens/s ceiling per sequence
```

That is the whole quantization story in two lines: **fewer bits = fewer bytes per token = faster decode**, before you even count the extra TFLOPS. Serving engines like vLLM then batch many sequences together so each weight-load is amortized across dozens of tokens, pushing the aggregate workload up the roofline — which is why throughput per GPU climbs steeply with concurrency while per-user latency degrades only gently, until HBM fills up with KV cache.

The capacity side matters just as much: on an agentic serving pool, most HBM is spent on **KV cache for concurrent long sessions**, not weights (the PRIMER's central point). That is why the B300's +96 GB per GPU translates directly into more concurrent agent sessions per node rather than just "bigger models."

> **Common pitfall:** comparing GPUs by TFLOPS alone. For decode-heavy agent traffic, HBM capacity and bandwidth predict real throughput far better than peak FLOPS. A GPU with 2× FLOPS and the same bandwidth decodes at nearly the same speed.

### 1.4 Worked example: sizing one node for an agent pool

Take a concrete (illustrative) model: a 120B-parameter dense model with 64 layers, 8 KV heads (grouped-query attention), head dimension 128, serving agent sessions at 64k context on one 8×B300 node.

**Step 1 — weights.** NVFP4 ≈ 0.55–0.6 bytes/parameter effective (4-bit values + block scales):

```
120e9 params × 0.6 B ≈ 72 GB weights
Node HBM (8 × ~270 GB usable) ≈ 2,160 GB → 2,088 GB left after weights
```

**Step 2 — KV cache per token.** Per token, per layer, KV stores 2 (K and V) × kv_heads × head_dim values:

```
2 × 8 heads × 128 dim × 64 layers = 131,072 values/token
FP8 KV (1 byte/value, SOP default)  ≈ 128 KB per token
64k-token session                   ≈ 8 GB of KV per session
```

**Step 3 — concurrency.** Reserve ~10% HBM headroom (vLLM `--gpu-memory-utilization 0.90`):

```
(2,160 × 0.90 − 72) GB ÷ 8 GB/session ≈ 234 concurrent full-length sessions
Same math on 8×B200 (~1,440 GB): ≈ 153 sessions — the B300 delta in one line
```

Real agent sessions average far below max context, so practical concurrency is higher; but this worst-case math is what you commit to in an SLO. It also shows why `--kv-cache-dtype fp8` matters: BF16 KV would halve both numbers. Run the same arithmetic for any candidate model before promising capacity — then verify with a `guidellm` sweep (SOP §7), because attention kernels, prefix-cache hit rates, and scheduling overhead move the real number.

---

## 2. The Blackwell architecture

### 2.1 Dual-die design and NV-HBI

Blackwell is NVIDIA's data-center GPU generation that succeeded Hopper (H100/H200). A single manufacturing die had hit the "reticle limit" (the largest chip a lithography machine can print), so Blackwell packages **two reticle-sized dies** side by side and joins them with **NV-HBI (NVIDIA High-Bandwidth Interface)**, a die-to-die interconnect providing **10 TB/s** of bandwidth. The package totals **208 billion transistors** on the TSMC 4NP process.

The critical operational fact: NV-HBI is fast and coherent enough that **software sees one GPU** with one unified 192 GB (B200) or 288 GB (B300) memory space. There is no "which die am I on" in CUDA, in vLLM, or in your monitoring. You configure tensor parallelism across *GPUs*, never across dies.

- **B200** ships with **148 streaming multiprocessors (SMs)** enabled across the two dies (an SM is the GPU's basic compute block, analogous to a CPU core cluster).
- **B300 (Blackwell Ultra)** enables the full **160 SMs**, each SM containing 4 fifth-generation tensor cores and 256 KB of Tensor Memory (TMEM).

### 2.2 Fifth-generation Tensor Cores and the second-generation Transformer Engine

**Tensor cores** are dedicated matrix-multiply units — the hardware that does virtually all the math in an LLM. Blackwell's fifth generation adds native support for 4-bit and 6-bit floating point (FP4/FP6) alongside FP8, FP16/BF16.

The **Transformer Engine** is the hardware-plus-software layer that manages precision per layer: it tracks value ranges and applies scaling factors so low-precision math stays numerically stable. Blackwell's **second-generation Transformer Engine** introduces **microscaling**: instead of one scale factor per whole tensor, scales are applied to small blocks of values.

**NVFP4** is NVIDIA's specific 4-bit format: each weight is an E2M1 value (2 exponent bits, 1 mantissa bit), with an FP8 scale factor shared by each block of 16 values, plus a tensor-level FP32 scale. The two-level scaling is what keeps 4-bit accuracy losses small (typically ~1% on quality benchmarks for well-quantized checkpoints — which your eval gate verifies per model, per the SOP).

### 2.3 The precision ladder

Peak dense throughput roughly doubles at each halving of precision (sparse figures — for hardware structured sparsity that production LLM weights essentially never use — are 2× the dense numbers; ignore them when sizing):

| Precision | B200 (HGX, 1,000 W) | B300 (chip max, 1,400 W) |
|---|---|---|
| BF16/FP16 dense | ~2.25 PFLOPS | ~2.25 PFLOPS (unverified exact) |
| FP8 dense | 4.5 PFLOPS | 5 PFLOPS |
| FP4 dense | 9 PFLOPS | **15 PFLOPS** |
| FP4 sparse | 18 PFLOPS | 20 PFLOPS |

Read the FP4 row carefully: B300's dense-FP4 uplift (1.5×) came from adding dense-mode throughput, *not* from raising the sparse ceiling — dense FP4 is the number that matters for real model serving.

### 2.4 What Blackwell Ultra (B300) actually changes

Per NVIDIA's Blackwell Ultra architecture blog (Aug 2025), versus B200:

1. **288 GB HBM3e** (eight 12-high stacks) versus 192 GB — 50% more, at the same ~8 TB/s bandwidth.
2. **1.5× dense NVFP4** (15 vs 10 PFLOPS at chip max power).
3. **~2× faster attention-layer special functions:** the SFU (special function unit) throughput for the exponential instructions inside softmax doubled — 10.7 tera-exponentials/s vs 5. Softmax sits in every attention layer, so long-context attention (exactly what agentic workloads generate) speeds up beyond what the FLOPS numbers suggest.
4. **Higher power:** up to 1,400 W TGP vs 1,000–1,200 W.
5. **PCIe Gen6 x16** host interface (256 GB/s bidirectional).

What did *not* meaningfully change: FP8 throughput, memory bandwidth, NVLink generation. B300 is a memory-and-FP4-and-attention refresh aimed squarely at inference — which is why the SOP steers the agentic serving pool to B300 and keeps B200 for batch and smaller models.

One trade-off worth knowing: Blackwell Ultra reallocated die area toward low-precision tensor math; FP64 (double-precision, used in scientific HPC, not LLMs) and INT8 throughput are reduced relative to what an HPC buyer might expect. Irrelevant for LLM serving; relevant if someone proposes sharing the cluster with traditional HPC codes. Note also that native INT8 paths are a dead end on Blackwell — quantization strategy should be FP8/NVFP4, matching the SOP.

---

## 3. Spec sheet: H200 vs B200 vs B300

Figures below are chip-level peaks from NVIDIA datasheets and the Blackwell Ultra architecture blog, as of mid-2026. Dense (no-sparsity) numbers throughout.

| Spec | H200 (Hopper) | B200 | B300 (Ultra) |
|---|---|---|---|
| HBM capacity | 141 GB HBM3e | 192 GB HBM3e (~180 GB usable in HGX/DGX) | 288 GB HBM3e |
| HBM bandwidth | 4.8 TB/s | ~7.7–8 TB/s | 8 TB/s |
| FP4 dense | — (no hardware FP4) | 9–10 PFLOPS (SKU-dependent) | 13.5–15 PFLOPS (SKU-dependent) |
| FP8 dense | ~1.98 PFLOPS | 4.5–5 PFLOPS | ~4.5–5 PFLOPS |
| BF16 dense | ~0.99 PFLOPS | ~2.25 PFLOPS | ~2.25 PFLOPS |

| Spec | H200 | B200 | B300 |
|---|---|---|---|
| NVLink per GPU | 900 GB/s (NVLink 4) | 1.8 TB/s (NVLink 5) | 1.8 TB/s (NVLink 5) |
| Power (TGP) | 700 W | 1,000 W (HGX) / 1,200 W (GB200) | up to 1,400 W |
| Compute capability | 9.0 (`sm_90`) | 10.0 (`sm_100`) | 10.3 (`sm_103`) |
| Dies per package | 1 | 2 (NV-HBI 10 TB/s) | 2 (NV-HBI 10 TB/s) |
| MIG support | up to 7 instances | up to 7 instances (1g.23gb smallest) | up to 7 instances |

> **SKU variation warning.** "B200" and "B300" are chip families, and the numbers flex with the power limit and platform:
> - **HGX B200** (1,000 W/GPU): 72 PFLOPS dense FP4 per 8-GPU board = **9 PFLOPS/GPU**; total board memory listed as **1.4 TB** (≈180 GB usable/GPU — `nvidia-smi` on shipping systems reports ~180 GB, not 192).
> - **GB200 NVL72** (1,200 W/GPU): **10 PFLOPS/GPU** dense FP4.
> - **HGX B300** (NVIDIA lists 108 PFLOPS dense FP4 per board = **13.5 PFLOPS/GPU**; board memory listed as 2.1 TB on the HGX page but 2.3 TB in the DGX B300 documentation — treat ~270 GB usable/GPU as the planning number until you read `nvidia-smi` on your own hardware).
> - **GB300 NVL72** (1,400 W/GPU): **15 PFLOPS/GPU** dense FP4.
> Capacity-plan from the datasheet of the exact system you bought, then trust only `nvidia-smi` output on the delivered node.

---

## 4. System form factors: HGX, DGX, MGX, NVL72

### 4.1 The vocabulary

- **SXM module:** the physical form of a data-center GPU — a mezzanine card that sockets onto a baseboard, not a PCIe card. All B200/B300 discussed here are SXM parts.
- **HGX:** NVIDIA's standard **8-GPU baseboard** (universal baseboard, UBB): 8 SXM GPUs plus NVSwitch chips wiring them all-to-all with NVLink. Server vendors (Supermicro, Dell, Lenovo, etc.) build servers around it and add CPUs, RAM, NICs, storage. When you buy an "8×B200 server," you are buying an HGX B200 inside a vendor chassis.
- **DGX:** NVIDIA's **own-brand server** built on the same 8-GPU foundation (DGX B200, DGX B300) with a fixed, supported hardware/software configuration. Functionally an HGX node with NVIDIA's integration and support contract.
- **MGX:** NVIDIA's **modular reference design** program for server vendors — standardized chassis/power/cooling building blocks. The DGX B300 is itself MGX-compliant. As a buyer you mostly notice MGX as "why the 2026 servers from different vendors look alike and can take DC busbar power."
- **GB200 / GB300 NVL72:** a **full rack as one computer**: 72 Blackwell GPUs plus 36 **Grace CPUs** (NVIDIA's Arm-based server CPU, 72 Neoverse V2 cores, up to 480 GB LPDDR5X each), all in one NVLink domain.

### 4.2 HGX B200 vs HGX B300 nodes

An HGX B200 node: 8 GPUs, ~1.4 TB pooled HBM over NVLink, 14.4 TB/s aggregate NVLink bandwidth, x86 CPUs, your choice of NICs, air- or liquid-cooled depending on vendor chassis, ~10–14 kW.

The **HGX B300 is named "HGX B300 NVL16"** — NVIDIA now counts the 16 compute *dies* (8 packages × 2 dies) in the NVLink domain rather than the 8 packages. Same 8-GPU reality. Two genuinely new things (per ServeTheHome's teardown and NVIDIA docs):

1. **ConnectX-8 NICs live on the baseboard itself** — eight 800 Gb/s SuperNICs integrated on the UBB (1.6 TB/s total external bandwidth per node), replacing the separate PCIe switch/retimer tree of earlier HGX designs (ConnectX-8 has a built-in PCIe Gen6 switch). Practical consequence: on B300 nodes your east-west GPU network is NVIDIA end-to-end by construction.
2. **~2.3 TB HBM per node** and DGX B300 at 14.5 kW max, air-cooled MGX chassis option — B300 does not force liquid cooling at the node level.

### 4.3 GB200/GB300 NVL72: the rack is the computer

An NVL72 rack contains **18 compute trays** (each: 2 Grace CPUs + 4 Blackwell GPUs on Grace-Blackwell "superchip" boards) and **9 NVLink switch trays**, joined by a blind-mate **passive copper backplane** carrying NVLink to every GPU. The result:

- **72 GPUs in one NVLink domain** — 130 TB/s aggregate NVLink bandwidth; any GPU reads any other GPU's HBM at 1.8 TB/s without touching a NIC.
- **Pooled memory:** ~13.4 TB HBM3e (GB200) or ~20.7 TB (GB300), plus ~17 TB of Grace LPDDR5X addressable over NVLink-C2C (900 GB/s coherent CPU-GPU link).
- **Rack totals:** GB200 NVL72 ≈ 720 PFLOPS dense FP4; GB300 NVL72 ≈ 1,080 PFLOPS (~1.1 exaFLOPS) dense FP4.
- **100% liquid-cooled, ~120–150 kW per rack.** There is no air option.

Why it exists: a trillion-parameter-class MoE (mixture-of-experts) model served with wide expert parallelism wants *all* of its expert-shuffling traffic on NVLink rather than InfiniBand. Inside an NVL72, "multi-node" communication patterns run at NVLink speed across 72 GPUs.

### 4.4 Choosing: HGX nodes vs NVL72 racks

| Factor | HGX B200/B300 nodes | GB200/GB300 NVL72 |
|---|---|---|
| Unit of purchase/failure | 1 server (8 GPUs, ~14 kW) | 1 rack (72 GPUs, ~120–150 kW) |
| Biggest model at full speed | what fits in 8 GPUs (huge with NVFP4: ~700B-class MoE) | multi-hundred-B to trillion-class MoE across the rack |
| Cooling | air possible (vendor-dependent) | liquid mandatory + facility water |
| CPU architecture | x86 (standard tooling) | Arm Grace (containers must be arm64) |
| Ops complexity | standard server ops | rack-scale: NVLink fabric mgmt, CDU, Arm images |

Rules of thumb for an enterprise like ours (matches the SOP's stance):

- If your production models fit on one node in NVFP4 — and with 8×288 GB = ~2.3 TB per B300 node, almost every open-weights model in 2026 does — **HGX nodes are the right unit.** Incremental growth, air-cooling options, x86 software compatibility, one-node blast radius.
- NVL72 earns its complexity when you are committed to **frontier-scale multi-node serving** (wide-EP across dozens of GPUs as the *standing* deployment) or training. Note the Arm detail: every container in your air-gapped Harbor registry needs arm64 builds for Grace — a real cost in a disconnected environment.
- The middle path if demand outgrows single nodes: HGX B300 nodes + 800 Gb/s fabric + llm-d wide-EP across 2–4 nodes, per SOP §2.2 — no rack-scale re-platforming.

---

## 5. Interconnects: inside the node, across the rack, between nodes

### 5.1 NVLink 5 and NVSwitch (inside node / inside rack)

**NVLink** is NVIDIA's GPU-to-GPU link. Generation 5 (Blackwell): **18 links × 100 GB/s = 1.8 TB/s bidirectional per GPU** (900 GB/s each direction). **NVSwitch** is the crossbar chip that makes it all-to-all: every GPU talks to every other at full rate, on the HGX board or across the NVL72 backplane (the architecture scales to 576-GPU NVLink domains, but 72 is the largest shipping product).

NVSwitch generation 4 also does math: **NVLS / NVLink SHARP** executes reduction operations (the "add everyone's partial results" step of an all-reduce) inside the switch itself, which is part of why intra-node collectives beat naive expectations.

Why you care for inference: **tensor parallelism (TP)** — slicing every layer across 8 GPUs — inserts an all-reduce into *every layer, every token*. At TP=8 that communication sits on the critical path of your ITL. NVLink 5's bandwidth is what makes TP=8 cheap inside a node; the absence of NVLink *between* nodes is why the SOP says "don't extend TP across nodes."

### 5.2 Between nodes: NICs, DPUs, and the two fabrics

- **ConnectX-8 SuperNIC:** NVIDIA's current NIC, up to **800 Gb/s per port**, speaking either InfiniBand or Ethernet, PCIe Gen6, with an integrated PCIe switch. The standard build is one NIC port per GPU (8 per node) so each GPU has a private path into the fabric ("rail-optimized" wiring: rail *i* of every node connects to the same switch plane).
- **BlueField-3 DPU (data processing unit):** a NIC with its own Arm cores that offloads infrastructure work — north-south traffic, storage, tenant isolation, telemetry. In inference clusters it typically handles the *management/storage* network while ConnectX NICs carry GPU east-west traffic. On a small enclave fleet, BlueField is optional; know what it is, don't feel obliged to deploy it.
- **RDMA (remote direct memory access):** the property both fabrics must deliver — one machine's NIC writes directly into another machine's (GPU) memory with no CPU copies. **GPUDirect RDMA** extends that to GPU HBM. The SOP's wide-EP requirement (IBGDA over NVSHMEM — GPU-*initiated* RDMA) needs this working end-to-end.

The two fabric choices:

| Aspect | Quantum-X800 InfiniBand | Spectrum-X Ethernet (RoCEv2) |
|---|---|---|
| What it is | dedicated lossless fabric; Q3400 switch: 144 × 800 Gb/s ports, 115.2 Tb/s | Ethernet tuned for AI: RoCEv2 (RDMA over Converged Ethernet) + adaptive routing |
| Latency | ~1–2 µs; credit-based lossless by design | ~5–10 µs typical; losslessness must be engineered (PFC/ECN) |
| In-network compute | SHARP v4 offloads reductions to switches | no switch-side reductions |
| Operations | separate fabric, UFM tooling, IB-specific skills | standard Ethernet ops, multi-vendor, existing monitoring |
| Best fit | large training, latency-critical multi-node serving | most enterprise inference; integrates with the DC network |

Honest guidance for our context: for **single-node serving** (the SOP default), the inter-node fabric barely matters — it carries weights loading, monitoring, and API traffic. It starts to matter at **wide-EP/multi-node** and **P/D disaggregation** (prefill/decode split with KV-cache transfer over NIXL). Both work on either fabric; InfiniBand gives more headroom, Spectrum-X gives operational familiarity. What actually fails in the field is not the choice but the tuning — an unvalidated RoCE fabric with broken PFC (priority flow control) silently retransmits and halves your all-reduce bandwidth. Hence SOP §1: validate the fabric before you need multi-node.

> **Common pitfall:** buying 8 NICs per node and wiring them through one PCIe switch uplink or one ToR port. Bandwidth per GPU to the fabric only exists if the PCIe topology (check `nvidia-smi topo -m`) and switch wiring are rail-aligned end to end.

### 5.3 Verifying what you actually have

Two commands answer most "is the plumbing right?" questions on a delivered node:

```bash
nvidia-smi topo -m        # connectivity matrix: GPUs, NICs, CPU affinity
nvidia-smi nvlink -s      # per-link NVLink state and speed per GPU
```

In the `topo -m` matrix you want to see `NV18` between every GPU pair on a Blackwell HGX board (18 NVLink lanes — the full complement; `NV#` counts lanes, and fewer than 18 means links are down or the board is misconfigured). Each GPU should show a NIC reachable at `PXB`/`PIX` (through at most one PCIe switch) rather than `SYS` (across the CPU interconnect — a bandwidth cliff for GPUDirect RDMA). ASCII sketch of a healthy rail-optimized node:

```
GPU0 ── PCIe sw ── NIC0 ──► leaf switch, rail 0
GPU1 ── PCIe sw ── NIC1 ──► leaf switch, rail 1
 ...                                 ...
GPU7 ── PCIe sw ── NIC7 ──► leaf switch, rail 7
  └───────── NVLink 5 / NVSwitch: all-to-all, 1.8 TB/s per GPU ─────────┘
```

(On HGX B300 the "PCIe sw" boxes are inside the ConnectX-8 NICs themselves.)

---

## 6. Power and cooling reality

Numbers you plan facilities around (mid-2026 figures):

| Unit | Power | Cooling |
|---|---|---|
| B200 GPU | 1,000 W (HGX) / 1,200 W (GB200) | air possible / liquid |
| B300 GPU | 1,100–1,400 W (chassis-dependent) | air possible (DGX B300) / liquid |
| DGX/HGX B200 node (8 GPU) | ~14.3 kW max | air or direct liquid (vendor) |
| DGX B300 node (8 GPU) | ~14.5 kW max | air-cooled MGX chassis |
| GB300 NVL72 rack | ~135 kW TDP, up to ~150+ kW peak | 100% direct liquid, mandatory |

Facility implications, in plain terms:

- **A single 8-GPU node now draws what a whole legacy rack used to.** Four DGX B300s in one rack (NVIDIA's SuperPOD reference density) exceeds 50 kW/rack — most legacy enterprise data-center rows were built for 5–15 kW/rack. Your constraint is usually power distribution and heat rejection, not floor space.
- **Air-cooled 8-GPU nodes are real but loud and marginal** — they need cold-aisle containment, high CFM, and tolerate no inlet-temperature excursions. Direct liquid cooling (cold plates on GPUs, CDU — coolant distribution unit — exchanging heat to facility water) is the comfortable path at B300 power and the *only* path for NVL72.
- **NVL72 racks bring plumbing requirements:** facility water at roughly 30–40 °C supply to the CDU, filtration (~50 µm), conductivity monitoring, corrosion inhibitors, quick-disconnects at every tray, and ~90% of heat captured to liquid. They also weigh on the order of 1.5 tonnes — check floor loading.
- **Power quality matters:** Blackwell workloads swing load fast (synchronized GPU power ramps). Size PDUs/busbars and UPS for peak, not average, and confirm with the vendor's EDP (electrical design power) guidance.
- Budget **PUE/cooling overhead** on top: a 14 kW node is ~17–20 kW at the meter in a typical air-cooled room; liquid cooling improves that materially.

> **Common pitfall:** accepting nodes in a room that can power them but not cool them at sustained load. The failure is invisible in acceptance smoke tests and shows up as `HW Thermal Slowdown` clock throttling under production load — see §9.4. Run the burn-in at full power *in the production room*.

---

## 7. MIG and time-slicing: sharing one GPU

**MIG (Multi-Instance GPU)** partitions one physical GPU into up to **7 hardware-isolated instances**, each with its own SMs, HBM slice, and cache — real isolation, like SR-IOV for GPUs. On B200, profiles run from `1g.23gb` (1/7 of compute, 23 GB) up to the full GPU; a B200 can host seven 23 GB instances. Reconfiguring MIG requires draining the GPU.

**Time-slicing** is the soft alternative: multiple processes share the whole GPU by context-switching. No isolation — one noisy process degrades all — but no reconfiguration either. (**MPS**, Multi-Process Service, is a finer-grained cousin allowing concurrent kernels from multiple processes.)

When each matters for LLM inference:

- **Production LLM serving on B200/B300: usually neither.** A production vLLM replica wants the *whole* GPU — every free GB of HBM becomes KV cache (vLLM's `--gpu-memory-utilization 0.90` claims it deliberately). Carving a B300 into sevenths to serve seven tiny models is almost always worse than serving those models on one shared vLLM instance or on cheaper GPUs.
- **MIG is genuinely useful for:** dev/test enclaves (give each engineer a `1g.23gb` slice for notebook work), CI runners for the eval gate, serving small auxiliary models (embedding/reranker models, guard models, draft models) with hard isolation from production, and multi-tenant chargeback where isolation is a policy requirement.
- **Time-slicing is for** best-effort dev clusters only. Never in the production serving pool: it destroys latency SLOs by design.
- A pragmatic middle path for auxiliary models: run them as *separate vLLM instances pinned to distinct GPUs* rather than MIG slices, unless GPU count forces sharing.

---

## 8. The software stack under vLLM

The layers between metal and serving engine, bottom-up:

1. **GPU driver.** Blackwell (`sm_100`/`sm_103`, compute capability 10.x) requires the **R570 driver branch or newer**. Mid-2026 fleets typically run R570/R580-era drivers. The SOP rule applies: the enclave driver version is pinned to whatever your frozen vLLM container was validated against.
2. **CUDA.** **CUDA 12.8 (Jan 2025) is the floor for Blackwell** — the first toolkit that can compile for `sm_100`. Anything built against older CUDA fails with `no kernel image is available for execution on the device`. CUDA 13.x is current in mid-2026; vLLM's official images ship with a Blackwell-capable toolkit — verify the container's CUDA matches your driver's supported range (`nvidia-smi` shows the max CUDA version the driver supports).
3. **Fabric Manager (`nvidia-fabricmanager`).** On any NVSwitch system (every HGX/DGX/NVL72), this privileged daemon configures the NVLink fabric and admits GPUs into the shared memory domain. **It must run, and its version must exactly match the driver version.** Symptom of it missing/broken: CUDA init fails with "system not initialized," or P2P bandwidth collapses to PCIe speeds.
4. **NCCL (NVIDIA Collective Communications Library).** The library every framework (PyTorch, vLLM) uses for all-reduce/all-gather/all-to-all across GPUs. It auto-discovers topology (NVLink vs PCIe vs NIC) and picks algorithms (ring, tree, NVLS). Blackwell support landed around NCCL 2.25 (early 2025); your vLLM container pins its own NCCL — record that version in the release bundle per SOP §4.1.
5. **DCGM (Data Center GPU Manager).** Health monitoring and diagnostics daemon; source of the Prometheus metrics (`dcgm-exporter`) in SOP §7 and of the `dcgmi diag` acceptance tests in §9.
6. On Kubernetes (SOP Tier 3), the **GPU Operator** installs and lifecycle-manages all of the above as containers — one fewer hand-managed layer, one more thing to mirror into Harbor.

> **Common pitfall:** letting `pip install torch` (or any dependency update) replace the container's Blackwell-built PyTorch with a generic wheel. The generic wheel may lack `sm_100` kernels and dies at runtime, not install time. In the enclave, production never pip-installs (SOP rule) — this is one of the concrete reasons why.

---

## 9. Node acceptance and burn-in

New node, returned-from-repair node, or post-driver-upgrade node: it does not carry production traffic until it passes this gauntlet. GPUs ship with real infant-mortality rates, and a marginal HBM stack or NVLink lane will corrupt long-running inference in ways that look like model bugs.

### 9.1 DCGM diagnostics

`dcgmi diag` runs NVIDIA's built-in test suites at four levels (durations from DCGM docs, 8-GPU systems):

| Level | Command | Duration (8 GPU) | What it adds |
|---|---|---|---|
| 1 | `dcgmi diag -r 1` | < 2.5 s | software stack + PCIe sanity |
| 2 | `dcgmi diag -r 2` | < 10.5 min | + GPU memory, memory bandwidth |
| 3 | `dcgmi diag -r 3` | < 35 min | + targeted stress, targeted power, NVBandwidth, NCCL tests |
| 4 | `dcgmi diag -r 4` | < 2.25 h | + memtest, pulse test (deep HBM + power-transient stress) |

Practice: **`-r 1` before every return-to-service; `-r 3` for acceptance; `-r 4` as the core of burn-in and for post-mortems** on suspect nodes. Run a burn-in of at least 24–48 h total (looped `-r 4` plus a realistic vLLM load test) in the production room at production cooling.

### 9.2 NCCL bandwidth validation

`nccl-tests` (github.com/NVIDIA/nccl-tests — mirror it into the enclave) measures collective bandwidth exactly the way vLLM will use it:

```bash
# intra-node: all 8 GPUs, sweep 8 B → 8 GiB
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8

# inter-node (2 nodes × 8 GPUs, via mpirun/srun):
mpirun -np 16 -H nodeA:8,nodeB:8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

Read the **`busbw` (bus bandwidth)** column at large message sizes (≥ 1 GiB). Interpretation:

- NVLink 5 is 1.8 TB/s bidirectional per GPU (900 GB/s per direction). All-reduce `busbw` reflects per-direction algorithm bandwidth, not the bidirectional headline — on 8×H100 (450 GB/s/direction) healthy systems report roughly 370–480 GB/s; on 8×B200/B300 expect roughly double that at large sizes with the NVLS algorithm. Public per-site figures vary with NCCL version and settings; treat any figure under **~80% of your own fleet's median** as a fault, and under ~50% as a topology/Fabric-Manager problem (traffic falling back to PCIe).
- **The actionable discipline is the golden number:** record `busbw` at 1 GiB for every node at acceptance in the config repo, and re-run + compare after every driver/NCCL change. Absolute thresholds age; regressions against your own baseline do not.
- Inter-node, `busbw` is capped by NIC bandwidth: 8 × 400 Gb/s ≈ 400 GB/s per node, 8 × 800 Gb/s (ConnectX-8) ≈ 800 GB/s. Healthy fabrics reach 80–90%+ of that; a single flapping link or misconfigured PFC shows up here first.

### 9.3 Memory (HBM/ECC) checks

ECC (error-correcting code) memory fixes single-bit errors transparently; the counters tell you when a stack is degrading. Blackwell (like Ampere/Hopper) uses **row remapping**: bad HBM rows are permanently swapped for spare rows (takes effect on GPU reset).

```bash
nvidia-smi -q -d ECC,ROW_REMAPPER   # per-GPU ECC counters + remapped-rows block
dcgmi dmon -e 393,394,395,396      # DCGM row-remap fields (uncorrectable/correctable/pending/failed)
```

Acceptance policy:

- **Volatile uncorrectable ECC errors during burn-in:** fail the node; investigate.
- **`Remapped Rows: Pending = Yes`:** the GPU wants a reset to apply a remap — drain, reset, re-test. Fine occasionally; a *pattern* on one GPU is a dying stack.
- **`Remapping Failure Occurred = Yes`:** spare rows exhausted or remap failed — this is RMA (return merchandise authorization) territory per NVIDIA's memory-error policy. Do not return the node to service.
- Track remap counts over time in Prometheus; a GPU accumulating uncorrectable-driven remaps is telling you its future.

### 9.4 Common failure modes cheat sheet

| Symptom | Where you see it | Likely cause / action |
|---|---|---|
| `Xid 63/64` in dmesg; pending row remap | `nvidia-smi -q`, DCGM | HBM row degradation → drain, reset; RMA if recurring or remap fails |
| `Xid 48` (DBE) or app crash w/ ECC error | dmesg, DCGM | uncorrectable memory error → reset; RMA if repeats |
| Clocks sag under load; `HW Thermal Slowdown` | `nvidia-smi -q -d PERFORMANCE` | cooling shortfall (airflow, cold-plate flow, room) → fix facility, not the GPU |
| NVLink link down; TP throughput halves | `nvidia-smi nvlink -s`, FM logs | reseat/replace module or baseboard; never run production with a degraded NVLink on a TP node |
| "system not initialized" on CUDA init | app logs | Fabric Manager not running or version-mismatched with driver |
| `busbw` far below golden number | nccl-tests | topology misdetect, PCIe fallback, PFC/ECN misconfig (RoCE), or a slow NIC/link |
| `Xid 79` (GPU fell off the bus) | dmesg | PCIe/power event → reseat, check power delivery; RMA if recurring |

> **Common pitfall:** running acceptance with the node idle-cool and calling it done. Thermal and power-transient faults only appear at sustained full load — which is why `-r 4` (pulse test) and a real vLLM load test at production concurrency belong in burn-in.

### 9.5 A minimal acceptance checklist (copy into the config repo)

Every step records output into the node's acceptance file; a node without a completed file carries no traffic.

```bash
NODE=$(hostname); OUT=/var/log/acceptance/${NODE}-$(date +%F)

# 1. Identity + stack versions (must match the frozen release bundle)
nvidia-smi --query-gpu=name,serial,vbios_version,driver_version --format=csv | tee -a $OUT
nvidia-smi | grep "CUDA Version" | tee -a $OUT
systemctl is-active nvidia-fabricmanager | tee -a $OUT      # must be "active"

# 2. Topology sanity: NV18 everywhere, NICs at PIX/PXB
nvidia-smi topo -m | tee -a $OUT
nvidia-smi nvlink -s | grep -ci "inactive" | tee -a $OUT     # expect 0

# 3. Health baseline: ECC + row remap must be clean
nvidia-smi -q -d ECC,ROW_REMAPPER | tee -a $OUT

# 4. Diagnostics: acceptance level, then burn-in level
dcgmi diag -r 3 | tee -a $OUT                                # ~35 min, must be all PASS
dcgmi diag -r 4 | tee -a $OUT                                # ~2.25 h, burn-in core

# 5. Collectives: record the golden number (busbw @ >=1 GiB)
./all_reduce_perf -b 8 -e 8G -f 2 -g 8 | tee -a $OUT

# 6. Realistic load: 2-4 h vLLM + guidellm at production concurrency,
#    then re-check step 3 for new ECC/remap events and throttle reasons:
nvidia-smi -q -d PERFORMANCE | grep -A2 "Clocks Event Reasons" | tee -a $OUT
```

Acceptance passes when: all diag levels PASS, zero inactive NVLinks, zero new uncorrectable ECC/remap events after the load phase, no thermal-slowdown reasons under sustained load, and the `busbw` golden number is within tolerance of the fleet median. File the golden number with the node record — every future driver or NCCL upgrade re-runs steps 4–6 and diffs against it (SOP runbook 1).

---

## 10. Which SKU for which workload

Assuming our environment: air-gapped, vLLM, agentic traffic, NVFP4-default (SOP Appendix A).

| Workload | Best fit | Why |
|---|---|---|
| Interactive agent pool, long contexts, many sessions | **HGX/DGX B300** | 288 GB/GPU = most KV cache per node; 2× softmax helps long-context attention; dense-FP4 uplift |
| Dense models ≤ ~200B (NVFP4), batch/offline, evals | **HGX B200** | cheaper per node; capacity/bandwidth ample for the job |
| Big MoE (400–700B-class) in NVFP4, single node | **HGX B300** (B200 workable) | fits in 2.1–2.3 TB node HBM with headroom for KV |
| Frontier/trillion-class MoE, standing multi-node wide-EP | **GB300 NVL72** | 72-GPU NVLink domain keeps expert all-to-all off the NIC fabric |
| Dev, CI, eval runners, small aux models (embedders, guards, drafts) | **B200 + MIG** (`1g.23gb`+) or older H200 nodes | isolation without burning a production-grade GPU per user |
| Existing H200 fleet | keep for FP8/BF16 and aux duty | no FP4 hardware; 141 GB/4.8 TB/s still respectable — don't put it in the NVFP4 fast path |

Sizing heuristics to remember:

- **Weights first:** parameter count × bytes/param (NVFP4 ≈ 0.5 B + scale overhead ≈ 0.55–0.6 B effective; FP8 ≈ 1 B) must fit in the node's usable HBM with ≥ 30–40% left for KV cache, or you'll serve at toy concurrency.
- **Then KV:** per-token KV bytes × context length × concurrent sequences (FP8 KV halves it — SOP default). This, not weights, is what the B300's extra 96 GB per GPU buys you.
- **Then bandwidth:** aggregate HBM bandwidth ÷ bytes-touched-per-token bounds total decode tokens/s per node. B200 and B300 are equal here — B300 wins on *capacity* and *attention math*, not bytes/s.

---

## Study questions

1. **Why is decode memory-bandwidth-bound while prefill is compute-bound?**
   Answer: Prefill reuses each weight across every prompt token in one parallel pass (high FLOPs per byte); decode must stream all weights and KV cache from HBM to produce each single token (about 1–2 FLOPs per byte at low batch), far below the GPU's ~560 FLOP/byte FP8 ridge point.

2. **A colleague says "B300 has the same 8 TB/s bandwidth as B200, so decode won't get faster — why buy it?" What's the right answer for our agentic workload?**
   Answer: Correct on single-stream decode speed, but B300's 288 GB (vs 192) holds far more KV cache, so each node sustains many more concurrent long-context sessions (higher goodput), and its doubled softmax/SFU throughput speeds up long-context attention; dense FP4 is also ~1.5× for prefill.

3. **What is NV-HBI and what do you have to do operationally about the two dies in a Blackwell package?**
   Answer: NVIDIA High-Bandwidth Interface, the 10 TB/s die-to-die link fusing two reticle-limit dies into one coherent GPU. Operationally: nothing — software sees a single GPU with unified memory.

4. **Why does the same "B200" show 9 PFLOPS dense FP4 in one datasheet and 10 in another?**
   Answer: SKU/power variation — HGX B200 boards run 1,000 W per GPU (9 PFLOPS dense FP4, ~180 GB usable), while GB200 NVL72 runs the chip at 1,200 W (10 PFLOPS). Always spec-check the exact platform.

5. **What makes NVFP4 more accurate than "naive" 4-bit quantization?**
   Answer: Two-level microscaling — each block of 16 E2M1 values shares an FP8 scale factor, plus an FP32 per-tensor scale — so the tiny 4-bit dynamic range is re-centered per block; the second-gen Transformer Engine manages this in hardware.

6. **When would you pick an NVL72 rack over HGX nodes, and what hidden software cost comes with it?**
   Answer: When a standing workload needs a model served across far more than 8 GPUs at NVLink speed (trillion-class MoE, wide-EP as the permanent deployment). Hidden cost: Grace CPUs are Arm, so every container in the air-gapped registry needs arm64 builds; plus mandatory liquid cooling at ~120–150 kW/rack.

7. **Quantum-X800 InfiniBand vs Spectrum-X Ethernet — what actually differs and when does the choice matter?**
   Answer: InfiniBand: ~1–2 µs latency, lossless by design, SHARP in-switch reductions; Spectrum-X: RoCEv2 on standard Ethernet ops, ~5–10 µs, losslessness engineered via PFC/ECN. It only becomes critical for multi-node serving (wide-EP, P/D disaggregation); single-node serving barely exercises it.

8. **What does Fabric Manager do, and what are the two classic symptoms when it's broken?**
   Answer: It configures NVSwitch/NVLink fabric membership on HGX/DGX/NVL systems. Symptoms: CUDA fails with "system not initialized," or GPU peer-to-peer bandwidth silently collapses to PCIe speeds. Its version must exactly match the driver.

9. **What is the CUDA/driver floor for Blackwell and what error do you get when you miss it?**
   Answer: CUDA 12.8+ with an R570-or-newer driver (compute capability 10.0, `sm_100`). Older-built binaries fail at runtime with "no kernel image is available for execution on the device."

10. **In `nccl-tests`, what number do you read and how do you judge it?**
    Answer: The `busbw` column at large message sizes (≥1 GiB). Judge it against your own recorded golden number per node from acceptance; investigate anything below ~80% of fleet median (per-direction NVLink 5 bandwidth is 900 GB/s per GPU, so 8-GPU all-reduce busbw lands in the several-hundred-GB/s range, roughly 2× healthy H100 results).

11. **A GPU shows `Remapped Rows: Pending = Yes` — what do you do? What if `Remapping Failure Occurred = Yes`?**
    Answer: Pending: drain the node, reset the GPU (remap applies on reset), re-run diagnostics — occasional occurrences are normal. Remapping failure: spare rows are exhausted/failed; the GPU meets RMA criteria and must not return to service.

12. **Why is MIG mostly irrelevant for the production agent-serving pool but useful elsewhere?**
    Answer: A production vLLM replica wants the entire GPU — spare HBM becomes KV cache for concurrency. MIG shines for dev/test slices, CI/eval runners, and small auxiliary models needing hard isolation (e.g., `1g.23gb` slices on B200).

---

## Sources

Primary (NVIDIA):

- Inside NVIDIA Blackwell Ultra (architecture blog): https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
- NVIDIA HGX platform page (HGX B200/B300 specs): https://www.nvidia.com/en-us/data-center/hgx/
- NVIDIA GB200 NVL72 page: https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- NVIDIA DGX B300 page and user guide: https://www.nvidia.com/en-us/data-center/dgx-b300/ , https://docs.nvidia.com/dgx/dgxb300-user-guide/
- Blackwell Compatibility/Tuning Guides (CUDA 12.8+, sm_100): https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
- DCGM Diagnostics (run levels, durations): https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html
- GPU memory error management / row remapping: https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html
- Fabric Manager user guide: https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html
- MIG user guide (supported profiles): https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html
- Quantum-X800 platform: https://www.nvidia.com/en-us/networking/products/infiniband/quantum-x800/
- GB200 NVL multi-node tuning guide (NCCL): https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html
- nccl-tests: https://github.com/NVIDIA/nccl-tests

Secondary (analysis/teardowns, used for cross-checks):

- ServeTheHome — HGX B300 NVL16 teardown: https://www.servethehome.com/the-nvidia-hgx-b300-nvl16-is-massively-different/
- ServeTheHome — ConnectX-8 SuperNIC detail: https://www.servethehome.com/nvidia-connectx-8-supernic-pcie-gen6-800g-nic-detailed/
- Tom's Hardware — Blackwell Ultra B300 announcement (15 PFLOPS dense FP4, 288 GB): https://www.tomshardware.com/pc-components/gpus/nvidia-announces-blackwell-ultra-b300-1-5x-faster-than-b200-with-288gb-hbm3e-and-15-pflops-dense-fp4
- Chips and Cheese — B200 microarchitecture analysis: https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut
- Microbenchmarking NVIDIA Blackwell (arXiv): https://arxiv.org/html/2512.02189v1
- Lenovo Press — ThinkSystem HGX B200 1000W product guide: https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu
- Lenovo Press — GB300 NVL72 rack product guide: https://lenovopress.lenovo.com/lp2357-lenovo-nvidia-gb300-nvl72-rack-scale-ai
- Sunbird — GB300 NVL72 power analysis: https://www.sunbirddcim.com/blog/how-much-power-does-nvidia-gb300-nvl72-need
- Crusoe — validating ECC and row-remap status: https://support.crusoecloud.com/hc/en-us/articles/50948432705563-How-To-Validate-GPU-ECC-and-Row-Remap-Status-Using-nvidia-smi
- vLLM GPU installation docs: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/

*Companion docs in this repo: [SOP](../SOP-production-ai-model-deployment.md), [PRIMER](../PRIMER.md).*
