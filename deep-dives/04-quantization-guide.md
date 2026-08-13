# Quantization for Production Inference: NVFP4, FP8, and Everything Below

**Series:** Deep dives for the [deployment SOP](../SOP-production-ai-model-deployment.md). Read the [PRIMER](../PRIMER.md) first if terms like KV cache or prefill are new.
**Context:** on-prem, air-gapped, NVIDIA B200/B300 (Blackwell), vLLM serving engine, agentic workloads.
**Freshness:** researched August 2026 against primary sources (vLLM v0.26.x era, LLM Compressor current docs, NVIDIA Model Optimizer main branch). Anything fast-moving is dated inline.

---

## Key takeaways

- Quantization stores a model's numbers in fewer bits. On Blackwell it is not just a memory trick: B200 tensor cores run FP4 math at 2x the speed of FP8 and 4x the speed of BF16, so a well-quantized model is smaller *and* faster.
- **NVFP4** is NVIDIA's 4-bit floating-point format: FP4 values in blocks of 16, each block scaled by an FP8 (E4M3) factor, plus one FP32 scale per tensor. The small block and high-precision scale are why it loses far less accuracy than the older MXFP4 format (block of 32, power-of-two scales).
- The stack of defaults, matching the SOP: **NVFP4 for production on Blackwell → FP8 when no NVFP4 checkpoint exists yet → BF16 only as the accuracy baseline.** Add `--kv-cache-dtype fp8` for agentic serving and validate it.
- Naming decodes as WxAy: **W4A16** means 4-bit weights with 16-bit activations (memory savings, math still runs in 16-bit); **W8A8/W4A4** quantize activations too, which is what unlocks the fast low-precision tensor cores.
- **FP8 dynamic (`FP8_DYNAMIC`) needs no calibration data at all** and is the safest "day-0" format for a new model drop. NVFP4 and INT8 W8A8 need a calibration dataset, and a bad calibration set is the most common self-inflicted quality bug.
- Two toolchains matter for you: **NVIDIA TensorRT Model Optimizer (ModelOpt)** (`hf_ptq.py --qformat nvfp4`, exports a unified Hugging Face checkpoint) and **vLLM's LLM Compressor** (`oneshot()` with a recipe, saves `compressed-tensors` format). Both load directly into vLLM.
- Prefer **pre-quantized checkpoints** from the `nvidia` and `RedHatAI` Hugging Face organizations over quantizing yourself; verify trust by checking the producing tool, the published eval table, and your own enclave eval gate.
- Perplexity is a smoke test, not an acceptance test. Correlation between perplexity and downstream task accuracy under compression ranges from strong to almost nonexistent depending on the task, so the gate must be task-level and agentic.
- Quantization error concentrates where you least want it: small models, long generations from reasoning models, extreme context lengths, and activation outlier channels. Your eval gate must probe exactly those places.
- **GGUF** (the llama.cpp checkpoint format) is a desktop/edge ecosystem. In vLLM it is an experimental out-of-tree plugin and is not a production path; do not let GGUF files cross into the enclave as serving artifacts.

---

## 1. Number formats from first principles

### 1.1 Bits, sign, exponent, mantissa

Every number in a model — weights, activations, KV-cache entries — is stored in some binary format. There are two families:

- **Integer formats (INT8, INT4)** store evenly spaced whole numbers. INT8 covers −128 to 127; INT4 covers −8 to 7. To represent real-valued weights, you pair the integers with a *scale factor*: `real_value ≈ integer × scale`. All the format's precision is uniform across its range.
- **Floating-point formats (FP32, BF16, FP8, FP4)** split their bits into a **sign** bit (positive or negative), an **exponent** (which power of two you are near — this sets *range*), and a **mantissa** (the detail within that power of two — this sets *precision*). Floating point puts more precision near zero and less far away, which happens to match how neural-network weights are distributed: most values are tiny, a few are large.

A useful mental model: the exponent is the ruler you pick up (millimeter ruler vs. meter stick), and the mantissa is how fine the tick marks on that ruler are.

The formats you will actually meet, written as E{exponent bits}M{mantissa bits}:

| Format | Bits | Layout | Max normal value |
|---|---|---|---|
| FP32 | 32 | 1 sign, 8 exp, 23 mantissa | ~3.4 × 10^38 |
| BF16 | 16 | 1 sign, 8 exp, 7 mantissa | ~3.4 × 10^38 |
| FP16 | 16 | 1 sign, 5 exp, 10 mantissa | 65,504 |
| FP8 E4M3 | 8 | 1 sign, 4 exp, 3 mantissa | 448 |
| FP8 E5M2 | 8 | 1 sign, 5 exp, 2 mantissa | 57,344 |
| FP4 E2M1 | 4 | 1 sign, 2 exp, 1 mantissa | 6 |

Notice the trade in FP8: **E4M3** spends bits on precision (3 mantissa bits, but tops out at ±448), while **E5M2** spends bits on range (up to ±57,344, but only 2 mantissa bits — it is literally FP16 with the bottom 8 mantissa bits cut off). The rule of thumb the whole industry converged on: E4M3 for anything you store and multiply during inference (weights, activations, KV cache), E5M2 mainly for gradients in training, where occasional huge values matter more than fine detail. When someone says "FP8" in an inference context in 2026, they almost always mean E4M3 with a per-tensor or finer-grained scale factor handling the range problem.

**FP4 (E2M1) can represent exactly sixteen values:** 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6 (with two encodings of zero). That is absurdly coarse on its own — everything hinges on the scale factors, which brings us to block formats.

### 1.2 Block scaling and microscaling: why the scale factor is the real format

With only 16 representable values, a single scale factor per tensor would be hopeless: one large outlier weight would force a huge scale, and 99.9% of the values would get squashed into two or three of the sixteen buckets. The fix is **block scaling** (also called *microscaling*): chop the tensor into small blocks and give each block its own scale factor, so each block's ruler is fitted to its local values.

The two 4-bit block formats you will see:

| Property | MXFP4 (OCP Microscaling) | NVFP4 (NVIDIA) |
|---|---|---|
| Element format | FP4 (E2M1) | FP4 (E2M1) |
| Block size | 32 values | 16 values |
| Block scale format | E8M0 (power of two only) | FP8 E4M3 (fractional) |
| Second-level scale | none | FP32 per tensor |
| Effective bits per value | 4.25 | ~4.5 |

Both facts about NVFP4's design matter and were verified against NVIDIA's own technical material (developer blog, mid-2025 onward; Transformer Engine docs):

1. **Smaller blocks (16 vs. 32).** A block's scale is set by its largest value. One outlier therefore "contaminates" its whole block by forcing a coarse scale on its neighbors. Halving the block size halves the blast radius of every outlier and doubles the opportunities to match local dynamic range.
2. **Fractional scales (E4M3 vs. E8M0).** MXFP4's E8M0 scale is an 8-bit exponent with no mantissa — scales can only be powers of two (…, 0.5, 1, 2, 4, …), so the fitted ruler can be off by up to ~2x. NVFP4's E4M3 scale can be fractional (e.g., 1.375), fitting the block much more tightly. In NVIDIA's published example, fitting a block with an E4M3 scale gave a mean squared quantization error of 0.08 versus 0.72 with an E8M0 scale. The FP32 per-tensor scale on top restores global dynamic range that the small FP8 block scales cannot cover alone.

The price of NVFP4's quality is a slightly fatter format (one FP8 scale per 16 values ≈ 0.5 extra bits per value → ~4.5 bits effective vs. MXFP4's 4.25) and a hard dependency on hardware that understands it.

One prominent MXFP4 sighting worth knowing: OpenAI's open-weights **gpt-oss** models (Aug 2025) ship their Mixture-of-Experts (MoE — a model built from many small "expert" sub-networks, few active per token) weights *natively* in MXFP4, at 4.25 bits per parameter, covering over 90% of parameters. They get away with the weaker format because the models were *trained* with the quantization in place (see §4 on QAT) rather than converted afterward.

### 1.3 Why hardware support decides everything

A quantized format only speeds you up if the GPU's **tensor cores** (the matrix-multiply units) can execute math directly in that format. Otherwise the weights must be *dequantized* to 16-bit before every multiply — you keep the memory savings but pay a conversion cost instead of getting a speedup.

Blackwell is the first NVIDIA generation with **FP4 tensor cores**. The published dense-math throughput ladder (per GPU, no sparsity):

- **B200:** ~9 PFLOPS FP4 · ~4.5 PFLOPS FP8/INT8 · ~2.25 PFLOPS FP16/BF16. (PFLOPS = 10^15 floating-point operations per second.)
- **B300 (Blackwell Ultra):** ~14–15 PFLOPS dense FP4, roughly 1.5x B200's FP4 rate, plus 288 GB of HBM (high-bandwidth memory) versus 192 GB.

So each halving of precision roughly doubles peak math *and* halves bytes moved from memory — and inference decode is usually memory-bandwidth-bound, which is why the bytes matter as much as the FLOPS. Hopper (H100/H200) has FP8 tensor cores but no FP4; on Hopper, an NVFP4 checkpoint can still run, but vLLM executes it as weight-only (dequantize to 16-bit, then multiply — W4A16 style), keeping the memory win and losing the math win. This is why the SOP's format default is tied to the hardware generation: **NVFP4 is the point of owning Blackwell.**

---

## 2. What quantization error actually does to an LLM

Quantization replaces every number with its nearest representable neighbor. Each individual error is tiny; the question is how millions of tiny errors compound through 60–100 transformer layers and thousands of generated tokens. In practice the damage is not uniform — it concentrates in specific, predictable places.

### 2.1 Perplexity vs. task accuracy

**Perplexity** measures how "surprised" a model is by held-out text (lower is better). It is the traditional quantization-paper metric because it is cheap and smooth. It is also a poor acceptance test: one 2026 study of compression degradation found the correlation between perplexity and downstream task accuracy ranged from r = 0.77 on one benchmark (HellaSwag) down to r = 0.20 on another (BoolQ). The mechanism is simple — task accuracy depends on the *ranking* of candidate answers, while perplexity measures absolute per-token probability. Compression can worsen all probabilities equally (perplexity worsens, accuracy unchanged) or selectively disrupt exactly the tokens that decide an answer (perplexity barely moves, accuracy craters).

Practical rule: use perplexity as a **tripwire** (a jump of more than a few percent means the quantization job itself is broken — wrong scales, wrong ignore list) and use task-level and agentic evals as the **gate** (§8).

### 2.2 Where the damage concentrates

- **Outlier channels.** LLM activations have a well-documented pathology: a handful of hidden-dimension channels carry values 10–100x larger than the rest, concentrated in a small fraction of channels (down-projection layers are the worst offenders). Naive quantization lets these few channels dictate the scale and destroy precision for everyone else. Every serious method is, at heart, an outlier-management strategy: SmoothQuant migrates the outliers from activations into weights; AWQ protects the weights that outlier activations touch; NVFP4 shrinks the blast radius with 16-value blocks (§1.2).
- **Long generations and reasoning models.** An empirical study of quantized reasoning models (arXiv 2504.04823, 2025) found reasoning tasks degrade more than short-answer tasks: errors compound over thousand-token chains of thought, and harder problems and smaller models degrade first. Weight-only 4-bit (W4A16) was relatively safe; aggressive W4A4 without a strong format was harmful. For your agentic workloads — long tool-calling loops are exactly long generations — this is the failure mode to probe hardest.
- **Long context.** FP8 KV cache holds up well at long context on validated paths (94–99% accuracy recovery out to 128k–1M tokens in the April 2026 vLLM study, §7), but *lower-bit* KV quantization errors accumulate with sequence length — a 3-bit KV experiment in the same ecosystem showed ~30% relative degradation concentrated at 128k–256k contexts. Long-context retrieval probes at your real context lengths belong in the gate (the SOP already requires them).
- **Small models.** A 7B model has less redundancy to absorb error than a 70B model. Red Hat / vLLM's NVFP4 study (Feb 2026) quantifies it: ~99% of the BF16 baseline recovered for 70B–235B models, 97–99% at ~30B, but 95–98% for 7–14B models. Below ~10B, treat NVFP4 as "verify, don't assume," and be readier to fall back to FP8.
- **MoE models — a nuance worth stating carefully.** You will hear "MoE is more sensitive to quantization." The evidence as of mid-2026 is mixed: NVIDIA and Red Hat both report MoE models (DeepSeek-class) as exceptionally robust under NVFP4, and at least one MXFP4 ablation measured *smaller* average drops on MoE than on dense models. The real, verified MoE risk is different: **calibration coverage.** Each expert sees only the slice of tokens routed to it, so rarely-activated experts may see almost no calibration data, producing garbage statistics for exactly those weights. The fix is MoE-aware calibration (both ModelOpt and LLM Compressor have MoE-specific handling) and pre-quantized checkpoints from teams who did that work — another reason the SOP prefers published ModelOpt checkpoints.

### 2.3 The quality numbers to anchor on

From NVIDIA's NVFP4 introduction (developer blog), DeepSeek-R1-0528 quantized FP8 → NVFP4 with ModelOpt:

| Benchmark | FP8 | NVFP4 |
|---|---|---|
| MMLU-Pro | 85% | 84% |
| GPQA Diamond | 81% | 80% |
| LiveCodeBench | 77% | 76% |
| SciCode | 40% | 40% |
| Math-500 | 98% | 98% |
| AIME 2024 | 89% | 91% |

That is a frontier-scale MoE losing ≤1 point everywhere (and noise going both directions). This — with the 3.5x memory reduction versus FP16 and 1.8x versus FP8 — is the factual basis for the SOP's "NVFP4 default" stance. The caveat from §2.2 stands: these are big models; your 8B utility model may not match this.

---

## 3. Weight-only vs. weight-and-activation quantization (the WxAy naming)

Quantization schemes are named **W{weight bits}A{activation bits}**. The distinction determines both what breaks and what speeds up:

| Naming | What is quantized | What you gain | Example methods |
|---|---|---|---|
| W4A16 / W8A16 | weights only | memory + bandwidth; math stays 16-bit | AWQ, GPTQ, NVFP4-on-Hopper fallback |
| W8A8 | weights + activations | memory + 8-bit tensor-core math | FP8, INT8 SmoothQuant |
| W4A4 | weights + activations | memory + 4-bit tensor-core math | NVFP4, MXFP4 |
| KV8 (informal) | KV cache only | 2x KV capacity | `--kv-cache-dtype fp8` |

Two things to internalize:

1. **Weights are easy; activations are hard.** Weights are static — you can study them offline, pick perfect scales per channel or per block, even iteratively correct errors (GPTQ). Activations are computed fresh per token, contain the outlier channels of §2.2, and must be quantized on the fly with scales chosen either dynamically per token (accurate, small overhead) or statically from calibration data (fast, riskier). This is why W4A16 was the dominant community format for years and why W8A8/W4A4 needed either clever algorithms (SmoothQuant) or better formats (block-scaled FP8/FP4) to become safe.
2. **Only "A" unlocks tensor cores.** A W4A16 model still does BF16 math — great for fitting a model on fewer GPUs and for memory-bound decode, but no FP4-tensor-core speedup. NVFP4 as deployed on Blackwell is W4A4 (weights and activations both FP4, activations scaled per 16-value block on the fly); FP8 checkpoints are W8A8. On your hardware, W4A16 formats like AWQ are a compatibility tool for older GPUs, not the plan.

The KV cache is a separate, composable axis: you can serve an NVFP4 model with an FP8 KV cache (the SOP's standard agentic combo) — see §7.

---

## 4. PTQ vs. QAT, and the calibration dataset

### 4.1 Post-training quantization (PTQ)

**PTQ** takes a finished model and converts it, typically in minutes-to-hours on one node. All the toolchain workflows in §6 are PTQ. Most PTQ methods need a **calibration dataset**: a few hundred representative text samples run through the model so the tool can observe real activation ranges and pick scales. The weights don't change (except in error-correcting methods like GPTQ, which nudges remaining weights to compensate for rounding); only the numeric representation does.

Calibration realities, verified against tool defaults:

- **FP8 dynamic needs zero calibration data** — weights get static per-channel scales computed from the weights themselves, activations get dynamic per-token scales at runtime. This is why FP8 is the day-0 format in the SOP's fast path: no dataset debate, no calibration bugs, near-lossless.
- LLM Compressor's INT8 example uses **512 samples at 2048 sequence length** from `ultrachat_200k` and documents "512 samples is a good place to start (increase if accuracy drops)." ModelOpt defaults to a mix of `cnn_dailymail` and NVIDIA's `nemotron-post-training-dataset-v2`, typically 128–512 samples. (Doc examples sometimes use as few as 20 samples for speed — do not copy that into production.)

### 4.2 Calibration pitfalls (the ones that actually bite)

- **Domain mismatch.** Calibrating on news prose for a code-generation or tool-calling model tunes scales to the wrong activation distribution. A 2026 study switching GPTQ calibration from WikiText2 to a math dataset improved reasoning-model accuracy by an average of 9.8 points. Calibrate on data shaped like production traffic — for you, that means agent transcripts: system prompt, tool schemas, tool-call JSON, multi-turn structure.
- **Chat-template mismatch.** An instruction-tuned model in production always sees text wrapped in its chat template (`<|im_start|>system…`). If calibration samples are raw unformatted text, the observed activation ranges are systematically wrong. Apply the model's own chat template to calibration samples.
- **Too-narrow data.** A calibration set of short, similar samples yields scales that fall apart on long, varied inputs. Mix lengths up to your real `max_seq_length`.
- **MoE expert starvation.** Per §2.2, rare experts may see no calibration tokens. Use the toolchains' MoE-aware calibration paths and larger sample counts for MoE models.
- **Air-gap note:** calibration datasets are release artifacts. Ship the exact calibration set in the bundle alongside the checkpoint (the SOP's §4.2 bundle format), so a re-quantization inside staging is reproducible and auditable.

### 4.3 Quantization-aware training (QAT) and its practical 2026 form, QAD

**QAT** simulates quantization *during training* so the model learns weights that survive it. It recovers accuracy PTQ cannot, but requires training infrastructure, data access, and stability engineering — historically impractical for deployment teams, and doubly so in an enclave.

What changed by 2026:

- **Model producers do the QAT for you.** gpt-oss shipped natively MXFP4-trained MoE weights (§1.2). NVIDIA publishes **QAD (Quantization-Aware Distillation)** NVFP4 checkpoints for its Nemotron family: a full-precision teacher distills into the quantized student via a KL-divergence loss, which NVIDIA reports as more stable than classic QAT for models that went through multi-stage post-training (SFT, RL, merging), and robust even without the original training data (NVFP4 QAD report, arXiv 2601.20088, early 2026).
- **Your position:** you consume QAT/QAD, you don't run it. When a small model fails your NVFP4 eval gate, the escalation ladder is: better calibration data → FP8 instead → look for a published QAD/QAT checkpoint. Standing up your own QAT pipeline is almost never the right use of enclave GPUs.

---

## 5. Method-by-method: what each one is and when you would use it

### 5.1 FP8 (W8A8) — the near-lossless workhorse

Weights and activations in FP8 E4M3. Two variants:

- **Dynamic** (`FP8_DYNAMIC` in LLM Compressor): static per-channel weight scales, dynamic per-token activation scales. No calibration data. This is the variant to reach for.
- **Static:** activation scales fixed from calibration. Slightly lower overhead per token, slightly higher risk; used when squeezing the last few percent of throughput.

Newer block-scaled FP8 variants exist (LLM Compressor's `FP8_BLOCK`; ModelOpt's 128×128-block `FP8_PB_WO`), matching how DeepSeek-style models are natively trained. Accuracy is typically within ~1% of BF16 across the board; memory is 2x smaller than BF16; math rate doubles on FP8 tensor cores (Hopper and Blackwell). **Use when:** day-0 model drops, models that fail the NVFP4 gate, anything accuracy-sensitive that still needs to be fast.

### 5.2 NVFP4 (W4A4) — the Blackwell default

Covered in §1.2 and §2.3. Produced by ModelOpt (`--qformat nvfp4`) or LLM Compressor (`scheme="NVFP4"`); requires calibration data. Weights ~3.5x smaller than BF16; FP4 tensor-core math on Blackwell; vLLM picks the GEMM (general matrix multiply) kernel automatically at load time and falls back to W4A16 execution on non-Blackwell GPUs. For FP4 MoE models, vLLM has several FlashInfer-based backends; the SOP's `VLLM_USE_FLASHINFER_MOE_FP4=1` opts into the FlashInfer FP4 MoE kernel. A common 2026 pattern for MoE checkpoints (seen across RedHatAI releases): **NVFP4 for the expert layers, FP8 for attention layers** — mixed precision inside one checkpoint, cutting size >70% while protecting the sensitive attention path. **Use when:** production serving on B200/B300 — this is the default per the SOP.

### 5.3 INT8 SmoothQuant (W8A8) — the pre-FP8 workhorse

The insight: activation outliers (§2.2) make activations hard to quantize, but weights are easy. SmoothQuant divides each activation channel by a smoothing factor and multiplies the corresponding weight column by it — mathematically identical model, but outlier magnitude migrates from activations (hard) into weights (easy). Then both quantize to INT8. In LLM Compressor it is composed with GPTQ: `SmoothQuantModifier(smoothing_strength=0.8)` then `GPTQModifier(scheme="W8A8")`, with ~512 calibration samples. INT8 tensor cores exist from Turing (compute capability 7.5) onward, which is exactly its 2026 role: **the W8A8 option for pre-Ada GPUs that lack FP8 hardware.** On your Blackwell fleet, FP8 dominates it in both simplicity and accuracy; you would only touch INT8 W8A8 if an older GPU pool enters the picture.

### 5.4 AWQ (W4A16) — activation-aware weight-only

AWQ (Activation-aware Weight Quantization) observes which weight channels see large activations and protects them by scaling before 4-bit grouped-INT4 quantization (typical group size 128, effective ~4.1–4.3 bits). Weight-only: memory savings, BF16 math. Huge community checkpoint availability; fast in vLLM via the Marlin kernel family. **Use when:** you need 4-bit on non-Blackwell hardware, or a community AWQ checkpoint is the only quantized option for an exotic model. On Blackwell, NVFP4 supersedes it (native kernels for the newest architectures were still stabilizing as of Aug 2026 — a Blackwell-native GPTQ/AWQ kernel generation landed in the GPTQModel toolkit that month).

### 5.5 GPTQ (W4A16) — error-correcting weight-only

GPTQ quantizes weights column by column, using second-order (Hessian-based) information to adjust the *remaining* unquantized weights to compensate for each rounding error. Same deployment profile as AWQ (grouped INT4, weight-only, Marlin kernels, wide community availability); more calibration-sensitive than AWQ (its accuracy moves visibly with calibration data — §4.2). The GPTQ *algorithm* also lives on inside LLM Compressor as the error-correcting engine applied to other schemes (as in the INT8 recipe above). **Use when:** same niche as AWQ; between them, benchmark on your model rather than trusting folklore — kernel choice matters more than AWQ-vs-GPTQ.

### 5.6 GGUF — the llama.cpp world (not your production path)

GGUF is the checkpoint container for **llama.cpp**, the CPU/consumer-GPU inference engine, with its own zoo of quantization types (Q4_K_M, Q5_K_S, …). It is the right format for laptops and edge boxes. In vLLM, GGUF support is explicitly "highly experimental and under-optimized," and as of mid-2026 it has moved out of the core engine into a separate `vllm-gguf-plugin` package, with single-file-model and tokenizer-conversion limitations. Throughput is poor compared to native formats. **Use when:** never, on this platform. If someone hands you a GGUF, go find (or produce) the safetensors + compressed-tensors/ModelOpt equivalent. Do not certify GGUF artifacts into the enclave as serving checkpoints.

### 5.7 The comparison table

Hardware speedup is relative to BF16 on B200, and "quality loss" assumes a well-executed quantization of a ≥30B model with your eval gate passing (small models: see §2.2).

| Format | Effective bits | B200 speedup | Typical quality loss | vLLM status (Aug 2026) |
|---|---|---|---|---|
| BF16 | 16 | 1x (baseline) | none (is the baseline) | native |
| FP8 W8A8 | ~8 | ~2x math, 2x memory | ≲1% | first-class (ModelOpt + compressed-tensors) |
| NVFP4 W4A4 | ~4.5 | ~4x math, ~3.5x memory | ~1–3% (size-dependent) | first-class on Blackwell; W4A16 fallback elsewhere |
| MXFP4 W4A4 | 4.25 | ~4x math | worse than NVFP4 unless QAT (gpt-oss) | supported (mainly for native-MXFP4 models) |
| INT8 SmoothQuant W8A8 | ~8 | ~2x | ~1% | supported; superseded by FP8 on Ada+ |
| AWQ / GPTQ W4A16 | ~4.1–4.3 | memory only (~4x weights) | ~1–3% | supported via Marlin kernels; community formats |
| GGUF | 2–8 (varies) | poor kernels | varies wildly | experimental, out-of-tree plugin — not production |

(That is seven rows, five columns — the row for KV-cache FP8 is deliberately absent because it composes with all of them; see §7.)

---

## 6. Toolchains: how checkpoints actually get made

Both toolchains run on the **connected staging side** of your air gap (they need model downloads and calibration data), and both produce directories of safetensors that ship across the gap in a release bundle like any other model.

### 6.1 NVIDIA TensorRT Model Optimizer (ModelOpt)

ModelOpt (GitHub: `NVIDIA/Model-Optimizer` — note the repo was renamed from `TensorRT-Model-Optimizer`; docs at nvidia.github.io/Model-Optimizer) is NVIDIA's quantization toolkit and the source of the official `nvidia/*-NVFP4` checkpoints. Install with `pip install nvidia-modelopt`.

The PTQ workflow is one script (from `examples/llm_ptq` / `examples/hf_ptq`):

```bash
# Quantize a HF model to NVFP4 and export a unified HF checkpoint
python hf_ptq.py \
  --pyt_ckpt_path meta-llama/Llama-3.3-70B-Instruct \
  --qformat nvfp4 \
  --export_path /staging/models/llama-3.3-70b-instruct-nvfp4

# Or the wrapper script with tensor-parallel calibration on multiple GPUs
scripts/huggingface_example.sh --model $HF_PATH --quant nvfp4 --tp 8
```

Facts verified from the ModelOpt README and docs (main branch, Aug 2026):

- `--qformat` accepts `fp8`, `int8_sq` (SmoothQuant), `int4_awq`, `w4a8_awq`, `nvfp4`, plus MoE-targeted variants `nvfp4_mlp_only` and `nvfp4_experts_only` (quantize the expert MLPs, keep attention higher precision — the mixed-precision pattern from §5.2).
- Default calibration is a mix of `cnn_dailymail` and `nemotron-post-training-dataset-v2`, typically 128–512 samples. Override with your agent-shaped data per §4.2.
- Export is the **unified Hugging Face checkpoint**: safetensors with quantized weights and scale tensors plus an `hf_quant_config.json` describing the scheme. One checkpoint deploys unmodified on TensorRT-LLM, vLLM, and SGLang — which is exactly the portability an enclave wants.
- In Python, the export API is `modelopt.torch.export.export_hf_checkpoint(model, export_dir)` under `torch.inference_mode()`.

Serving in vLLM: detection via `hf_quant_config.json` is automatic for recipe-covered models, and you can be explicit:

```bash
vllm serve /models/nvidia/llama-3.3-70b-instruct-nvfp4 --quantization modelopt_fp4
# FP8 ModelOpt checkpoints: --quantization modelopt
```

vLLM's ModelOpt integration (docs, v0.26 era) supports FP8 per-tensor, FP8 per-channel/per-token, block-scaled FP8 weight-only, NVFP4 (`modelopt_fp4`), and MXFP8, and selects GEMM kernels automatically at load time.

### 6.2 vLLM's LLM Compressor (compressed-tensors format)

LLM Compressor (GitHub: `vllm-project/llm-compressor`) is the vLLM project's own quantization library, born from Neural Magic and now Red Hat–stewarded. It applies a **recipe** — a declarative list of modifiers — to a Hugging Face model via one call, and saves in the **compressed-tensors** format: standard safetensors plus a quantization config that vLLM's `compressed-tensors` backend reads natively (no flags needed at serve time). As of Aug 2026 it covers FP8/NVFP4/MXFP4/MXFP8/INT8/W4A16 schemes; GPTQ, AWQ, SmoothQuant, AutoRound, and rotation-based (SpinQuant/QuIP-style) algorithms; attention and KV-cache quantization; and MoE-aware calibration.

FP8 dynamic — the no-calibration day-0 path:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-30B-A3B"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

oneshot(model=model, recipe=recipe)   # no dataset argument needed

SAVE_DIR = MODEL_ID.split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

NVFP4 — same shape, plus calibration data:

```python
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

oneshot(
    model=model,
    dataset=ds,                    # your agent-shaped calibration set, chat-templated
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,   # docs demo uses 20; use 512+
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
```

Recipes can equally be YAML files (LLM Compressor writes the applied recipe into the checkpoint, which doubles as your audit record in the bundle):

```yaml
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      targets: ["Linear"]
      scheme: "NVFP4"
      ignore: ["lm_head"]
```

Two recurring details in every recipe: `ignore=["lm_head"]` (the output projection is small and disproportionately accuracy-critical, so it stays high-precision by convention), and for MoE models an ignore pattern for router/gate layers (e.g., `"re:.*mlp.gate$"` in the project's own MoE examples) — quantizing the router that picks experts is a known way to hurt MoE quality for negligible savings.

Serving the output is just `vllm serve /models/qwen3-30b-a3b-fp8-dynamic` — the compressed-tensors config is auto-detected.

**ModelOpt or LLM Compressor?** Outcomes are close on shared formats. Practical tiebreakers: ModelOpt is the source of the official NVIDIA checkpoints and first to support brand-new NVIDIA formats; LLM Compressor is the vLLM-native path with the richer open algorithm menu (rotations, AutoRound, expert pruning) and the format (`compressed-tensors`) that most RedHatAI checkpoints use. Keep both installed in staging; prefer whichever already has a published checkpoint or example for the model at hand — which is the recipes-first philosophy applied to quantization.

### 6.3 Finding pre-quantized checkpoints, and judging them

Your fastest path is almost always someone else's checkpoint. Where to look (all mirrored into staging, per SOP §4):

1. **`nvidia` on Hugging Face** — official ModelOpt NVFP4/FP8 checkpoints (e.g., `nvidia/DeepSeek-R1-0528-NVFP4`, Nemotron NVFP4 and QAD variants), usually with an eval table in the model card.
2. **`RedHatAI` on Hugging Face** — the vLLM ecosystem's checkpoint factory: FP8-dynamic and NVFP4 versions of major open models in compressed-tensors format, often with the mixed NVFP4-experts + FP8-attention pattern for MoE, and with reproducible recipes.
3. **Model vendors themselves** — increasingly ship official FP8 (or natively FP8/MXFP4-trained) weights at release.
4. **Community quantizers** — valuable for coverage, wildly variable in quality.

Trust checklist before a checkpoint enters your bundle pipeline:

- **Provenance of the tool:** does the card state the producing tool and version (ModelOpt X.Y, llm-compressor X.Y) and include the recipe/`hf_quant_config.json`? A checkpoint you cannot reproduce is a checkpoint you cannot debug.
- **Published evals:** is there a BF16-vs-quantized comparison table on named benchmarks? Absence is a yellow flag; "we ran perplexity only" is too (§2.1).
- **Scheme sanity:** `lm_head` ignored, MoE routers ignored, sensible mixed precision for MoE. Read the checkpoint's quantization config; it takes two minutes.
- **The org's track record:** `nvidia` and `RedHatAI` publish methodology; a drive-by account with one upload does not.
- **Your gate is the real test:** whatever the card claims, the enclave agentic eval gate (§8, SOP §5 Phase 2) decides. External evals are evidence, not acceptance.

---

## 7. KV-cache quantization (FP8 KV cache)

The KV cache — the model's per-conversation working memory — is the dominant GPU-memory consumer for agentic serving (long tool traces, many concurrent sessions). Storing it in FP8 instead of BF16 halves its footprint, which roughly **doubles the number of concurrent long sessions per node**, and on modern attention kernels also speeds up decode because attention reads half the bytes.

```bash
vllm serve /models/... --kv-cache-dtype fp8        # fp8 = E4M3 (also: fp8_e5m2)
# Hybrid-attention models (small sliding-window layers):
#   --kv-cache-dtype-skip-layers sliding_window
```

The definitive study is the vLLM blog's FP8 KV-cache validation (April 22, 2026, AWS + Red Hat AI). Verified findings worth remembering:

- **Accuracy, uncalibrated (scales = 1.0):** 94–99% recovery across reasoning and long-context benchmarks — e.g., Llama-3.3-70B at 97–98% AUC on 128k-token multi-round retrieval; ~1–2 point worst-case drops on decode-heavy reasoning tasks. Earlier long-context horror stories (needle-in-a-haystack collapsing from 91% to 13% at 128k on Hopper) traced to an attention-kernel accumulation bug, fixed with two-level accumulation — an object lesson that "FP8 KV regressed" is sometimes a kernel bug, not a format property, and that engine upgrades require eval re-runs (SOP runbook 1).
- **Performance:** inter-token latency slope cut to ~54% of BF16, with break-even around 4k context on B200 (7k on H100) — below that, the quantization overhead can outweigh the bandwidth win. Under load, ~15% higher output throughput on an 8B model.
- **Where it regresses:** models with large attention-head dimension (head_dim = 256) pay ~1.6x TTFT in prefill; hybrid models with many small sliding-window layers gain little (skip those layers with the flag above); some non-standard attention backends (e.g., MLA-family kernels) show consistent small accuracy shifts and may need calibration.
- **Calibration:** three levels exist in vLLM — default scales (1.0), on-the-fly estimation from a warmup batch, or proper calibrated scales baked into the checkpoint via LLM Compressor (a `kv_cache_scheme` on the QuantizationModifier). The blog's guidance: calibrate when uncalibrated accuracy recovery drops below ~95% on your workload. Scales are per-tensor today, with finer granularity in development.

The SOP's stance survives contact with the evidence: **KV FP8 on by default for the agentic pools, validated per model by the eval gate** — and for agentic serving specifically, run the long-context retrieval probes, because that is where KV error would show. (NVIDIA has begun publishing NVFP4-KV-cache work for TensorRT-LLM at very long context; in vLLM, as of Aug 2026, FP8 is the production KV format.)

---

## 8. Accuracy validation: the eval gate, air-gapped

Quantization acceptance is an eval problem, and yours must run inside the enclave. The methodology, layered:

### 8.1 Layer 1 — standardized benchmarks with lm-evaluation-harness

EleutherAI's **lm-evaluation-harness** (`lm-eval`) is the standard offline benchmark runner. The production pattern is to point it at your running vLLM server through the OpenAI-compatible API, so you evaluate the *deployed* artifact — same engine, same kernels, same quantized checkpoint the agents will hit:

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # enclave: never phone home

lm_eval \
  --model local-completions \
  --model_args model=candidate-modelX,base_url=http://vllm.internal:8000/v1/completions,num_concurrent=16 \
  --tasks gsm8k,mmlu_pro,ifeval \
  --batch_size auto \
  --output_path /evals/candidate-modelX/
```

Air-gap mechanics, verified against harness docs and community practice:

- Benchmark datasets are HF datasets; **pre-download them on the staging side** so they land in the HF cache, ship the cache (or better, exported JSONL) in the bundle, and set the offline env vars in the enclave. If you see network errors despite `HF_DATASETS_OFFLINE=1`, the usual cause is a wrong `HF_HOME` or a dataset that was never actually cached.
- For full determinism, define **custom task YAMLs that load local JSON/JSONL** via `dataset_path` — no HF IDs anywhere. This also gives you a frozen, auditable eval set per release, which fits the bundle discipline.
- Run the *same* harness against the BF16 baseline once per model and record it; every quantized candidate compares against that stored number, not against a re-run (GPU nondeterminism makes re-runs wobble by a few tenths of a point).

### 8.2 Layer 2 — the agentic gate (the one that actually decides)

Standard benchmarks miss precisely what your workload stresses (§2.2). The SOP's Phase-2 gate is the acceptance test, and every element targets a known quantization failure mode:

- **Tool-selection accuracy on your real tool schemas** — probes whether error shifted the model's choices among close alternatives (the ranking failure of §2.1).
- **20–50 golden multi-turn agent trajectories, scored on task completion** — probes error accumulation over long generations (§2.2, reasoning degradation).
- **Structured-output validity ≈100% with guided decoding on** — with `response_format`/tool schemas enforced, validity should be near-perfect regardless of quantization; a drop means something else broke (parser, template).
- **Long-context retrieval probes at production context lengths** — probes KV-cache quantization and long-context weight-quantization effects where they live.

### 8.3 Thresholds

Recommended defaults (consistent with the SOP's "compare against the incumbent's recorded scores"; tune to your risk tolerance):

- Layer 1: quantized candidate within **1% relative** of its own BF16 baseline on each core task for FP8, within **2%** for NVFP4; any single-task cliff (>5% drop) is an automatic fail pending investigation even if the average passes.
- Layer 2: no regression beyond run-to-run noise versus the incumbent production model on tool-selection and trajectory completion; structured-output validity ≥99.5%.
- Record everything in the bundle's `evals/` directory — the eval result is the deployment record's evidence, not a promise.

And after promotion, watch the **eval-adjacent runtime metrics** (tool-call parse failure rate, retry rate, task abandonment): a quantization problem that slipped the gate shows up there first.

---

## 9. Decision tree: which format, when

```
New model to serve (target: B200/B300, vLLM, agentic profile)
│
├─ 1. Official NVFP4 checkpoint exists (nvidia/ or RedHatAI/, ModelOpt
│     or compressed-tensors)?
│     ├─ YES → take it. Run enclave eval gate vs BF16 baseline.
│     │        Pass → PRODUCTION: NVFP4 (+ --kv-cache-dtype fp8, validate)
│     │        Fail → step 3.
│     └─ NO  → step 2.
│
├─ 2. FP8 checkpoint exists, or quantize FP8_DYNAMIC in staging
│     (no calibration data needed, hours not days)?
│     ├─ Deploy FP8 as the day-0 production format.
│     └─ In parallel: produce/await NVFP4 (+6–12 h; agent-shaped
│        calibration data; MoE → experts-only NVFP4 + FP8 attention),
│        gate it, then promote via alias flip.
│
├─ 3. NVFP4 fails the gate?
│     ├─ Model ≤ ~10B → expected (§2.2). Ship FP8; look for a
│     │   published QAD/QAT-NVFP4 variant later.
│     ├─ Retry with better calibration (512+ samples, chat-templated,
│     │   agent-shaped, longer sequences).
│     └─ Still failing → FP8 is the production format. Record why.
│
├─ 4. KV cache: fp8 ON for agentic/long-context pools; validate with
│     long-context probes; calibrate scales if recovery < ~95%;
│     skip-layers for sliding-window hybrids; reconsider only for
│     short-context (<4k) latency-critical pools.
│
├─ 5. BF16: eval baseline and reference only — never the production
│     default on Blackwell.
│
└─ 6. Edge cases:
      ├─ Natively-quantized model (gpt-oss MXFP4, FP8-trained) →
      │   serve its native format; do not re-quantize.
      ├─ Older non-Blackwell pool → AWQ/GPTQ W4A16 (Marlin) or
      │   INT8 SmoothQuant W8A8.
      └─ GGUF artifact → not a production input; obtain/produce the
          safetensors equivalent.
```

This is the SOP's Appendix-A row — "NVFP4 (ModelOpt) → FP8 fallback → BF16 baseline only" — expanded with its reasons attached.

---

## 10. Common pitfalls checklist

Every item below is a failure mode observed in the wild; check them before blaming the format.

- [ ] **Quantized without the chat template in calibration.** Scales fitted to raw prose, model serves templated chat. Re-calibrate with templated, agent-shaped samples (§4.2).
- [ ] **Copied a docs example's calibration size (20 samples) into production.** Start at 512; increase if the gate is marginal.
- [ ] **Quantized `lm_head` or the MoE router.** Check the recipe's `ignore` list; both should be excluded.
- [ ] **Judged quality by perplexity alone.** Perplexity-vs-accuracy correlation can be as low as r ≈ 0.2 on some tasks (§2.1). Gate on tasks and trajectories.
- [ ] **Evaluated the HF checkpoint in transformers instead of the deployed vLLM artifact.** Kernels differ; evaluate through the serving stack (§8.1).
- [ ] **Compared against a re-run baseline instead of the recorded one.** Nondeterministic wobble masquerades as regression (or hides one).
- [ ] **Assumed a small model behaves like the 70B blog result.** 7–14B models recover only ~95–98% under NVFP4 (§2.2); gate them harder, fall back to FP8 sooner.
- [ ] **Enabled KV FP8 fleet-wide without long-context probes.** Most models are fine (94–99% recovery); a few backends/architectures regress — and short-context pools may even get slower (break-even ~4k tokens on B200) (§7).
- [ ] **Ignored the engine version when accuracy shifted.** The 91%→13% long-context collapse was a kernel bug, fixed in-engine. Re-run the gate on every vLLM upgrade (SOP runbook 1).
- [ ] **Served an NVFP4 checkpoint on a non-Blackwell canary and drew throughput conclusions.** It silently ran W4A16 fallback. Benchmark on target hardware only.
- [ ] **Forgot the MoE FP4 kernel toggle.** For FP4 MoE models, set `VLLM_USE_FLASHINFER_MOE_FP4=1` per the recipe; a missing env var costs you the MoE speedup you bought.
- [ ] **Accepted a community checkpoint with no recipe and no eval table.** Provenance checklist first (§6.3); your gate second; production never on vibes.
- [ ] **Let a GGUF into the enclave as a serving artifact.** Wrong ecosystem (§5.6).
- [ ] **Didn't ship the calibration set and recipe in the bundle.** Six months later nobody can reproduce or debug the checkpoint. The recipe YAML, calibration data hash, and eval results all belong in `bundle/.../evals` and the manifest.

---

## Study questions

1. **Why does FP8 come in E4M3 and E5M2 variants, and which do you use for inference?**
   Answer: The 8 bits can favor precision (E4M3: 3 mantissa bits, max ±448) or range (E5M2: 5 exponent bits, max ±57,344). Inference stores and multiplies values whose range you can manage with scale factors, so E4M3's extra precision wins; E5M2 is mainly for training gradients. vLLM's `fp8` means E4M3.

2. **Name the three structural differences between NVFP4 and MXFP4 and the effect of each.**
   Answer: Block size 16 vs. 32 (halves each outlier's blast radius); FP8 E4M3 block scales vs. E8M0 power-of-two scales (fractional scales fit blocks tightly — MSE 0.08 vs. 0.72 in NVIDIA's example); an added FP32 per-tensor scale (restores global dynamic range). Cost: ~4.5 vs. 4.25 effective bits per value.

3. **What does W4A16 vs. W4A4 mean, and which one gets FP4 tensor-core speedups?**
   Answer: W4A16 = 4-bit weights, 16-bit activations — memory savings only, math still runs in BF16. W4A4 quantizes activations too, which is what lets Blackwell's FP4 tensor cores execute the matrix math (~4x BF16 rate on B200). Only "A" unlocks tensor cores.

4. **Why is FP8 dynamic the recommended day-0 format when a new model drops?**
   Answer: `FP8_DYNAMIC` needs no calibration dataset (static per-channel weight scales, dynamic per-token activation scales), so it can be produced in staging within hours with no calibration-quality risk, and it is near-lossless (≲1%). NVFP4 follows once calibrated and gated.

5. **What are outlier channels and how do SmoothQuant and AWQ each deal with them?**
   Answer: A few hidden-dimension activation channels carry values 10–100x larger than the rest, wrecking naive activation quantization. SmoothQuant migrates the outlier magnitude from activations into the (easier-to-quantize) weights via a mathematically equivalent rescaling; AWQ leaves activations at 16-bit and instead protects the weight channels that large activations multiply.

6. **Your 8B utility model loses 4% on the agentic gate under NVFP4. What is the escalation ladder?**
   Answer: First re-calibrate properly (512+ chat-templated, agent-shaped samples at realistic lengths); if it still fails, ship FP8 as the production format (expected for ≤~10B models, which recover only ~95–98% under NVFP4); longer-term, look for a published QAT/QAD NVFP4 checkpoint. Do not run your own QAT.

7. **Why is perplexity insufficient as a quantization acceptance test?**
   Answer: Task accuracy depends on ranking among candidate answers; perplexity measures absolute per-token probability. Compression can move one without the other — measured correlations range from r ≈ 0.77 to r ≈ 0.20 depending on the task. Use perplexity as a cheap tripwire, tasks and agent trajectories as the gate.

8. **When would you *not* enable `--kv-cache-dtype fp8`?**
   Answer: Short-context latency-critical pools (break-even is ~4k tokens on B200 — below that BF16 can be faster); models with head_dim = 256 where prefill TTFT matters (~1.6x prefill penalty); hybrid models with many small sliding-window layers (skip those layers instead); and any model whose long-context probes show <~95% recovery uncalibrated — calibrate scales or revert.

9. **What is the real quantization risk specific to MoE models, and what mitigates it?**
   Answer: Not raw format sensitivity (large MoEs are among the most robust models under NVFP4) but calibration coverage: each expert sees only its routed tokens, so rare experts get starved of calibration data. Mitigations: MoE-aware calibration in ModelOpt/LLM Compressor, larger calibration sets, ignoring router layers, mixed precision (NVFP4 experts + FP8 attention), and preferring published checkpoints where this work was done.

10. **How do you run lm-eval-harness in an enclave with no internet?**
    Answer: Pre-cache or export benchmark datasets on the staging side and ship them in the bundle; set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`; point `lm_eval --model local-completions` at the running vLLM server's OpenAI-compatible endpoint; for full auditability define task YAMLs that load local JSONL via `dataset_path`.

11. **What distinguishes a trustworthy pre-quantized checkpoint from a risky one?**
    Answer: Trustworthy: names the producing tool and version, ships the recipe/`hf_quant_config.json`, publishes a BF16-vs-quantized eval table, ignores `lm_head`/routers sensibly, comes from an org with a track record (`nvidia`, `RedHatAI`). Risky: no recipe, no evals or perplexity-only, anonymous uploader. Either way, your enclave eval gate is the actual acceptance test.

12. **Why does an NVFP4 checkpoint run on an H100 but not deliver Blackwell-class speedups?**
    Answer: Hopper has no FP4 tensor cores, so vLLM falls back to weight-only (W4A16-style) execution — dequantize to 16-bit, multiply in BF16. Memory savings persist; the ~2x-over-FP8 math rate requires Blackwell's hardware FP4 path. Benchmark on target hardware only.

---

## Sources

Primary sources used for this document (fetched/verified August 2026):

- NVIDIA — Introducing NVFP4 for Efficient and Accurate Low-Precision Inference: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- NVIDIA — NVFP4 Trains with Precision of 16-Bit…: https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/
- NVIDIA Transformer Engine docs — NVFP4: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html
- NVIDIA Model Optimizer (ModelOpt) repo + LLM PTQ examples: https://github.com/NVIDIA/Model-Optimizer (examples/llm_ptq, examples/hf_ptq)
- ModelOpt — Unified HuggingFace Checkpoint: https://nvidia.github.io/Model-Optimizer/deployment/3_unified_hf.html
- vLLM docs — Quantization index / supported methods: https://docs.vllm.ai/en/latest/features/quantization/
- vLLM docs — NVIDIA ModelOpt integration: https://docs.vllm.ai/en/latest/features/quantization/modelopt/ (verified via repo docs/features/quantization/modelopt.md)
- vLLM docs — Quantized KV Cache: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- vLLM blog — The State of FP8 KV-Cache and Attention Quantization in vLLM (Apr 22, 2026): https://vllm.ai/blog/2026-04-22-fp8-kvcache
- vLLM docs — GGUF (out-of-tree plugin status): https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/gguf.md
- LLM Compressor repo + README: https://github.com/vllm-project/llm-compressor
- LLM Compressor — NVFP4 (W4A4) example: https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/
- LLM Compressor — FP8 W8A8 example: https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w8a8_fp8
- LLM Compressor — INT8 W8A8 (SmoothQuant + GPTQ) example: https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w8a8_int8
- Red Hat Developer — Accelerating large language models with NVFP4 quantization (Feb 4, 2026): https://developers.redhat.com/articles/2026/02/04/accelerating-large-language-models-nvfp4-quantization
- NVIDIA HF checkpoints: https://huggingface.co/nvidia/DeepSeek-R1-0528-NVFP4 and related NVFP4 collection
- RedHatAI HF checkpoints (examples): https://huggingface.co/RedHatAI/Qwen3-Next-80B-A3B-Instruct-NVFP4, https://huggingface.co/RedHatAI/GLM-5.2-NVFP4-FP8
- NVIDIA Research — Quantization-Aware Distillation for NVFP4 (arXiv 2601.20088): https://arxiv.org/abs/2601.20088 and https://research.nvidia.com/labs/nemotron/nemotron-qad
- OpenAI — Introducing gpt-oss (native MXFP4 MoE weights): https://openai.com/index/introducing-gpt-oss/ and https://github.com/openai/gpt-oss
- Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models (arXiv 2504.04823): https://arxiv.org/pdf/2504.04823
- EleutherAI lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- vLLM v0.26.0 release notes (Jul 25, 2026): https://github.com/vllm-project/vllm/releases/tag/v0.26.0
- B200/B300 tensor-core throughput references: https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus, https://verda.com/blog/nvidia-b300-vs-b200-complete-gpu-comparison-to-date
- Perplexity-vs-accuracy correlation under compression: https://arxiv.org/pdf/2604.18085
- Calibration-data selection studies: https://openreview.net/forum?id=pfw3saHzGU, https://arxiv.org/html/2601.11200v1

*Companion documents: [SOP](../SOP-production-ai-model-deployment.md) · [PRIMER](../PRIMER.md).*
