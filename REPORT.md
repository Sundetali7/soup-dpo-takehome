# DPO + layer streaming on a free T4: ship or not?

**Verdict: DON'T SHIP.** The run is real (the adapter loads, is active and moves held-out preferences in the right
direction, and two negative controls fail the same checks), but the effect is small, the only SHIP signal comes from a
metric that cannot tell a good detox rewrite from a fluent wrong one, and training and serving use different prompt formats.

**Setup.** Colab T4 (15 360 MiB), Qwen2.5-0.5B-Instruct, soup-cli **0.75.1** in a Python 3.12 venv. Default Colab has
Python 3.13, so a plain `pip install soup-cli` silently resolves to 0.72.4, which crashes on DPO + streaming
(`Tensor on device cuda:0 is not on the expected device meta`). See `logs/why_py312_*.log` and `notebooks/draft_plan_A.ipynb`.
Data: `r1char9/ru-detox-dpo` filtered to customer-support scenarios, with the instruction "Перепиши сообщение клиента
вежливо…". 450 train pairs / 50 held-out pairs from the HF `test` split, with no source-text overlap. The Soup data package
was not provided; as the email allowed, an open preference dataset was used instead.
Run **A** is the candidate: lr 1e-5, LoRA r16/α32 with default targets (q,v → 1.08 M params), batch 4, 2 epochs, NF4,
`stream_layers`, max_length 512.
Run **B** is a diagnostic run: lr 5e-5, all 7 linear projections (8.8 M params).

## 1. Memory budget

| Component (batch 4 → 8 rows, seq 512) | Estimate |
|---|---|
| Embeddings / tied lm_head, fp16, resident | 0.27 GB |
| Decoder: 2 streamed NF4 buffers (+1 layer dequantized) | ~0.05 GB (0.19 GB if all resident) |
| LoRA weights + grads + AdamW (16 B/param) | 0.02 GB (A) / 0.14 GB (B) |
| Activations, checkpointed (24 layers × 8×512×896×2 B) | 0.18 GB |
| **Logits**: 8×512×151 936 fp32 = 2.49 GB per copy; ×3 (logits, log-softmax, grad) | **7.5 GB** |
| DPO reference pass | same weights with adapter disabled → 0 extra weights; transient logits under no_grad |
| **Predicted `torch.max_memory_allocated`** | **≈ 8.0 GB** |

Measured values:

| | Run A | Run B |
|---|---|---|
| Soup pre-flight | 9.33 GB (logits 8.71) | 9.36 GB |
| Soup-reported (`max_memory_allocated`) | 8.0 GB | 8.1 GB |
| **nvidia-smi peak** | **12.86 GiB** | **14.21 GiB = 95 % of the card** |

The torch number matches the estimate once logits are counted as ~3 fp32 copies at padded length. Logits are
~93 % of the budget, so streaming the weights saves almost nothing for a 0.5 B model with a 152 k vocabulary.
The ~4.9 GiB gap to nvidia-smi comes from three sources:
1. CUDA context and cuBLAS workspace (~0.4 GB).
2. Caching-allocator memory that is reserved but not allocated. The log says
   `expandable_segments allocator hint not enabled: CUDA was already initialised`, so the large, variable-size logits
   blocks from the policy and reference passes fragment the cache.
3. Everything outside PyTorch (bitsandbytes).

Soup reports and gates on the torch number. Run B was 0.8 GiB from OOM while Soup showed 8.1/14.6 GB.
I do **not** trust the reported tok/s (7 805 / 8 705). It is unclear whether padding and the reference forward pass
are counted, and the README itself notes its tok/s was measured before a correctness fix.

## 2. Did the model train?

Falling loss is not proof. DPO loss can fall while:
- the adapter is never applied at inference (key mismatch on load, which PEFT only warns about);
- only a few modules move;
- the policy "wins" by making **both** answers less likely.

A no-op run also starts at the same loss, ln 2 = 0.693.

`scripts/verify_training.py` runs four independent checks on the **held-out** 50 pairs, in fp32:

- **C1:** `lora_B ≠ 0`. PEFT initializes B to 0, so a non-zero B means the optimizer touched that module.
- **C2:** every tensor in the adapter file is present, bit-identical, in the loaded PeftModel.
- **C3:** logits differ with the adapter ON vs OFF.
- **C4:** the DPO implicit-reward margin β[(logπ−logref)(c) − (logπ−logref)(r)] has a bootstrap 95 % CI above 0, and
  reward accuracy is above 0.5 (exact binomial p < 0.05).

It also runs two **controls that must fail**:
- **NULL**: B zeroed.
- **RANDOM**: B replaced by noise with the same per-module norm.

The RANDOM control shows that weight and activity checks alone prove nothing.

| Held-out, raw format (as trained) | margin (95 % CI) | reward acc | Δlogp chosen / rejected (×β) | Verdict |
|---|---|---|---|---|
| A real | +0.057 (+0.042, +0.074) | 0.82 (p=3e-6) | +0.073 / +0.017 | PASS |
| A NULL | 0 | 0.00 | 0 / 0 | FAIL C1, C3, C4 |
| A RANDOM (same norms) | −0.000 (−0.001, +0.000) | 0.54 (p=0.34) | ≈0 / ≈0 | passes C1–C3, **FAIL C4** |
| B real | +2.93 (+2.35, +3.55) | 0.96 (p=1e-12) | +0.54 / −2.39 | PASS |
| **B real, chat template (as served)** | +2.59 (+2.11, +3.09) | 0.96 | **−0.12** / −2.71 | PASS, but chosen got *less* likely |

Adapter integrity is clean: 96/96 (A) and 336/336 (B) tensors loaded, with 0 mismatches and 0 zero B.
The train-subset margin is close to the held-out margin (A 0.069 vs 0.057), so there is no sign of memorization.

**What the check does NOT detect:**
- whether the preferred answers are *good*: it only measures agreement with the dataset's labels, and label noise is
  learned too;
- generation quality, degeneration, hallucination and fluency;
- whether meaning is preserved;
- behaviour on out-of-distribution tickets;
- exact equivalence to the NF4 training-time base (I evaluate the adapter on an fp32 base);
- reward hacking through length or style.

## 3. Silent failures

| Risk | How checked | Caught by Soup? | Evidence |
|---|---|---|---|
| Wrong Soup version installed | `pip index`, `soup version` | **No**: resolves to 0.72.4 on Py 3.13 without a warning | `why_py312` log, draft notebook |
| Train/serve prompt mismatch | verifier in raw vs chat format | **No** | string DPO data trains without a chat template, while `ship`/`chat` apply one. In B, chosen becomes more likely raw (+0.54) but less likely as served (−0.12) |
| Likelihood displacement | TensorBoard rewards | **No**: the tracker stores loss/grad_norm only | B train rewards: chosen −0.81, rejected −5.56, loss 0.08 "looks great" |
| fp16 overflow step | TensorBoard `grad_norm` | **No** | B: 1 step with `grad_norm = nan/inf`, silently skipped by GradScaler |
| LoRA default = q,v only | `adapter_config`, trainable count | **No** (the earlier `soup profile` printed 4.8 M, actual 1.08 M) | A: relative ΔW median 6e-4, tiny effect |
| Hidden 10 % val split never evaluated; hidden 4× grad-accum | logs | **No** | 405 train rows, 52 steps, 0 `eval_loss` lines |
| VRAM under-reported | nvidia-smi vs Soup | **No** | B: 14.2 GiB actual vs 8.1 reported |
| Preference-data defects | synthetic set with injected defects + `lint`/`validate` | Partly | `near_duplicates` reported **OK while skipped** (datasketch missing); empty `rejected` caught by nothing; swapped labels uncatchable by rules; `data doctor` refuses DPO |
| `bf16` capability check | `torch.cuda.is_bf16_supported()` | n/a | returns **True** on T4 (sm_75, no hardware bf16); Soup correctly picked fp16 for training, but `soup ship` without `--config` loads bf16 |
| `soup ship` verdict strength | NULL adapter through `ship` | Partly | NULL → DON'T SHIP (good). Real A → SHIP on 0.70 → 0.78 = **4 of 50** tasks, with no significance test (`--noise-floor` hung > 50 min; each live ship run took ~48 min on T4). The same config in an earlier run gave 0.84, so run-to-run spread ≈ the effect. Worst benchmark delta −0.05 sits exactly on the 5 % threshold and passes |
| Ship leg-1 metric | read `eval/custom.py` | n/a | regex "no profanity and ≥ 20 chars", greedy, 64 tokens: a fluent answer that changes the meaning passes |

## 4. Verdict: DON'T SHIP

What must change first:
1. **Train in conversational format** (`prompt`/`chosen`/`rejected` as message lists) so training matches serving;
   re-run the verifier in chat format.
2. Set `target_modules` explicitly and choose lr/epochs between A and B. Gate on the **chosen** log-ratio not going
   negative, not only on the margin.
3. Replace the regex ship metric with a **meaning-preservation plus toxicity** check (judge or embedding similarity to
   `chosen`), use more than 50 paired items, a McNemar/bootstrap test and a working noise floor.
4. Do a generation audit of 30+ held-out rewrites, base vs tuned.
5. Run the validation split that is already being held out, pin Python ≤ 3.12 / soup ≥ 0.75, and set
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before CUDA initializes.

**Surprised me / still concerns me.** Most of the serious problems were silent and happened *before* training: the
installer picked an old broken version without a word, and the linter printed "OK" for a check it never ran.
Then run B's curves looked textbook (loss 0.69 → 0.08, accuracy 1.0) while the chosen answers were getting *less*
likely, and under the serving template that was true on held-out data too. I am also concerned that three memory
numbers for one config (Soup pre-flight 9.3, torch 8.0, nvidia-smi 12.9 GiB) disagree by 60 %, and the gate uses the
smallest. An earlier verifier I used reported "NOT PROVEN" for a similar run only because its criterion (raw summed
log-prob accuracy) is length-confounded, which shows how easily the check itself can be wrong.

**AI tools.** Claude (Anthropic) drafted the notebook cells, `verify_training.py`, `memory_budget.py` and the synthetic
defect generator. I ran everything on Colab and checked every number in this report against the raw logs.
Suggestions I rejected or fixed:
- a log wrapper that swallowed Soup's confirmation prompt and was garbled by IPython `$l` substitution;
- a `pgrep` guard that matched its own shell, so the nvidia-smi logger never started;
- a data cell that would have overwritten the real training set;
- a length-confounded pass criterion, replaced by an implicit-reward binomial test.

The controls (NULL/RANDOM) were added specifically so the verifier could not pass vacuously.
