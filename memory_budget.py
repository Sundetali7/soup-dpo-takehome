"""Part 1: VRAM budget for streamed LoRA-DPO, BEFORE training.
python scripts/memory_budget.py <model> <max_len> <batch> <lora_r> <n_lora_modules_per_layer:2|7> <quant:nf4|fp16> [data.jsonl]"""
import json, sys
GB = 1024 ** 3

def cfg_of(m):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(m)
    return dict(h=c.hidden_size, L=c.num_hidden_layers, I=c.intermediate_size, heads=c.num_attention_heads,
                kv=c.num_key_value_heads, V=c.vocab_size, tied=bool(c.tie_word_embeddings))

def real_len(m, path, max_len):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(m); Ls = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); p = min(len(tok(r["prompt"])["input_ids"]), max_len // 2)
        Ls += [min(p + len(tok(r[k])["input_ids"]) + 1, max_len) for k in ("chosen", "rejected")]
    Ls.sort(); return dict(p50=Ls[len(Ls) // 2], p95=Ls[int(.95 * len(Ls))], max=Ls[-1])

def budget(c, seq, batch, r, nmod, quant):
    h, L, I, V = c["h"], c["L"], c["I"], c["V"]; kvd = c["kv"] * (h // c["heads"])
    tok = 2 * batch * seq                                            # chosen+rejected in ONE tensor
    lin = {"q": (h, h), "k": (h, kvd), "v": (h, kvd), "o": (h, h), "gate": (h, I), "up": (h, I), "down": (I, h)}
    p_layer = sum(a * b for a, b in lin.values())
    wbytes = 0.5625 if quant == "nf4" else 2                         # NF4 = 4 bit + absmax overhead
    used = ["q", "v"] if nmod == 2 else list(lin)
    lora = L * sum(r * sum(lin[k]) for k in used)
    rows = [
        ("embeddings / tied lm_head fp16 (resident)",   V * h * 2),
        ("decoder, ALL layers resident (no streaming)", L * p_layer * wbytes),
        ("decoder, streamed: 2 buffers",                2 * p_layer * wbytes),
        ("1 layer dequantized to fp16 for compute",     p_layer * 2 if quant == "nf4" else 0),
        ("LoRA w + grad + Adam (16 B/param)",           lora * 16),
        ("activations, checkpointing (layer inputs)",   L * tok * h * 2),
        ("logits fp32, x1 copy",                        tok * V * 4),
    ]
    fixed = sum(b for n, b in rows if "ALL layers" not in n and "logits" not in n)
    lg = rows[-1][1]
    return rows, lora, fixed + lg, fixed + 3 * lg

m, ml, b, r, nmod, q = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
c = cfg_of(m); print("config:", c)
seqs = [("worst case, padded to max_length", ml)]
if len(sys.argv) > 7:
    rl = real_len(m, sys.argv[7], ml); print("real lengths:", rl); seqs.append(("typical, p95 real length", rl["p95"]))
for label, s in seqs:
    rows, lora, lo, hi = budget(c, s, b, r, nmod, q)
    print(f"\n=== {label}: seq={s} batch={b} -> {2*b} rows, LoRA r={r} on {nmod} modules/layer = {lora/1e6:.2f}M params ===")
    for n, v in rows: print(f"  {n:<46s} {v/GB:7.3f} GB")
    print(f"  PREDICTED torch peak (logits x1..x3)          {lo/GB:.2f} - {hi/GB:.2f} GB")
    print(f"  PREDICTED nvidia-smi (+0.4 ctx, +15% cache)   {lo*1.15/GB+0.4:.2f} - {hi*1.15/GB+0.4:.2f} GB")
    print("  DPO reference pass: no_grad, adapter disabled on SAME weights -> no extra weights, transient logits only")
