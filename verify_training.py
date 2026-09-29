"""Part 2 - did the DPO run change the model in the intended direction?
C1 FILE    lora_B != 0 (PEFT inits B=0)          C2 LOAD   every file tensor lands bit-identical in live model
C3 ACTIVE  adapter ON vs OFF logits differ       C4 LEARNED held-out DPO implicit-reward margin CI > 0
                                                            and reward-accuracy > 0.5 (binomial p<0.05)
Controls: NULL (B=0) must fail C1,C3,C4; RANDOM (B=noise, same norms) passes C1,C3 but must fail C4.
--chat : score under the chat template (how the model is SERVED) instead of raw prompt+completion
         (how soup's string-format DPO TRAINS it)."""
import argparse, json, math, random, time
from pathlib import Path
import torch
from safetensors.torch import load_file

CHAT = False
def log(*a): print(time.strftime("%Y-%m-%dT%H:%M:%S"), *a, flush=True)
def read_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def encode_pair(tok, prompt, completion, max_len):
    if CHAT:
        ptxt = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        p = tok(ptxt, add_special_tokens=False)["input_ids"]
        c = tok(completion + "<|im_end|>", add_special_tokens=False)["input_ids"]
    else:   # mirrors TRL non-conversational DPO: separate tokenization, EOS appended
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        c = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    p = p[-(max_len // 2):]
    return p, c[: max(1, max_len - len(p))]

@torch.no_grad()
def completion_logp(model, p, c, device):
    ids = torch.tensor([p + c], device=device)
    lp = torch.log_softmax(model(input_ids=ids).logits.float()[0, len(p) - 1:-1], -1)
    return lp.gather(1, torch.tensor(c, device=device)[:, None]).sum().item()

def pair_logps(model, tok, rows, max_len, device, adapter_on):
    out = []
    for r in rows:
        pc, cc = encode_pair(tok, r["prompt"], r["chosen"], max_len)
        pr, rr = encode_pair(tok, r["prompt"], r["rejected"], max_len)
        if adapter_on:
            out.append((completion_logp(model, pc, cc, device), completion_logp(model, pr, rr, device)))
        else:
            with model.disable_adapter():
                out.append((completion_logp(model, pc, cc, device), completion_logp(model, pr, rr, device)))
    return out

def binom_p_greater(k, n): return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n

def bootstrap_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return means[int(.025 * n)], means[int(.975 * n)]

def lora_modules(model):
    return {n: m for n, m in model.named_modules()
            if hasattr(m, "lora_A") and hasattr(m, "lora_B") and "default" in getattr(m, "lora_A", {})}

def check_file(d):
    sd = load_file(str(Path(d) / "adapter_model.safetensors"))
    nz = {k: v.float().norm().item() for k, v in sd.items() if "lora_B" in k}
    zero = [k for k, n in nz.items() if n == 0.0]
    return sd, dict(n_tensors=len(sd), n_lora_B=len(nz), n_zero_B=len(zero), zero_B_examples=zero[:3],
                    median_B_norm=sorted(nz.values())[len(nz) // 2] if nz else 0.0)

def check_load(model, file_sd):
    live = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    missing, mism = [], []
    for k, v in file_sd.items():
        lk = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
        if lk not in live: missing.append(k); continue
        if not torch.equal(live[lk].detach().cpu().to(v.dtype), v): mism.append(k)
    return dict(file_tensors=len(file_sd), live_lora_tensors=len(live), missing_in_model=len(missing),
                missing_examples=missing[:3], value_mismatch=len(mism))

@torch.no_grad()
def delta_w_stats(model):
    rel = []
    for m in lora_modules(model).values():
        A, B = m.lora_A["default"].weight.float(), m.lora_B["default"].weight.float()
        rel.append((m.scaling["default"] * (B @ A)).norm().item() / (m.base_layer.weight.float().norm().item() + 1e-12))
    rel.sort()
    return dict(n_modules=len(rel), rel_dW_median=rel[len(rel) // 2] if rel else 0.0,
                rel_dW_max=rel[-1] if rel else 0.0, modules_unchanged=sum(x == 0.0 for x in rel))

@torch.no_grad()
def check_active(model, tok, rows, device, n=16):
    d = []
    for r in rows[:n]:
        p, _ = encode_pair(tok, r["prompt"], "", 10 ** 6)
        ids = torch.tensor([p], device=device)
        on = model(input_ids=ids).logits.float()
        with model.disable_adapter(): off = model(input_ids=ids).logits.float()
        d.append((on - off).abs().max().item())
    return dict(max_abs_logit_diff=max(d), mean_max_abs_logit_diff=sum(d) / len(d))

def check_learned(model, tok, rows, beta, max_len, device):
    pol = pair_logps(model, tok, rows, max_len, device, True)
    ref = pair_logps(model, tok, rows, max_len, device, False)
    m = [beta * ((pc - rc) - (pr - rr)) for (pc, pr), (rc, rr) in zip(pol, ref)]
    ch = [beta * (pc - rc) for (pc, _), (rc, _) in zip(pol, ref)]
    rj = [beta * (pr - rr) for (_, pr), (_, rr) in zip(pol, ref)]
    lo, hi = bootstrap_ci(m); wins = sum(x > 0 for x in m)
    return dict(n=len(rows), margin_mean=sum(m) / len(m), margin_ci95=[lo, hi],
                implicit_reward_acc=wins / len(m), implicit_reward_acc_p=binom_p_greater(wins, len(m)),
                chosen_reward_mean=sum(ch) / len(ch), rejected_reward_mean=sum(rj) / len(rj),
                raw_pref_acc_tuned=sum(a > b for a, b in pol) / len(pol),   # info only: length-confounded
                raw_pref_acc_base=sum(a > b for a, b in ref) / len(ref),
                per_row_margin=m)

def per_type(rows, margins):
    agg = {}
    for r, x in zip(rows, margins):
        agg.setdefault(r.get("rejected_type", "?"), []).append(x)
    return {k: dict(n=len(v), mean_margin=round(sum(v) / len(v), 4), acc=round(sum(x > 0 for x in v) / len(v), 2))
            for k, v in sorted(agg.items())}

def evaluate(tag, model, tok, rows_eval, rows_train, beta, max_len, device, fs=None, ls=None):
    log(f"--- {tag} | format={'chat-template' if CHAT else 'raw (as trained)'} ---")
    res = dict(tag=tag, chat=CHAT, file=fs, load=ls, delta_w=delta_w_stats(model),
               active=check_active(model, tok, rows_eval, device))
    le = res["learned_eval"] = check_learned(model, tok, rows_eval, beta, max_len, device)
    res["per_rejected_type_eval"] = per_type(rows_eval, le.pop("per_row_margin"))
    if rows_train:
        res["learned_train"] = check_learned(model, tok, rows_train, beta, max_len, device); res["learned_train"].pop("per_row_margin")
    dw = res["delta_w"]
    crit = {
        "C1_B_nonzero(>=90% modules)": dw["n_modules"] > 0 and dw["modules_unchanged"] <= .1 * dw["n_modules"],
        "C2_all_keys_loaded": ls is None or (ls["missing_in_model"] == 0 and ls["value_mismatch"] == 0),
        "C3_adapter_changes_logits(>1e-3)": res["active"]["max_abs_logit_diff"] > 1e-3,
        "C4a_heldout_margin_CI_lower>0": le["margin_ci95"][0] > 0,
        "C4b_heldout_reward_acc>0.5(p<0.05)": le["implicit_reward_acc_p"] < 0.05,
    }
    res["criteria"], res["PASS"] = crit, all(crit.values())
    for k, v in crit.items(): log(f"  {'PASS' if v else 'FAIL'}  {k}")
    log("  delta_w:", dw); log("  active :", res["active"]); log("  eval   :", le)
    if rows_train: log("  train  :", res["learned_train"])
    log("  per rejected_type:", res["per_rejected_type_eval"])
    log(f"  => {tag}: {'PASS' if res['PASS'] else 'FAIL'}")
    return res

def main():
    global CHAT
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval", required=True); ap.add_argument("--train", default=None)
    ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--n-train", type=int, default=50); ap.add_argument("--controls", action="store_true")
    ap.add_argument("--chat", action="store_true"); ap.add_argument("--out", default="verify.json")
    a = ap.parse_args(); CHAT = a.chat
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
    log("device", dev, "| torch", torch.__version__)
    file_sd, fs = check_file(a.adapter); log("adapter file:", fs)
    log("target_modules:", json.load(open(Path(a.adapter) / "adapter_config.json")).get("target_modules"))
    tok = AutoTokenizer.from_pretrained(a.base)
    # fp32 base on purpose: removes NF4/fp16 noise from a small-difference measurement.
    # NOTE: training used an NF4 base; the adapter is evaluated here on the fp32 base.
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32).to(dev).eval()
    model = PeftModel.from_pretrained(base, a.adapter).to(dev).eval()
    ls = check_load(model, file_sd); log("load integrity:", ls)
    rows_eval = read_jsonl(a.eval)
    rows_train = read_jsonl(a.train)[: a.n_train] if a.train else []
    R = {"real": evaluate("REAL adapter", model, tok, rows_eval, rows_train, a.beta, a.max_length, dev, fs, ls)}
    if a.controls:
        mods = lora_modules(model); saved = {n: m.lora_B["default"].weight.detach().clone() for n, m in mods.items()}
        with torch.no_grad():
            for m in mods.values(): m.lora_B["default"].weight.zero_()
        R["null"] = evaluate("CONTROL null (B=0)", model, tok, rows_eval, [], a.beta, a.max_length, dev)
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for n, m in mods.items():
                w = saved[n]; z = torch.randn(w.shape, generator=g).to(w.device, w.dtype)
                m.lora_B["default"].weight.copy_(z * (w.norm() / (z.norm() + 1e-12)))
        R["random"] = evaluate("CONTROL random (same norms)", model, tok, rows_eval, [], a.beta, a.max_length, dev)
        with torch.no_grad():
            for n, m in mods.items(): m.lora_B["default"].weight.copy_(saved[n])
        R["controls_ok"] = (not R["null"]["PASS"]) and (not R["random"]["PASS"])
        log(f"CONTROLS REJECTED AS EXPECTED: {R['controls_ok']}  | REAL: {'PASS' if R['real']['PASS'] else 'FAIL'}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2, ensure_ascii=False); log("written", a.out)

if __name__ == "__main__":
    main()
