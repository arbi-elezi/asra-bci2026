"""Does the BeaverTails result REPLICATE? The consequence-shaped operator on a FROZEN LLM across"""
from __future__ import annotations
import sys, json, argparse, time, re
from pathlib import Path
import numpy as np
from datasets import load_dataset
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.llm_beavertails import load_lm, mean_logprob, choose, REFUSE
from current_work.asra_fast.analyze import paired_bootstrap_diff
RES = Path("current_work/results_v3")


def _truthy(v): return (v is True) or (str(v).lower() == "true")


def pairs_beavertails(n, seed=0, maxc=400):
    ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    safe, unsafe = {}, {}
    for r in ds:
        (safe if _truthy(r["is_safe"]) else unsafe).setdefault(r["prompt"], r["response"][:maxc])
    ps = [p for p in unsafe if p in safe]; np.random.default_rng(seed).shuffle(ps)
    return [{"prompt": p, "safe": safe[p], "unsafe": unsafe[p]} for p in ps[:n]]


def pairs_saferlhf(n, seed=0, maxc=400):
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")
    out = []
    for r in ds:
        s0, s1 = _truthy(r["is_response_0_safe"]), _truthy(r["is_response_1_safe"])
        if s0 and not s1: out.append({"prompt": r["prompt"], "safe": r["response_0"][:maxc], "unsafe": r["response_1"][:maxc]})
        elif s1 and not s0: out.append({"prompt": r["prompt"], "safe": r["response_1"][:maxc], "unsafe": r["response_0"][:maxc]})
    np.random.default_rng(seed).shuffle(out); return out[:n]


def _last_turn(convo):
    parts = re.split(r"\n\nAssistant:", convo)
    if len(parts) < 2: return None, None
    resp = parts[-1].strip()
    prompt = "\n\nAssistant:".join(parts[:-1])
    m = re.split(r"\n\nHuman:", prompt); human = m[-1].strip() if len(m) > 1 else prompt.strip()
    return human, resp


def pairs_hh(n, seed=0, maxc=400):
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")
    out = []
    for r in ds:
        hp, cs = _last_turn(r["chosen"]); hr, rj = _last_turn(r["rejected"])
        if cs and rj and hp and len(cs) > 3 and len(rj) > 3:
            out.append({"prompt": hp, "safe": cs[:maxc], "unsafe": rj[:maxc]})
    np.random.default_rng(seed).shuffle(out); return out[:n]


LOADERS = {"BeaverTails": pairs_beavertails, "PKU-SafeRLHF": pairs_saferlhf, "HH-RLHF": pairs_hh}


def eval_dataset(name, model_name, pairs, gains, device="cuda"):
    tok, model, device = load_lm(model_name, device)
    harm = np.array([1.0, 0.0, 0.0]); S = 1.0; per = []
    for ex in pairs:
        opts = [ex["unsafe"], ex["safe"], REFUSE]
        per.append(np.array([mean_logprob(tok, model, device, ex["prompt"], o) for o in opts]))
    res = {}
    for mode, g in [("raw", 0.), ("override", 0.), ("mask", 0.)] + [(f"op_g{gg:g}", gg) for gg in gains]:
        m = "shaped" if mode.startswith("op_") else mode
        ch = np.array([choose(lp, harm, m, g, S) for lp in per])
        res[mode] = {"unsafe": float(np.mean(ch == 0)), "useful": float(np.mean(ch == 1)), "choices": ch.tolist()}
    # best operator gain at unsafe <= override, compare USEFUL vs override & mask
    ov = res["override"]["unsafe"]
    feas = [res[f"op_g{g:g}"] for g in gains if res[f"op_g{g:g}"]["unsafe"] <= ov + 1e-9]
    best = max(feas, key=lambda r: r["useful"]) if feas else res[f"op_g{gains[-1]:g}"]
    comp = {}
    for base in ["override", "mask"]:
        a = (np.array(best["choices"]) == 1).astype(float); b = (np.array(res[base]["choices"]) == 1).astype(float)
        dd = paired_bootstrap_diff(a, b); sg = int(np.sum(a > b)); nt = int(np.sum(a != b))
        comp[base] = {"d": dd["diff"], "lo": dd["lo"], "hi": dd["hi"], "p": dd["pvalue"], "sign": f"{sg}/{nt}"}
    return {"summary": {k: {"unsafe": v["unsafe"], "useful": v["useful"]} for k, v in res.items()},
            "op_unsafe": best["unsafe"], "op_useful": best["useful"], "comp": comp, "n": len(pairs)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["TinyLlama-1.1B", "Qwen2.5-0.5B", "SmolLM2-360M"])
    ap.add_argument("--datasets", nargs="+", default=["BeaverTails", "PKU-SafeRLHF", "HH-RLHF"])
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--out", default=str(RES / "llm_safety_suite.json"))
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    gains = (2., 5., 10., 20.); n = 15 if a.smoke else a.n
    data = {d: LOADERS[d](n) for d in a.datasets}
    for d in a.datasets: print(f"{d}: {len(data[d])} pairs")
    allout = {}; t0 = time.time()
    for mdl in a.models:
        allout[mdl] = {}
        for d in a.datasets:
            r = eval_dataset(d, mdl, data[d], gains); allout[mdl][d] = r
            s = r["summary"]
            print(f"[{mdl}/{d}] raw unsafe={s['raw']['unsafe']:.2f}/useful={s['raw']['useful']:.2f} | "
                  f"op unsafe={r['op_unsafe']:.2f}/useful={r['op_useful']:.2f} | "
                  f"vs override +{r['comp']['override']['d']:.2f}({r['comp']['override']['sign']},p={r['comp']['override']['p']:.3f}) "
                  f"vs mask +{r['comp']['mask']['d']:.2f}({r['comp']['mask']['sign']},p={r['comp']['mask']['p']:.3f}) mask_unsafe={s['mask']['unsafe']:.2f} [{time.time()-t0:.0f}s]", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(allout, open(a.out, "w"), indent=2)
    print("saved", a.out)
