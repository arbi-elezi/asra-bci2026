"""GPU experiment: the decode-controlled frontier + mask-and-defer on the SmolLM2-135M LLM policy."""
from __future__ import annotations
import sys, json, time, argparse, os
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.environment.highway_wrapper import HighwayFRAEnv
from experiment_v3.config import MODELS
from experiment_v3.llm_policy import LLMPolicy
from current_work.asra_fast.core import compute_risk, apply_gaussian_, bootstrap_ci
RES = Path("current_work/results_v3")
BASE = "experiment_v3/SmolLM2-135M"


def load_llm(device="cuda"):
    cfg = next(m for m in MODELS if m.name == "SmolLM2-135M")
    p = LLMPolicy(cfg, device=device); p.load(f"{BASE}/policy_trained.pt")
    fisher = torch.load(f"{BASE}/fisher.pt", weights_only=False, map_location=device)
    return p, fisher


def eval_llm(p, fisher, spec, n_ep, seed, vehicles=15, density=1.5, max_steps=120,
             eta_w=0.05, sigma=0.05, kappa=4.0, rho=0.92, eta_h=0.1, device="cuda", cc=None):
    """spec: {'method':'asra','gain','decode'} | {'method':'mask_defer'} | {'method':'baseline','kind',...}"""
    env = HighwayFRAEnv(seed=seed, vehicles_count=vehicles, vehicles_density=density)
    rng = np.random.default_rng(seed + 7)
    pnames = [n for n, _ in p.get_perturbable_params()]
    crs, sps, brk = [], [], []
    for ep in range(n_ep):
        p.restore_w0()
        pdict = dict(p.get_perturbable_params())
        sup = np.zeros(4); obs, info = env.reset(seed=seed * 100000 + ep)
        cost, ttc = info.get("cost", 0.), info.get("ttc", 10.); col, L, nb = 0, 0, 0
        for t in range(max_steps):
            with torch.no_grad():
                logits = p.get_logits_from_obs(obs)
            greedy = int(logits.argmax())
            m = spec["method"]
            if m == "asra":
                S = float(np.clip(cost, 0, 1)); R = compute_risk(cost, ttc, greedy); alpha = S * R
                if alpha > 0.05 and spec["gain"] > 0:
                    p.model.zero_grad(); p.action_head.zero_grad()
                    lg = p.get_logits_from_obs(obs); torch.log_softmax(lg, -1)[greedy].backward()
                    with torch.no_grad():
                        for nm in pnames:
                            g = pdict[nm].grad
                            if g is None: continue
                            epi = int(g.data.abs().flatten().argmax())
                            apply_gaussian_(pdict[nm], g.data, epi, sigma, eta_w * spec["gain"] * alpha)
                    sup[greedy] -= kappa * spec["gain"] * alpha
                with torch.no_grad():
                    adj = p.get_logits_from_obs(obs) + torch.tensor(sup, device=logits.device, dtype=logits.dtype)
                a = int(adj.argmax()) if spec["decode"] == "greedy" else int(torch.distributions.Categorical(logits=adj).sample())
                sup *= rho
                with torch.no_grad():                              # Fisher-weighted recovery toward W0
                    pidx = 0; w0 = p.get_w0()
                    for nm in pnames:
                        pr = pdict[nm]; ne = pr.numel()
                        if nm in w0:
                            fs = fisher[pidx:pidx+ne].reshape(pr.shape).to(pr.dtype).to(pr.device)
                            pr.data += eta_h * fs * (w0[nm].to(pr.dtype) - pr.data)
                        pidx += ne
            elif m == "shaped":                                    # PRINCIPLED operator on the LLM: logit tilt by cost critic
                S = float(np.clip(cost, 0, 1))
                with torch.no_grad():
                    qc = cc.predict_state(torch.as_tensor(obs, dtype=torch.float32, device=logits.device)).squeeze(0)
                    adj = logits - spec["gain"] * S * (qc - qc.mean())
                a = int(adj.argmax()) if spec["decode"] == "greedy" else int(torch.distributions.Categorical(logits=adj).sample())
            elif m == "mask_defer":
                S = float(np.clip(cost, 0, 1)); R = compute_risk(cost, ttc, greedy)
                lg2 = logits.clone()
                if S * R > 0.05: lg2[greedy] = -1e9
                a = int(lg2.argmax())
            elif m == "baseline":
                k = spec["kind"]
                if k == "noop_greedy": a = greedy
                elif k == "noop_sample": a = int(torch.distributions.Categorical(logits=logits).sample())
                elif k == "ttc_brake": a = 2 if ttc < spec.get("ttc_k", 2.) else int(torch.distributions.Categorical(logits=logits).sample())
                elif k == "prob_brake": a = 2 if rng.random() < spec.get("p", 0.5) else int(torch.distributions.Categorical(logits=logits).sample())
                else: raise ValueError(k)
            if a == 2: nb += 1
            sps.append(0)  # placeholder; speed added below per-step
            spd_step = float(obs[2]); L += 1
            obs, r, term, trunc, info = env.step(a); cost, ttc = info.get("cost", 0.), info.get("ttc", 10.)
            sps[-1] = spd_step
            if term: col = int(info.get("collision", False)); break
            if trunc: break
        crs.append(col); brk.append(nb / max(L, 1))
    env.close()
    # aggregate: crs per-episode; sps is per-step -> mean speed
    return {"cr": float(np.mean(crs)), "speed": float(np.mean(sps)), "brake_frac": float(np.mean(brk)), "n_ep": n_ep}


def _load_cc(path, device):
    import torch as _t
    from src.agents.cost_critic import CostCriticNet
    ck = _t.load(path, map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else (ck.get("state_dict", ck) if isinstance(ck, dict) else ck)
    cc = CostCriticNet(12, 4, 64).to(device); cc.load_state_dict(sd); cc.eval()
    for pp in cc.parameters(): pp.requires_grad_(False)
    return cc


def run(out_path, seeds, n_ep, max_steps, g_grid=(0., 1., 3., 6.), device="cuda",
        cc_path="checkpoints/cost_critic/full.pt"):
    p, fisher = load_llm(device); cc = _load_cc(cc_path, device)   # cost critic for the principled operator
    conds = ([("noop_greedy", {"method": "baseline", "kind": "noop_greedy"}),
              ("noop_sample", {"method": "baseline", "kind": "noop_sample"}),
              ("mask_defer", {"method": "mask_defer"}),
              ("ttc_brake@2", {"method": "baseline", "kind": "ttc_brake", "ttc_k": 2.}),
              ("prob_brake@.5", {"method": "baseline", "kind": "prob_brake", "p": 0.5})]
             + [(f"shaped_g{g}_greedy", {"method": "shaped", "gain": g, "decode": "greedy"}) for g in g_grid]
             + [(f"shaped_g{g}_sample", {"method": "shaped", "gain": g, "decode": "sample"}) for g in g_grid if g > 0]
             + [(f"asra_g{g}_greedy", {"method": "asra", "gain": g, "decode": "greedy"}) for g in g_grid])
    results = []; t0 = time.time()
    for s in seeds:
        for name, spec in conds:
            r = eval_llm(p, fisher, spec, n_ep, s, max_steps=max_steps, device=device, cc=cc)
            results.append({"cond": name, "seed": s, **spec, **r})
            print(f"  seed{s} {name:16s} CR={r['cr']:.3f} speed={r['speed']:.2f} brake={r['brake_frac']:.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "seeds": list(seeds), "n_ep": n_ep, "elapsed_s": time.time()-t0}, open(out_path, "w"), indent=2)
    print(f"saved {out_path} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "llm_frontier.json"))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed_start", type=int, default=0)  # for parallel seed-range instances (backward compat: 0)
    ap.add_argument("--n_ep", type=int, default=25)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        p, fisher = load_llm()
        for name, spec in [("noop_greedy", {"method": "baseline", "kind": "noop_greedy"}),
                           ("mask_defer", {"method": "mask_defer"}),
                           ("asra_g3_greedy", {"method": "asra", "gain": 3., "decode": "greedy"})]:
            r = eval_llm(p, fisher, spec, n_ep=3, seed=0, max_steps=40)
            print(f"[SMOKE] {name}: CR={r['cr']:.2f} speed={r['speed']:.2f} brake={r['brake_frac']:.2f}")
        # weights drift mid-episode (perturbed); the invariant is that W0 is restorable exactly.
        p.restore_w0()
        assert p.verify_w0(), "W0 not restorable — restore_w0 is incomplete!"
        print("[SMOKE] OK — W0 restorable to exact hash after perturbation")
    else:
        run(a.out, list(range(a.seed_start, a.seed_start + a.seeds)), a.n_ep, a.max_steps)
