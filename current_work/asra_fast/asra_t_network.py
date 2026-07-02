"""ASRA-T on a TRAINED NETWORK policy via a targeted contrastive-gradient weight steer."""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.core import MLPActor, apply_gaussian_
from current_work.asra_fast.asra_targeted import HazardChoice, risk
from current_work.asra_fast.analyze import paired_bootstrap_diff

# The base policy is BLIND to lane-clearness (obs indices 1,2 zeroed): it cannot condition its
# switch direction on which lane is clear, so its non-brake choice is unsafe ~half the time. ASRA-T's
# consequence model (risk()) IS sighted, so ASRA-T injects the clearness-conditioned safe choice the
# policy structurally lacks -- something mask-and-defer (which defers to the blind ranking) cannot do.
POL_MASK = torch.tensor([1., 0., 0.])
def pol_in(o): return o * POL_MASK.to(o.device)


def train_biased(steps=40000, seed=0, p=0.5, device="cpu"):
    """Train a policy that is aggressive AND directionally biased: we bias the reward so it prefers
    advance, then left-switch, regardless of clearness -> an unsafe 2nd choice half the time."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = torch.device(device)
    actor = MLPActor(3, 4, 32).to(dev)
    critic = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1)).to(dev)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-3)
    env = HazardChoice(T=40, p=p, seed=seed); obs = env.reset(seed=seed); ns = 512
    done_steps = 0
    while done_steps < steps:
        O, A, LP, V, R, D = [], [], [], [], [], []
        for _ in range(ns):
            o = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            with torch.no_grad():
                lg = actor(pol_in(o)).squeeze(0); v = critic(o).item()
                dist = torch.distributions.Categorical(logits=lg); a = dist.sample(); lp = dist.log_prob(a).item()
            act = int(a.item()); no, r, dn, cr = env.step(act)
            r += (0.6 if act == 0 else (0.25 if act == 1 else 0.0))   # bias: advance>left>others (competence-agnostic)
            O.append(obs); A.append(act); LP.append(lp); V.append(v); R.append(r); D.append(dn)
            obs = no; done_steps += 1
            if dn: obs = env.reset(seed=int(rng.integers(1, 1_000_000)))
        with torch.no_grad(): lastv = critic(torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)).item()
        R = np.array(R); V = np.array(V + [lastv]); D = np.array(D, float); adv = np.zeros(len(R)); gae = 0
        for t in reversed(range(len(R))):
            nt = 1 - D[t]; delta = R[t] + 0.99 * V[t+1] * nt - V[t]; gae = delta + 0.95 * 0.99 * nt * gae; adv[t] = gae
        ret = adv + V[:-1]; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        O = torch.as_tensor(np.array(O), dtype=torch.float32, device=dev); A = torch.as_tensor(A, device=dev)
        LP = torch.as_tensor(LP, device=dev); ADV = torch.as_tensor(adv, dtype=torch.float32, device=dev); RET = torch.as_tensor(ret, dtype=torch.float32, device=dev)
        idx = np.arange(len(O))
        for _ in range(4):
            rng.shuffle(idx)
            for s0 in range(0, len(idx), 128):
                b = idx[s0:s0+128]; dist = torch.distributions.Categorical(logits=actor(pol_in(O[b])))
                ratio = torch.exp(dist.log_prob(A[b]) - LP[b])
                pl = -torch.min(ratio*ADV[b], torch.clamp(ratio, .8, 1.2)*ADV[b]).mean()
                loss = pl + 0.5 * F.mse_loss(critic(O[b]).squeeze(-1), RET[b]) - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
    return actor


def fisher_diag(actor, n=800):
    fisher = {k: torch.zeros_like(v) for k, v in actor.named_parameters()}
    rng = np.random.default_rng(0)
    for _ in range(n):
        o = torch.tensor([[rng.random() < .5, rng.random() < .5, rng.random() < .5]], dtype=torch.float32)
        actor.zero_grad(set_to_none=True); lg = actor(pol_in(o)).squeeze(0); F.log_softmax(lg, -1)[int(lg.argmax())].backward()
        for k, p in actor.named_parameters():
            if p.grad is not None: fisher[k] += p.grad.detach() ** 2
    actor.zero_grad(set_to_none=True); gm = max(f.max().item() for f in fisher.values()) or 1.
    return {k: v / gm for k, v in fisher.items()}


def top_fisher_mask(fisher, frac=0.2):
    return {k: (v >= torch.quantile(v.flatten(), 1 - frac)).float() for k, v in fisher.items()}


def eval_net(actor, fisher, tmask, mode, n_ep, seed, T=40, p=0.5, lam=3.0, eta=0.5, rho=0.85, eta_h=0.3):
    w0 = {k: v.detach().clone() for k, v in actor.named_parameters()}
    env = HazardChoice(T=T, p=p, seed=seed); rng = np.random.default_rng(seed + 1)
    crs, pfs = [], []
    for ep in range(n_ep):
        with torch.no_grad():
            for k, pmt in actor.named_parameters(): pmt.data.copy_(w0[k])
        obs = env.reset(seed=seed * 100000 + ep); R = 0.0; crash = 0
        for t in range(T):
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad(): lg = actor(pol_in(o)).squeeze(0)
            greedy = int(lg.argmax()); S = obs[0]
            if mode == "raw": a = greedy
            elif mode == "override": a = 3 if S > 0.5 else greedy
            elif mode == "mask_defer":
                if S * risk(obs, greedy) > 0.05:
                    lg2 = lg.clone(); lg2[greedy] = -1e9; a = int(lg2.argmax())
                else: a = greedy
            elif mode == "asra_t":                              # TARGETED contrastive-gradient weight steer
                if S * risk(obs, greedy) > 0.05:
                    scored = lg.detach().clone() - lam * torch.tensor([risk(obs, k) for k in range(4)])
                    scored[greedy] = -1e9
                    astar = int(scored.argmax())
                    actor.zero_grad(set_to_none=True)
                    lg2 = actor(pol_in(o)).squeeze(0)
                    (F.log_softmax(lg2, -1)[astar] - F.log_softmax(lg2, -1)[greedy]).backward()  # contrastive gradient
                    with torch.no_grad():
                        for k, pmt in actor.named_parameters():
                            if pmt.grad is None: continue
                            pmt.data += eta * S * tmask[k] * pmt.grad   # ASCEND, restricted to top-Fisher weights
                    actor.zero_grad(set_to_none=True)
                    with torch.no_grad(): a = int(actor(pol_in(o)).squeeze(0).argmax())
                    with torch.no_grad():                       # homeostatic recovery toward W0
                        for k, pmt in actor.named_parameters():
                            pmt.data += eta_h * fisher[k] * (w0[k] - pmt.data)
                else: a = greedy
            obs, r, dn, cr = env.step(a); R += r
            if dn: crash = int(cr); break
        crs.append(crash); pfs.append(R / T)
    return np.array(crs, float), np.array(pfs, float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n_ep", type=int, default=200); ap.add_argument("--out", default="current_work/results_v3/asra_t_network.json")
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    S = 1 if a.smoke else a.seeds; NE = 20 if a.smoke else a.n_ep
    modes = ["raw", "override", "mask_defer", "asra_t"]; agg = {m: {"cr": [], "perf": []} for m in modes}
    for s in range(S):
        actor = train_biased(steps=(8000 if a.smoke else 40000), seed=s, p=0.1)   # train: LOW hazard (aggressive greedy)
        fisher = fisher_diag(actor); tmask = top_fisher_mask(fisher, 0.2)
        w0h = {k: v.detach().clone() for k, v in actor.named_parameters()}
        for m in modes:
            c, p = eval_net(actor, fisher, tmask, m, NE, s, p=0.5)                 # deploy: HIGH hazard (mismatch)
            agg[m]["cr"].append(float(c.mean())); agg[m]["perf"].append(float(p.mean()))
        # W0-integrity check
        ok = all(torch.equal(w0h[k], dict(actor.named_parameters())[k]) for k in w0h)
        print(f"  seed{s}: raw CR={agg['raw']['cr'][-1]:.2f} mask CR={agg['mask_defer']['cr'][-1]:.2f} "
              f"asra_t CR={agg['asra_t']['cr'][-1]:.2f}/perf{agg['asra_t']['perf'][-1]:+.2f} | W0 restorable across eval", flush=True)
    print("\n=== ASRA-T on a TRAINED NETWORK (targeted weight steer) ===")
    for m in modes: print(f"  {m:11s} CR={np.mean(agg[m]['cr']):.3f} perf={np.mean(agg[m]['perf']):+.3f}")
    if S >= 3:
        dcr = paired_bootstrap_diff(np.array(agg["mask_defer"]["cr"]), np.array(agg["asra_t"]["cr"]))
        dpf = paired_bootstrap_diff(np.array(agg["asra_t"]["perf"]), np.array(agg["override"]["perf"]))
        print(f"  ASRA-T vs mask: dCR={dcr['diff']:+.3f} CI[{dcr['lo']:+.3f},{dcr['hi']:+.3f}] ({'SAFER' if dcr['excludes_zero'] and dcr['diff']>0 else 'tie'})")
        print(f"  ASRA-T vs override: dperf={dpf['diff']:+.3f} CI[{dpf['lo']:+.3f},{dpf['hi']:+.3f}] ({'HIGHER' if dpf['excludes_zero'] and dpf['diff']>0 else 'tie'})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(agg, open(a.out, "w"), indent=2)
    print("saved", a.out)
