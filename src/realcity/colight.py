"""
CoLight-style graph-attention multi-agent policy for the real Cologne district.

Extends the IPPO harness with **coordination**: each junction's observation is
encoded, then a multi-head self-attention layer over all junctions (the
intersection graph) lets each agent attend to the others before choosing its
phase. This is the mechanism designed to beat max-pressure, which acts purely
locally.

Reuses the IPPO `Spaces` (obs padding + action masking) and `evaluate`
(throughput-aware) so results are directly comparable.

    python src/realcity/colight.py --iterations 200 --episodes-per-iter 2
    python src/realcity/colight.py --iterations 3 --seconds 800   # quick smoke
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("LIBSUMO_AS_TRACI", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cologne_env import make_env, ROOT, PEAK_SECONDS, EXPECTED_TRIPS, REWARD_SCALE  # noqa: E402
from ippo import Spaces, evaluate  # noqa: E402

torch.set_num_threads(1)
DEVICE = "cpu"


class CoLightPolicy(nn.Module):
    """Per-junction encoder -> self-attention over junctions -> actor/critic."""
    def __init__(self, obs_dim, act_dim, hidden=128, heads=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # one graph-attention block (self-attention across the N junctions)
        self.attn = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
            dropout=0.0, batch_first=True,
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        single = obs.dim() == 2          # [N,O] single step vs [B,N,O] batch
        if single:
            obs, mask = obs.unsqueeze(0), mask.unsqueeze(0)
        h = self.enc(obs)                # [B,N,H]
        z = self.attn(h)                 # [B,N,H] — agents attend to each other
        logits = self.pi(z).masked_fill(mask == 0, -1e9)
        val = self.v(z).squeeze(-1)
        if single:
            return logits.squeeze(0), val.squeeze(0)
        return logits, val


def collect(env, policy, sp, n_episodes, gamma=0.99, lam=0.95):
    """Rollout keeping per-timestep agent groups (the graph net mixes agents)."""
    O, M, A, LP, RET, ADV = [], [], [], [], [], []
    ep_returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        ep = []
        while True:
            obs_t = torch.stack([torch.from_numpy(sp.pad(ts, obs[ts])) for ts in sp.ids])
            mask_t = torch.stack([sp.mask[ts] for ts in sp.ids])
            with torch.no_grad():
                logits, vals = policy(obs_t, mask_t)        # [N,A],[N]
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logps = dist.log_prob(acts)
            nobs, rewards, dones, _ = env.step({ts: int(acts[i]) for i, ts in enumerate(sp.ids)})
            rew_t = REWARD_SCALE * torch.tensor([rewards[ts] for ts in sp.ids], dtype=torch.float32)
            ep.append((obs_t, mask_t, acts, logps, vals, rew_t))
            obs = nobs
            if dones.get("__all__", False):
                break
        # GAE per agent across the episode (terminates -> bootstrap 0)
        vseq = torch.stack([e[4] for e in ep])     # [T,N]
        rseq = torch.stack([e[5] for e in ep])     # [T,N]
        adv = torch.zeros_like(vseq)
        gae = torch.zeros(vseq.shape[1])
        next_v = torch.zeros(vseq.shape[1])
        for t in reversed(range(len(ep))):
            delta = rseq[t] + gamma * next_v - vseq[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
            next_v = vseq[t]
        ret = adv + vseq
        for t in range(len(ep)):
            O.append(ep[t][0]); M.append(ep[t][1]); A.append(ep[t][2])
            LP.append(ep[t][3]); ADV.append(adv[t]); RET.append(ret[t])
        ep_returns.append(float(rseq.sum() / vseq.shape[1]))
    adv = torch.stack(ADV)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return (torch.stack(O), torch.stack(M), torch.stack(A),
            torch.stack(LP), torch.stack(RET), adv, float(np.mean(ep_returns)))


def ppo_update(policy, optim, batch, epochs=4, clip=0.2, mb=256, vf=0.5, ent=0.02):
    obs, mask, act, old_logp, ret, adv = batch          # [T,N,*]
    T = obs.shape[0]
    for _ in range(epochs):
        for idx in torch.randperm(T).split(mb):
            logits, val = policy(obs[idx], mask[idx])    # [b,N,A],[b,N]
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act[idx])               # [b,N]
            ratio = torch.exp(logp - old_logp[idx])
            a = adv[idx]
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            vloss = ((val - ret[idx]) ** 2).mean()
            loss = pg + vf * vloss - ent * dist.entropy().mean()
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optim.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--episodes-per-iter", type=int, default=2)
    ap.add_argument("--seconds", type=int, default=PEAK_SECONDS)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    env = make_env(use_gui=False, num_seconds=args.seconds, reward_fn="pressure")
    env.reset()
    sp = Spaces(env)
    print(f"agents={len(sp.ids)} padded_obs={sp.MAXO} max_actions={sp.MAXA} (CoLight graph-attention)")

    policy = CoLightPolicy(sp.MAXO, sp.MAXA).to(DEVICE)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    def score(m):
        return m["mean_time_loss"] + 1000.0 * (1.0 - m["completed_trips"] / EXPECTED_TRIPS)

    best = float("inf")
    for it in range(1, args.iterations + 1):
        batch = collect(env, policy, sp, args.episodes_per_iter)
        ppo_update(policy, optim, batch[:6])
        print(f"[it {it:3d}] mean pressure-return/agent = {batch[6]:8.1f}")
        if it % args.eval_every == 0 or it == args.iterations:
            m = evaluate(policy, sp, tag="colight")
            s = score(m)
            print(f"         eval: wait={m['mean_waiting_time']:.2f}s "
                  f"timeLoss={m['mean_time_loss']:.2f}s "
                  f"trips={m['completed_trips']}/{EXPECTED_TRIPS} score={s:.1f}")
            if s < best:
                best = s
                torch.save(policy.state_dict(), os.path.join(ROOT, "models", "colight_cologne_best.pt"))
    env.close()
    print(f"done. best score = {best:.2f}")


if __name__ == "__main__":
    main()
