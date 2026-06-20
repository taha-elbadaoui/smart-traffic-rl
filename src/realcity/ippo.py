"""
Shared-parameter IPPO for the real Cologne district (8 traffic signals).

One actor-critic network is shared across all junctions. Because the junctions
are heterogeneous (different observation sizes and phase counts), observations
are zero-padded to a common width and a per-agent action mask hides the invalid
phases. Each agent acts every `delta_time` seconds; transitions from all agents
are pooled into one PPO update.

This is the non-graph learned baseline and the harness CoLight builds on (swap
the shared MLP body for a graph-attention encoder over neighboring junctions).

    python src/realcity/ippo.py --iterations 200 --episodes-per-iter 2
    python src/realcity/ippo.py --iterations 3 --seconds 800   # quick smoke
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("LIBSUMO_AS_TRACI", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cologne_env import make_env, ROOT, PEAK_SECONDS, EXPECTED_TRIPS  # noqa: E402
from baselines import parse_tripinfo  # noqa: E402

torch.set_num_threads(1)
DEVICE = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        h = self.body(obs)
        logits = self.pi(h).masked_fill(mask == 0, -1e9)
        return logits, self.v(h).squeeze(-1)


class Spaces:
    """Padded observation / masked action bookkeeping for heterogeneous agents."""
    def __init__(self, env):
        self.ids = list(env.ts_ids)
        self.obs_dim = {ts: env.observation_spaces(ts).shape[0] for ts in self.ids}
        self.act_dim = {ts: env.action_spaces(ts).n for ts in self.ids}
        self.MAXO = max(self.obs_dim.values())
        self.MAXA = max(self.act_dim.values())
        self.mask = {ts: torch.tensor([1.0] * self.act_dim[ts]
                                      + [0.0] * (self.MAXA - self.act_dim[ts]))
                     for ts in self.ids}

    def pad(self, ts, obs):
        v = np.asarray(obs, dtype=np.float32)
        if v.shape[0] < self.MAXO:
            v = np.concatenate([v, np.zeros(self.MAXO - v.shape[0], dtype=np.float32)])
        return v


def collect(env, policy, sp, n_episodes, gamma=0.99, lam=0.95):
    obs_b, mask_b, act_b, logp_b, ret_b, adv_b = [], [], [], [], [], []
    ep_returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        traj = {ts: [] for ts in sp.ids}
        while True:
            obs_t = torch.stack([torch.from_numpy(sp.pad(ts, obs[ts])) for ts in sp.ids])
            mask_t = torch.stack([sp.mask[ts] for ts in sp.ids])
            with torch.no_grad():
                logits, vals = policy(obs_t, mask_t)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logps = dist.log_prob(acts)
            actions = {ts: int(acts[i]) for i, ts in enumerate(sp.ids)}
            nobs, rewards, dones, _ = env.step(actions)
            for i, ts in enumerate(sp.ids):
                traj[ts].append((obs_t[i], mask_t[i], acts[i], logps[i],
                                 float(vals[i]), float(rewards[ts])))
            obs = nobs
            if dones.get("__all__", False):
                break
        # GAE per agent (episode terminates -> bootstrap 0)
        total = 0.0
        for ts in sp.ids:
            steps = traj[ts]
            adv, gae, next_v = [0.0] * len(steps), 0.0, 0.0
            for t in reversed(range(len(steps))):
                r, v = steps[t][5], steps[t][4]
                delta = r + gamma * next_v - v
                gae = delta + gamma * lam * gae
                adv[t] = gae
                next_v = v
                total += r
            for t, s in enumerate(steps):
                obs_b.append(s[0]); mask_b.append(s[1]); act_b.append(s[2])
                logp_b.append(s[3]); adv_b.append(adv[t]); ret_b.append(adv[t] + s[4])
        ep_returns.append(total / len(sp.ids))
    adv = torch.tensor(adv_b)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return (torch.stack(obs_b), torch.stack(mask_b), torch.stack(act_b),
            torch.stack(logp_b), torch.tensor(ret_b), adv, float(np.mean(ep_returns)))


def ppo_update(policy, optim, batch, epochs=4, clip=0.2, mb=2048, vf=0.5, ent=0.01):
    obs, mask, act, old_logp, ret, adv = batch
    n = obs.shape[0]
    for _ in range(epochs):
        for idx in torch.randperm(n).split(mb):
            logits, val = policy(obs[idx], mask[idx])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act[idx])
            ratio = torch.exp(logp - old_logp[idx])
            a = adv[idx]
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            vloss = ((val - ret[idx]) ** 2).mean()
            loss = pg + vf * vloss - ent * dist.entropy().mean()
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optim.step()


def evaluate(policy, sp, tag="ippo"):
    out = os.path.join(ROOT, "results", f"cologne_{tag}_tripinfo.xml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    env = make_env(use_gui=False, num_seconds=PEAK_SECONDS, sumo_seed=42,
                   additional_sumo_cmd=f"--tripinfo-output {out}")
    obs = env.reset()
    while True:
        obs_t = torch.stack([torch.from_numpy(sp.pad(ts, obs[ts])) for ts in sp.ids])
        mask_t = torch.stack([sp.mask[ts] for ts in sp.ids])
        with torch.no_grad():
            logits, _ = policy(obs_t, mask_t)
            acts = torch.argmax(logits, dim=-1)
        _, _, dones, _ = env.step({ts: int(acts[i]) for i, ts in enumerate(sp.ids)})
        if dones.get("__all__", False):
            break
    env.close()
    return parse_tripinfo(out)


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
    print(f"agents={len(sp.ids)} padded_obs={sp.MAXO} max_actions={sp.MAXA}")

    policy = ActorCritic(sp.MAXO, sp.MAXA).to(DEVICE)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # Throughput-aware score: penalize trips that never clear (avoids rewarding a
    # gridlocking policy whose few completed trips look artificially fast).
    def score(m):
        return m["mean_time_loss"] + 1000.0 * (1.0 - m["completed_trips"] / EXPECTED_TRIPS)

    best = float("inf")
    for it in range(1, args.iterations + 1):
        batch = collect(env, policy, sp, args.episodes_per_iter)
        ppo_update(policy, optim, batch[:6])
        print(f"[it {it:3d}] mean pressure-return/agent = {batch[6]:8.1f}")
        if it % args.eval_every == 0 or it == args.iterations:
            m = evaluate(policy, sp)
            s = score(m)
            print(f"         eval: wait={m['mean_waiting_time']:.2f}s "
                  f"timeLoss={m['mean_time_loss']:.2f}s "
                  f"trips={m['completed_trips']}/{EXPECTED_TRIPS} score={s:.1f}")
            if s < best:
                best = s
                torch.save(policy.state_dict(), os.path.join(ROOT, "models", "ippo_cologne_best.pt"))
    env.close()
    print(f"done. best score = {best:.2f} (lower = clears more demand with less delay)")


if __name__ == "__main__":
    main()
