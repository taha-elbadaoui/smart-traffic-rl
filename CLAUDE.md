# CLAUDE.md — AI working context

> Auto-loaded by Claude Code each session. This is **my** state/handoff file:
> current project state, conventions, gotchas, and what's next. Keep it short and
> **update it at the end of each session.** Human-facing decisions + results live
> in `PROJECT_LOG.md`; cross-session facts also live in the memory dir.

## What this project is
Multi-agent **traffic-signal control with RL on a real city network**. Current
focus: a real **8-intersection Cologne district** (`envs/cologne8/`) with real
TAPAS rush-hour demand. Goal: a **CoLight** graph-attention policy that beats the
classic baselines. The old synthetic intersections are archived in `legacy/`.

## Repo map
- `envs/cologne8/` — real Cologne net + route (TAPAS) + buildings + sumocfg
- `envs/rabat_real/` — real Rabat OSM scenario (proof of concept; see its BUILD.md)
- `src/realcity/` — the current work: `cologne_env.py` (sumo-rl factory),
  `baselines.py` (fixed-time + max-pressure), `ippo.py` (shared-param IPPO harness)
- `tools/` — net builders (decorate / build_realistic / rebuild)
- `legacy/` — original synthetic T-junction/crossroad/boulevard project (the first
  report's basis; self-contained, still runnable; see `legacy/README.md`)
- `PROJECT_LOG.md` — decisions (ADRs) + results table. `docs/` — previews.

## Conventions & gotchas (these bit us — don't repeat)
- **Git: push to `dev`, NEVER `origin`.** `origin` is the graded repo; local `main`
  tracks `dev`. Use `gh` via the stored token; `git push dev main`.
- **Training is CPU-bound on SUMO**, not the net. Keep `device="cpu"`, use libsumo,
  `torch.set_num_threads(1)`. GPU does not help.
- **For sumo-rl control loops set `LIBSUMO_AS_TRACI=1`** (much faster than TraCI).
- **Windows console crash:** the legacy training scripts `print()` emojis (🚀/📊),
  which crash in a non-UTF-8 console / when output is redirected. Run with
  `PYTHONIOENCODING=utf-8` if backgrounding them.
- **Eval survivorship bias:** a gridlocking policy completes only easy trips, so its
  tripinfo average looks fast. Always check `completed_trips` (≈2046 expected) and
  use the throughput-aware score (see `ippo.py`).
- Cologne junctions are **heterogeneous** (obs 17/11/10…, actions 4/2/3…) — shared
  policy needs obs padding + action masking.
- `pyproj` is required for OSM lon/lat ↔ net XY (building import).

## Baselines to beat (real Cologne, 07:00–08:00 peak, 2046 trips)
| Controller | wait(s) | timeLoss(s) | travel(s) | trips |
|---|--:|--:|--:|--:|
| fixed-time | 29.27 | 49.26 | 114.39 | 1995 |
| **max-pressure** | **6.46** | 24.69 | 90.28 | 2015 |

Reproduce: `python src/realcity/baselines.py`.

## State (update me)
- ✅ Real Cologne scenario + buildings; sumo-rl 8-agent env validated.
- ✅ Baselines measured (fixed-time, max-pressure). Max-pressure is strong (−78% wait).
- ✅ IPPO + CoLight harnesses built (`ippo.py`, `colight.py`) + `compare.py` capstone.
- ⚠️ **Open problem (ADR-10):** learned policies don't converge in short runs — they
  stick in a gridlock local optimum (greedy eval = constant phase, ~633/2046 trips).
  Reward scaling (REWARD_SCALE) + entropy helped gradient balance but didn't escape it.
  Env is responsive (max-pressure clears 2015), so it's an RL-optimization issue.
- ⏭️ DECISION PENDING: (a) tune custom harness (LR schedule, much longer training,
  reward shaping/curriculum); (b) use **RESCO's reference agents** (tuned for these
  Cologne nets — reuse the learner too); (c) ship baselines as the result.
- Baselines are the validated, defensible result. Don't overstate the learned models.

_Last updated: 2026-06-20._
