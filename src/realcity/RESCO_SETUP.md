# Running RESCO's reference agents on the Cologne scenario

RESCO ships tuned learned agents (IPPO, MPLight, IDQN, …) for these exact Cologne
networks. Its dependency stack is **legacy** (pfrl + old gym + numpy<2), so it
needs its **own Python 3.10 environment** — it will not install on the project's
Python 3.13.

## One-time setup

```bash
# 1. Python 3.10 (RESCO's deps don't build on 3.11+)
winget install --id Python.Python.3.10

# 2. Dedicated venv (kept out of git via .venv-resco/)
"$LOCALAPPDATA/Programs/Python/Python310/python.exe" -m venv .venv-resco

# 3. Clone RESCO and install it (with the torch + pfrl agents) into that venv
git clone https://github.com/Pi-Star-Lab/RESCO.git   # e.g. into a scratch dir
.venv-resco/Scripts/python -m pip install -e "<path-to>/RESCO[pfrl,torch]"
```

## Gotchas (these cost real time — don't rediscover them)

- **SUMO comes from the venv, not the system.** Point SUMO at the venv's bundled
  SUMO 1.27 or libsumo's DLLs clash with the system SUMO (1.26):
  ```bash
  export SUMO_HOME="<proj>/.venv-resco/Lib/site-packages/sumo"
  export PATH="$SUMO_HOME/bin:$PATH"
  ```
- **`pfrl` breaks `libsumo`** on Windows: importing pfrl loads a DLL that makes the
  later `import libsumo` fail (`_libsumo` procedure-not-found). Fix: run with
  **`libsumo:False`** — RESCO then uses socket TraCI (slower, ~85 s/episode, but
  it works). `import torch`/`gym`/`matplotlib` before libsumo is fine; only pfrl
  conflicts.
- **RESCO's DQN agents (MPLight, IDQN) crash** with current pfrl 0.4
  (`batch_act` IndexError — research-code rot). **IPPO works** — use it.
- RESCO must be run from its `resco_benchmark/` directory.

## Run (from `RESCO/resco_benchmark/`, venv env exported as above)

```bash
python main.py @cologne8 @IPPO libsumo:False gui:False episodes:40 save_console_log:False
```

Results (per-episode delay, models, logs) are written to `RESCO/results/<run_name>/`.
Compare the reported delay against our baselines in `PROJECT_LOG.md`
(fixed-time wait 29.3 s, max-pressure 6.46 s).
