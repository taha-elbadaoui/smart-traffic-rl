"""
Traffic-light controllers compared by the benchmark lab.

Every controller exposes the same tiny interface so the runner can drive them
identically:

    controller.reset()          # called once per episode
    action = controller.act(obs) # called once per macro-step -> 0 or 1

The CrossroadEnv action space is Discrete(2):
    0 -> serve the North/South axis (green)
    1 -> serve the East/West axis (green)
"""
import numpy as np


class Controller:
    """Base controller interface."""
    name = "base"

    def reset(self):
        pass

    def act(self, obs):
        raise NotImplementedError


class FixedTimeController(Controller):
    """
    A conventional fixed-time signal: it ignores traffic entirely and simply
    alternates the served axis on a fixed cycle. This is the "normal traffic
    light" baseline.

    `hold` is the number of macro-steps each axis stays green before switching.
    One crossroad macro-step is ~25 simulated seconds, so hold=2 ~= 50 s green.
    """
    def __init__(self, hold=2, start_phase=0):
        self.hold = max(1, int(hold))
        self.start_phase = int(start_phase)
        self.name = f"fixed_time(hold={self.hold})"
        self.reset()

    def reset(self):
        self._phase = self.start_phase
        self._counter = 0

    def act(self, obs):
        if self._counter >= self.hold:
            self._phase = 1 - self._phase   # flip served axis
            self._counter = 0
        self._counter += 1
        return self._phase


class RandomController(Controller):
    """Uniformly random phase each macro-step. A noisy lower-bound baseline."""
    def __init__(self, seed=0):
        self.name = "random"
        self._rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, obs):
        return int(self._rng.integers(0, 2))


class PPOController(Controller):
    """
    The trained PPO policy. Observations are normalized with the exact
    VecNormalize statistics saved at training time before being fed to the
    network, then the deterministic action is returned.
    """
    def __init__(self, model, vecnormalize=None, name="ppo"):
        self.model = model
        self.vecnormalize = vecnormalize   # VecNormalize instance (for normalize_obs)
        self.name = name

    def reset(self):
        pass

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if self.vecnormalize is not None:
            # normalize_obs expects a batch; feed a single-row batch back and forth.
            obs = self.vecnormalize.normalize_obs(obs[None, :])[0]
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
