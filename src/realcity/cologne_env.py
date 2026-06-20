"""
Shared factory for the real Cologne multi-agent traffic-signal environment.

Built on `sumo-rl`, which exposes each real intersection as its own agent with
its own (variable) phase count — so the irregular multi-phase signal programs of
the OSM-imported Cologne junctions are handled generically. All learners
(IPPO baseline, CoLight) and the baselines import this one factory so the
scenario definition stays in a single place.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
NET = os.path.join(ROOT, "envs", "cologne8", "cologne8.net.xml")
ROUTE = os.path.join(ROOT, "envs", "cologne8", "cologne8.rou.xml")
SUMOCFG = os.path.join(ROOT, "envs", "cologne8", "cologne8.sumocfg")

BEGIN = 25200          # 07:00 — start of the TAPAS morning peak
PEAK_SECONDS = 3600    # one hour of demand
DELTA_TIME = 5         # seconds between agent decisions
EXPECTED_TRIPS = 2046  # TAPAS trips in the demand; a good controller clears ~all of them


def make_env(use_gui=False, num_seconds=PEAK_SECONDS, delta_time=DELTA_TIME,
             reward_fn="pressure", single_agent=False, **kwargs):
    """Create the Cologne SumoEnvironment (multi-agent by default).

    reward_fn="pressure" minimizes intersection pressure (incoming minus
    outgoing queues) — the proven objective for signal control, and the fix for
    the throughput-weighted reward that capped the earlier synthetic models.
    """
    from sumo_rl import SumoEnvironment
    return SumoEnvironment(
        net_file=NET,
        route_file=ROUTE,
        use_gui=use_gui,
        begin_time=BEGIN,
        num_seconds=num_seconds,
        delta_time=delta_time,
        reward_fn=reward_fn,
        single_agent=single_agent,
        **kwargs,
    )
