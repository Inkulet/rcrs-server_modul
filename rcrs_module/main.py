from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path


SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from config import (  # noqa: E402
    KERNEL_HOST,
    KERNEL_PORT,
    setup_logging,
)
from world.entities import AgentType  # noqa: E402


_shutdown_requested: bool = False


def _sigterm_handler(signum: int, frame: object) -> None:
    global _shutdown_requested
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _sigterm_handler)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="RCRS diploma agent")
    parser.add_argument(
        "--agent-type",
        choices=[t.value for t in AgentType],
        default=AgentType.FIRE_BRIGADE.value,
        help="Тип агента (FIRE_BRIGADE / AMBULANCE_TEAM / POLICE_FORCE / центральные)",
    )
    parser.add_argument("--host", default=KERNEL_HOST)
    parser.add_argument("--port", type=int, default=KERNEL_PORT)
    parser.add_argument("--name", default="diploma-agent")
    args = parser.parse_args()

    agent_type = AgentType(args.agent_type)

    from agent.fire      import run_fire_brigade
    from agent.ambulance import run_ambulance_team
    from agent.police    import run_police_force
    from agent._bootstrap import bootstrap_and_run
    from agent.center    import CENTER_AGENT_TYPES
    from config          import AVERAGE_SPEED, C_SWITCH, KERNEL_TIMEOUT
    from action.selection import TargetSelector
    from decision.filters.pre_filter import PreFilterDispatcher
    from decision.utility.aggregator import UtilityAggregator
    from network.client  import RCRSClient
    from world.cache     import WorldModel

    if agent_type == AgentType.FIRE_BRIGADE:
        run_fire_brigade(args.host, args.port, args.name)

    elif agent_type == AgentType.AMBULANCE_TEAM:
        run_ambulance_team(args.host, args.port, args.name)

    elif agent_type == AgentType.POLICE_FORCE:
        run_police_force(args.host, args.port, args.name)

    else:
        # Центральные агенты: FIRE_STATION, AMBULANCE_CENTRE, POLICE_OFFICE.
        client = RCRSClient(host=args.host, port=args.port, timeout=KERNEL_TIMEOUT)
        world_model = WorldModel()
        dispatcher = PreFilterDispatcher(work_rate=1.0, average_speed=AVERAGE_SPEED)
        aggregator = UtilityAggregator(w_c=0.4, w_d=0.2, w_e=0.2, w_n=0.2)
        selector = TargetSelector(c_switch=C_SWITCH)
        bootstrap_and_run(client, world_model, agent_type, dispatcher, aggregator, selector, args.name)


if __name__ == "__main__":
    main()
