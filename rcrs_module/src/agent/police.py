from __future__ import annotations

import logging

from config import AVERAGE_SPEED, C_SWITCH, KERNEL_TIMEOUT
from action.selection import TargetSelector
from decision.filters.pre_filter import PreFilterDispatcher
from decision.utility.aggregator import UtilityAggregator
from network.client import RCRSClient
from world.cache import WorldModel
from world.entities import AgentType


logger = logging.getLogger(__name__)


def run_police_force(host: str, port: int, name: str) -> None:
    agent_type = AgentType.POLICE_FORCE
    client = RCRSClient(host=host, port=port, timeout=KERNEL_TIMEOUT)
    world_model = WorldModel()

    dispatcher = PreFilterDispatcher(work_rate=1.0, average_speed=AVERAGE_SPEED)

    # Я увеличиваю w_d для полиции: после фикса MaxMapDistance дистанция
    # стала значимой, и ближайший завал на пути к спасателям важнее,
    # чем самый дорогой по repair_cost.
    aggregator = UtilityAggregator(w_c=0.30, w_d=0.40, w_e=0.20, w_n=0.10)
    selector = TargetSelector(c_switch=C_SWITCH)

    from agent._bootstrap import bootstrap_and_run
    bootstrap_and_run(client, world_model, agent_type, dispatcher, aggregator, selector, name)


__all__ = ["run_police_force"]
