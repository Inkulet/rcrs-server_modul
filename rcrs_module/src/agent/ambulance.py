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


def run_ambulance_team(host: str, port: int, name: str) -> None:
    agent_type = AgentType.AMBULANCE_TEAM
    client = RCRSClient(host=host, port=port, timeout=KERNEL_TIMEOUT)
    world_model = WorldModel()

    # Сервер снижает buriedness на 1 за каждый AKRescue (MiscSimulator:458).
    # work_rate=1.0 даёт корректную оценку t_work = buriedness тиков.
    # ESTIMATED_TRIP_TO_REFUGE в pre_filter снижен до 10, чтобы не
    # отсеивать спасаемых гражданских из-за чрезмерного запаса.
    dispatcher = PreFilterDispatcher(work_rate=1.0, average_speed=AVERAGE_SPEED)
    aggregator = UtilityAggregator(w_c=0.45, w_d=0.30, w_e=0.15, w_n=0.10)
    selector = TargetSelector(c_switch=C_SWITCH)

    from agent._bootstrap import bootstrap_and_run
    bootstrap_and_run(client, world_model, agent_type, dispatcher, aggregator, selector, name)


__all__ = ["run_ambulance_team"]
