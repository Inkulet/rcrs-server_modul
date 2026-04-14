from __future__ import annotations

import logging

from config import AVERAGE_SPEED, C_SWITCH, KERNEL_TIMEOUT
from agent.loop import run_field_agent
from action.selection import TargetSelector
from decision.filters.pre_filter import PreFilterDispatcher
from decision.utility.aggregator import UtilityAggregator
from network.client import RCRSClient
from world.cache import WorldModel
from world.entities import AgentType


logger = logging.getLogger(__name__)


def run_fire_brigade(host: str, port: int, name: str) -> None:
    agent_type = AgentType.FIRE_BRIGADE
    client = RCRSClient(host=host, port=port, timeout=KERNEL_TIMEOUT)
    world_model = WorldModel()

    dispatcher = PreFilterDispatcher(work_rate=1.0, average_speed=AVERAGE_SPEED)
    aggregator = UtilityAggregator(w_c=0.4, w_d=0.2, w_e=0.2, w_n=0.2)
    selector = TargetSelector(c_switch=C_SWITCH)

    _connect_and_run(client, world_model, agent_type, dispatcher, aggregator, selector, name)


def _connect_and_run(
    client: RCRSClient,
    world_model: WorldModel,
    agent_type: AgentType,
    dispatcher: PreFilterDispatcher,
    aggregator: UtilityAggregator,
    selector: TargetSelector,
    name: str,
) -> None:
    from agent._bootstrap import bootstrap_and_run
    bootstrap_and_run(client, world_model, agent_type, dispatcher, aggregator, selector, name)


__all__ = ["run_fire_brigade"]
