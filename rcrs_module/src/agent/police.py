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

    # По матмодели f_urgency_police = 1/(d_blockade_to_target+ε) — это уже
    # ранжирует завалы по важности (близко к целям → высокий приоритет).
    # Поэтому w_c доминирует; w_d (агент→завал) остаётся значимым как
    # тайбрейкер «при равной важности еду к ближайшему».
    aggregator = UtilityAggregator(w_c=0.50, w_d=0.25, w_e=0.15, w_n=0.10)
    selector = TargetSelector(c_switch=C_SWITCH)

    from agent._bootstrap import bootstrap_and_run
    bootstrap_and_run(client, world_model, agent_type, dispatcher, aggregator, selector, name)


__all__ = ["run_police_force"]
