from __future__ import annotations


import logging
import time

from config import MAX_CONNECT_RETRIES
from agent.center import CENTER_AGENT_TYPES, run_center_agent
from agent.loop import run_field_agent
from action.selection import TargetSelector
from decision.filters.pre_filter import PreFilterDispatcher
from decision.utility.aggregator import UtilityAggregator
from network.client import RCRSClient
from world.cache import WorldModel
from world.entities import AgentType


logger = logging.getLogger(__name__)


def bootstrap_and_run(
    client: RCRSClient,
    world_model: WorldModel,
    agent_type: AgentType,
    dispatcher: PreFilterDispatcher,
    aggregator: UtilityAggregator,
    selector: TargetSelector,
    agent_name: str,
) -> None:
    for attempt in range(1, MAX_CONNECT_RETRIES + 1):
        try:
            client.connect()
            break
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            if attempt == MAX_CONNECT_RETRIES:
                logger.error(
                    "Bootstrap: исчерпаны все попытки подключения к ядру, агент завершает работу [attempts=%d]: %s",
                    MAX_CONNECT_RETRIES, exc,
                )
                return
            logger.info("Bootstrap: ожидание готовности ядра [attempt=%d/%d]: %s", attempt, MAX_CONNECT_RETRIES, exc)
            time.sleep(3)

    try:
        agent_id = client.handshake(agent_name, agent_type)
        logger.info(
            "Bootstrap: рукопожатие завершено [agent_id=%d, agent_type=%s]",
            agent_id, agent_type.value,
        )
    except (ConnectionError, TimeoutError, OSError) as exc:
        logger.error("Bootstrap: рукопожатие не выполнено — %s", exc)
        client.disconnect()
        return

    if agent_type in CENTER_AGENT_TYPES:
        run_center_agent(client, agent_type)
    else:
        run_field_agent(client, world_model, agent_type, dispatcher, aggregator, selector)


__all__ = ["bootstrap_and_run"]
