from __future__ import annotations

import logging

from network.client import RCRSClient
from world.entities import AgentType


logger = logging.getLogger(__name__)

CENTER_AGENT_TYPES: frozenset[AgentType] = frozenset({
    AgentType.FIRE_STATION,
    AgentType.AMBULANCE_CENTRE,
    AgentType.POLICE_OFFICE,
})


def run_center_agent(client: RCRSClient, agent_type: AgentType) -> None:
    logger.info("Я запускаю центральный агент: type=%s", agent_type.value)

    try:
        while True:
            try:
                packet = client.receive_sense()
            except TimeoutError:
                continue
            except (ConnectionError, OSError) as exc:
                logger.error("Я потерял соединение центрального агента: %s", exc)
                break

            client.send_rest(packet.tick)
            logger.debug("Центральный агент %s: AKRest, такт=%d", agent_type.value, packet.tick)

    except KeyboardInterrupt:
        logger.info("Я завершаю центральный агент %s по сигналу", agent_type.value)
    finally:
        client.disconnect()


__all__ = ["CENTER_AGENT_TYPES", "run_center_agent"]
