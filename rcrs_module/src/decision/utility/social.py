from __future__ import annotations

"""В этом модуле я вычисляю социальный фактор f_social для предотвращения интерференции агентов."""

# Я реализую формулу диплома: N_i(r) = sum_{k in A, k≠j} I(||pos_k - loc_i|| < r) * I(type_k = type_j)
# Затем нормирую результат к [0,1], деля на общее число однотипных агентов (кроме себя).

import logging
import math

from world.cache import WorldModel
from world.entities import AgentType, Position


logger = logging.getLogger(__name__)

# Я задаю радиус по умолчанию в единицах карты RCRS (мм).
DEFAULT_RADIUS: float = 2000.0


def _euclidean_distance(pos_a: Position, pos_b: Position) -> float:
    """Здесь я вычисляю евклидово расстояние ||pos_a - pos_b|| между двумя точками карты.

    Я использую евклидову метрику, а не граф-расстояние, потому что формула диплома
    определяет социальный фактор через ||pos_k - loc_i|| — физическое расстояние
    на плоскости, а не длину маршрута по дорогам.
    """
    return math.hypot(pos_a.x - pos_b.x, pos_a.y - pos_b.y)


def social_factor(
    world_model: WorldModel,
    target_position: Position,
    agent_type: AgentType,
    current_agent_id: int,
    radius: float = DEFAULT_RADIUS,
) -> float:
    """Здесь я вычисляю нормированный социальный фактор f_social in [0, 1].

    Я считаю долю однотипных союзников в радиусе r вокруг цели:
        f_social = N_i(r) / max(1, |A_same_type|)
    где N_i(r) — количество агентов того же типа в радиусе r,
    |A_same_type| — общее число наблюдаемых агентов того же типа (кроме себя).

    Нормировка на |A_same_type| гарантирует принадлежность к [0, 1] независимо
    от размера команды, что соответствует требованию диплома о соизмеримости факторов.
    """
    if radius <= 0:
        logger.warning(
            "Я получил неположительный радиус для социального фактора: radius=%.2f",
            radius,
        )
        return 0.0

    # Я собираю всех агентов того же типа, кроме себя — это знаменатель нормировки.
    same_type_agents = [
        agent
        for agent_id, agent in world_model.agents.items()
        if agent_id != current_agent_id and agent.type == agent_type
    ]

    total = len(same_type_agents)
    if total == 0:
        # Я не обнаружил союзников того же типа — конкуренции нет, f_social = 0.
        logger.debug(
            "Я не нашел однотипных агентов при расчете социального фактора для цели entity_id=%s",
            target_position.entity_id,
        )
        return 0.0

    # Я применяю евклидовый критерий близости из формулы диплома: ||pos_k - loc_i|| < r.
    # Это O(N) по числу агентов, что в сочетании с внешним O(M) даёт O(N*M) на такт.
    # При N ≤ 30 это укладывается в бюджет ≤ 1000 мс.
    count = sum(
        1
        for agent in same_type_agents
        if _euclidean_distance(agent.position, target_position) < radius
    )

    try:
        result = float(count) / float(total)
    except ZeroDivisionError:
        # Я страхуюсь от деления на ноль, хотя total > 0 проверен выше.
        logger.warning(
            "Я поймал деление на ноль при нормировке социального фактора для цели entity_id=%s",
            target_position.entity_id,
        )
        return 0.0

    logger.debug(
        "Я рассчитал f_social=%.4f для цели entity_id=%s: %d из %d агентов в радиусе %.0f",
        result,
        target_position.entity_id,
        count,
        total,
        radius,
    )
    return result


__all__ = ["DEFAULT_RADIUS", "social_factor"]
