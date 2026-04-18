from __future__ import annotations

import logging
import math

from config import (
    FIRE_EXTINGUISH_MAX_DISTANCE,
    MAX_WATER_DISCHARGE,
    POLICE_CLEAR_MAX_DISTANCE,
)
from action.navigation import compute_path, find_nearest_node
from network.client import RCRSClient
from world.cache import WorldModel
from world.entities import AgentState, AgentType, EntityType


logger = logging.getLogger(__name__)


def _resolve_entity_coords(
    entity_id: int,
    world_model: WorldModel,
) -> tuple[int, int]:
    entity = world_model.tasks.get(entity_id)

    if entity is not None and entity.entity_x is not None and entity.entity_y is not None:
        return entity.entity_x, entity.entity_y

    if entity is not None and entity.raw_sensor_data.position_on_edge is not None:
        pos_edge = entity.raw_sensor_data.position_on_edge
        edge_attrs = world_model.road_graph.nodes.get(pos_edge, {})
        ex: int = edge_attrs.get("x", 0)
        ey: int = edge_attrs.get("y", 0)
        if ex != 0 or ey != 0:
            return ex, ey

    return 0, 0


def get_nav_node(target_id: int, world_model: WorldModel) -> int | None:
    entity = world_model.tasks.get(target_id)

    if entity is not None and entity.raw_sensor_data.position_on_edge is not None:
        pos = entity.raw_sensor_data.position_on_edge
        if world_model.road_graph.has_node(pos):
            return pos

    if world_model.road_graph.has_node(target_id):
        return target_id

    if entity is not None and entity.entity_x is not None and entity.entity_y is not None:
        nearest = find_nearest_node(world_model.road_graph, entity.entity_x, entity.entity_y)
        if nearest is not None:
            return nearest

    logger.warning(
        "Я не нашёл узла графа для target_id=%d",
        target_id,
    )
    return None


def dispatch_action(
    client: RCRSClient,
    agent_type: AgentType,
    agent_state: AgentState,
    tick: int,
    target_id: int,
    agent_node_id: int,
    world_model: WorldModel,
) -> tuple[bool, bool, bool]:
    nav_node_id = get_nav_node(target_id, world_model)

    if nav_node_id is None:
        logger.warning("Я не нашёл узла графа для target_id=%d, сбрасываю цель", target_id)
        # Команду НЕ отправляю — вызывающий код перейдёт к исследованию.
        return False, True, False

    path = compute_path(world_model.road_graph, agent_node_id, nav_node_id)

    if not path:
        logger.warning(
            "Я не могу построить путь к target_id=%d (nav_node=%d), сбрасываю цель",
            target_id, nav_node_id,
        )
        # Команду НЕ отправляю — вызывающий код перейдёт к исследованию.
        return False, True, False

    entity = world_model.tasks.get(target_id)
    tx, ty = _resolve_entity_coords(target_id, world_model)

    if tx == 0 and ty == 0:
        fallback_attrs = world_model.road_graph.nodes.get(nav_node_id, {})
        tx = fallback_attrs.get("x", 0)
        ty = fallback_attrs.get("y", 0)

    ax, ay = agent_state.position.x, agent_state.position.y
    eucl_dist = math.hypot(tx - ax, ty - ay)

    # Пожарный может действовать дистанционно ТОЛЬКО при тушении здания
    # (AKExtinguish — дальнобойный). Для AKRescue сервер требует, чтобы
    # пожарный стоял на одной Area с жертвой (MiscSimulator.checkRescue:
    # h.getPosition().equals(ag.getPosition())).
    fire_rescue_mode = (
        agent_type == AgentType.FIRE_BRIGADE
        and entity is not None
        and entity.type in (EntityType.CIVILIAN, EntityType.HUMAN)
    )
    if agent_type == AgentType.POLICE_FORCE:
        at_target = eucl_dist <= POLICE_CLEAR_MAX_DISTANCE
    elif agent_type == AgentType.FIRE_BRIGADE and not fire_rescue_mode:
        at_target = eucl_dist <= FIRE_EXTINGUISH_MAX_DISTANCE
    else:
        at_target = agent_node_id == nav_node_id

    if at_target:
        return _execute_at_target(
            client, agent_type, agent_state, tick, target_id, tx, ty, entity, world_model,
        )

    return _execute_move(
        client, agent_type, agent_state, tick, target_id,
        agent_node_id, nav_node_id, path, tx, ty, entity, world_model,
    )


def _execute_at_target(
    client: RCRSClient,
    agent_type: AgentType,
    agent_state: AgentState,
    tick: int,
    target_id: int,
    tx: int,
    ty: int,
    entity: object,
    world_model: WorldModel,
) -> tuple[bool, bool, bool]:
    if agent_type == AgentType.AMBULANCE_TEAM:
        return _ambulance_at_target(client, agent_state, tick, target_id, entity, world_model)

    if agent_type == AgentType.FIRE_BRIGADE:
        if entity is None:
            # Я не нашёл сущность — помечаю как недостижимую, чтобы не
            # зацикливаться. Команду НЕ отправляю: вызывающий код
            # перейдёт к исследованию и отправит AKMove.
            return False, True, False

        # Спасение заваленных мирных/союзников — пожарный тоже его умеет
        # (сервер принимает AKRescue от FireBrigade, MiscSimulator.java:407).
        if entity.type in (EntityType.CIVILIAN, EntityType.HUMAN):
            buriedness = entity.raw_sensor_data.buriedness
            hp = entity.raw_sensor_data.hp
            if hp is not None and hp == 0:
                logger.info(
                    "Я пропускаю спасение: цель target_id=%d мертва, tick=%d",
                    target_id, tick,
                )
                world_model.remove_task(target_id)
                return False, True, False
            if buriedness is None or buriedness <= 0:
                logger.info(
                    "Я пропускаю спасение: buriedness=%s у target_id=%d, tick=%d",
                    buriedness, target_id, tick,
                )
                # Откопан — передаю цель медикам (убираю из своих задач).
                world_model.remove_task(target_id)
                return False, True, False
            client.send_rescue(tick, target_id)
            logger.info(
                "Я (пожарный) отправил AKRescue: target_id=%d, buriedness=%d, tick=%d",
                target_id, buriedness, tick,
            )
            return True, False, True

        fieryness = entity.raw_sensor_data.fieryness
        if fieryness is not None and fieryness not in {1, 2, 3}:
            logger.info(
                "Я пропускаю тушение: здание target_id=%d не горит (fieryness=%s), tick=%d",
                target_id, fieryness, tick,
            )
            # Команду НЕ отправляю — вызывающий код перейдёт к
            # исследованию и отправит AKMove вместо бесполезного AKRest.
            # unreachable=True заблокирует повторный выбор этой цели.
            return False, True, False
        if fieryness is None:
            logger.info(
                "Я не знаю fieryness здания target_id=%d, пропускаю тушение, tick=%d",
                target_id, tick,
            )
            # Данные могут появиться на следующем такте; помечаю как
            # недостижимое с коротким сроком блокировки.
            return False, True, False
        water = min(MAX_WATER_DISCHARGE, agent_state.resources.water_quantity)
        client.send_extinguish(tick, target_id, water=water)
        logger.info("Я отправил AKExtinguish: target_id=%d, water=%d, tick=%d", target_id, water, tick)
        return True, False, True

    if agent_type == AgentType.POLICE_FORCE:
        if entity is None:
            logger.info(
                "Я пропускаю расчистку: завал target_id=%d пропал из кэша, tick=%d",
                target_id, tick,
            )
            return False, True, False
        repair_cost = entity.raw_sensor_data.repair_cost
        if repair_cost is not None and repair_cost <= 0:
            logger.info(
                "Я пропускаю расчистку: завал target_id=%d уже расчищен (repair_cost=%s), tick=%d",
                target_id, repair_cost, tick,
            )
            return False, True, False
        # Использую AKClear (по entity ID): MiscSimulator надёжно вычитает
        # clearRate из repair_cost каждый такт. AKClearArea с нулевым
        # коридором (dest == agent pos) не работал — завалы не удалялись.
        client.send_clear(tick, target_id)
        logger.info(
            "Я отправил AKClear: target_id=%d, repair_cost=%s, tick=%d",
            target_id, repair_cost, tick,
        )
        return True, False, True

    # Неизвестный тип агента — не отправляю команду, вызывающий код
    # перейдёт к исследованию.
    return False, True, False


def _ambulance_at_target(
    client: RCRSClient,
    agent_state: AgentState,
    tick: int,
    target_id: int,
    entity: object,
    world_model: WorldModel,
) -> tuple[bool, bool, bool]:
    if entity is None:
        logger.warning(
            "Я не нашёл сущность target_id=%d в кэше, сбрасываю цель (tick=%d)",
            target_id, tick,
        )
        return False, True, False

    buriedness = entity.raw_sensor_data.buriedness  # type: ignore[union-attr]
    entity_type = entity.type  # type: ignore[union-attr]
    logger.info(
        "ДИАГ_AT [AMBULANCE] at_target=True: target_id=%d, buriedness=%s, type=%s, tick=%d",
        target_id, buriedness, entity_type.value, tick,
    )

    if buriedness is not None and buriedness > 0:
        client.send_rescue(tick, target_id)
        logger.info(
            "Я отправил AKRescue: target_id=%d, buriedness=%s, tick=%d",
            target_id, buriedness, tick,
        )
        return True, False, True

    if entity_type == EntityType.HUMAN:
        world_model.remove_task(target_id)
        logger.info(
            "Я снял цель target_id=%d: союзник уже откопан, tick=%d", target_id, tick,
        )
        return False, True, False

    client.send_load(tick, target_id)
    logger.info("Я отправил AKLoad: target_id=%d, tick=%d", target_id, tick)
    return True, False, True


def _execute_move(
    client: RCRSClient,
    agent_type: AgentType,
    agent_state: AgentState,
    tick: int,
    target_id: int,
    agent_node_id: int,
    nav_node_id: int,
    path: list[int],
    tx: int,
    ty: int,
    entity: object,
    world_model: WorldModel,
) -> tuple[bool, bool, bool]:
    dest_x, dest_y = tx, ty

    # Полицейский при движении к цели НЕ останавливается на каждом завале
    # по пути. Веса рёбер графа уже учитывают завалы (refresh_blockade_weights),
    # поэтому A* строит обходной маршрут. Остановка для расчистки каждого
    # промежуточного завала приводила к бесконечному зацикливанию:
    # полицейский отправлял AKClearArea каждый такт, но завал не исчезал
    # (нулевой коридор или проблемы с симулятором), и агент стоял на месте.
    # Расчистка целевого завала происходит в _execute_at_target (при at_target).

    client.send_move(tick, path, dest_x=dest_x, dest_y=dest_y)
    logger.info(
        "Я отправил AKMove: target_id=%d, nav=%d, path_len=%d, dest=(%d,%d), tick=%d",
        target_id, nav_node_id, len(path), dest_x, dest_y, tick,
    )
    return True, False, False


def try_clear_local_blockade(
    client: RCRSClient,
    tick: int,
    agent_node_id: int,
    world_model: WorldModel,
) -> bool:
    """Расчищает ближайший завал на ТЕКУЩЕМ узле полицейского.

    Проверяет только текущий узел (не соседние), чтобы не зацикливаться
    на удалённых завалах. Пропускает завалы с repair_cost=0 или None.
    Использует AKClear (по entity ID) — MiscSimulator надёжно вычитает
    clearRate из repair_cost каждый такт. Возвращает True, если была
    отправлена команда.
    """
    blockade_set = world_model.blockades_by_node.get(agent_node_id)
    if not blockade_set:
        return False

    for blockade_id in blockade_set:
        entity = world_model.tasks.get(blockade_id)
        if entity is None:
            continue
        repair_cost = entity.raw_sensor_data.repair_cost
        if repair_cost is None or repair_cost <= 0:
            continue
        client.send_clear(tick, blockade_id)
        logger.info(
            "Я расчищаю ближайший завал (AKClear): blockade_id=%d, "
            "repair_cost=%d, node=%d, tick=%d",
            blockade_id, repair_cost, agent_node_id, tick,
        )
        return True

    return False


__all__ = [
    "get_nav_node",
    "dispatch_action",
    "try_clear_local_blockade",
]
