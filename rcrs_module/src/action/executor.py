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
        client.send_rest(tick)
        return False, True, False

    path = compute_path(world_model.road_graph, agent_node_id, nav_node_id)

    if not path:
        if agent_type == AgentType.POLICE_FORCE:
            tx_fb, ty_fb = _resolve_entity_coords(target_id, world_model)
            if tx_fb == 0 and ty_fb == 0:
                fallback_attrs = world_model.road_graph.nodes.get(nav_node_id, {})
                tx_fb = fallback_attrs.get("x", 0)
                ty_fb = fallback_attrs.get("y", 0)
            if tx_fb != 0 or ty_fb != 0:
                client.send_clear_area(tick, dest_x=tx_fb, dest_y=ty_fb)
                logger.info(
                    "Я пробиваю путь к target_id=%d (нет графового пути): AKClearArea dest=(%d,%d), tick=%d",
                    target_id, tx_fb, ty_fb, tick,
                )
                return True, False, True

        logger.warning(
            "Я не могу построить путь к target_id=%d (nav_node=%d), сбрасываю цель",
            target_id, nav_node_id,
        )
        client.send_rest(tick)
        return False, True, False

    entity = world_model.tasks.get(target_id)
    tx, ty = _resolve_entity_coords(target_id, world_model)

    if tx == 0 and ty == 0:
        fallback_attrs = world_model.road_graph.nodes.get(nav_node_id, {})
        tx = fallback_attrs.get("x", 0)
        ty = fallback_attrs.get("y", 0)

    ax, ay = agent_state.position.x, agent_state.position.y
    eucl_dist = math.hypot(tx - ax, ty - ay)

    if agent_type in (AgentType.FIRE_BRIGADE, AgentType.POLICE_FORCE):
        max_dist = (
            FIRE_EXTINGUISH_MAX_DISTANCE
            if agent_type == AgentType.FIRE_BRIGADE
            else POLICE_CLEAR_MAX_DISTANCE
        )
        at_target = eucl_dist <= max_dist
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
            client.send_rest(tick)
            return False, False, False
        client.send_extinguish(tick, target_id, water=MAX_WATER_DISCHARGE)
        logger.info("Я отправил AKExtinguish: target_id=%d, tick=%d", target_id, tick)
        return True, False, True

    if agent_type == AgentType.POLICE_FORCE:
        if entity is None:
            logger.info(
                "Я пропускаю AKClear: завал target_id=%d пропал из кэша, tick=%d",
                target_id, tick,
            )
            client.send_rest(tick)
            return False, False, False
        client.send_clear(tick, target_id)
        logger.info(
            "Я отправил AKClear: target_id=%d, dest=(%d,%d), tick=%d",
            target_id, tx, ty, tick,
        )
        return True, False, True

    client.send_rest(tick)
    return False, False, False


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
        client.send_rest(tick)
        return False, False, False

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
        return False, False, False

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

    if agent_type == AgentType.POLICE_FORCE:
        ax, ay = agent_state.position.x, agent_state.position.y
        nearest_blockade: tuple[int, int, int] | None = None
        min_d = float("inf")
        for t in world_model.tasks.values():
            if t.type != EntityType.BLOCKADE:
                continue
            bx, by = _resolve_entity_coords(t.id, world_model)
            if bx == 0 and by == 0:
                continue
            d = math.hypot(bx - ax, by - ay)
            if d < min_d:
                min_d = d
                nearest_blockade = (bx, by, t.id)

        if nearest_blockade is not None and min_d <= POLICE_CLEAR_MAX_DISTANCE:
            bx, by, bid = nearest_blockade
            client.send_clear_area(tick, dest_x=bx, dest_y=by)
            logger.info(
                "Я расчищаю завал на пути: blockade_id=%d, dist=%.1f, tick=%d",
                bid, min_d, tick,
            )
            return True, False, True

        if len(path) <= 1 and entity is not None:
            client.send_clear_area(tick, dest_x=dest_x, dest_y=dest_y)
            logger.info(
                "Я упёрся в завал target_id=%d → AKClearArea dest=(%d,%d), tick=%d",
                target_id, dest_x, dest_y, tick,
            )
            return True, False, True

    client.send_move(tick, path, dest_x=dest_x, dest_y=dest_y)
    logger.debug(
        "Я отправил AKMove: nav_node=%d, path_len=%d, dest=(%d,%d)",
        nav_node_id, len(path), dest_x, dest_y,
    )
    return True, False, False


__all__ = [
    "get_nav_node",
    "dispatch_action",
]
