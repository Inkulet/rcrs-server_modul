from __future__ import annotations

import logging
import math
from collections.abc import Container

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
    skip_blockades: Container[int] | None = None,
) -> tuple[bool, bool, bool, int | None]:
    # Четвёртый элемент кортежа — id завала, по которому в этом такте
    # реально был отправлен AKClear. None, если действие другое
    # (AKMove / AKClearArea / AKRescue / AKExtinguish / AKLoad). Нужен
    # трекеру прогресса расчистки в loop.py, чтобы инкрементировать
    # stale-счётчик ТОЛЬКО тем завалам, которые мы действительно пробовали
    # чистить (а не всем, что когда-либо попали в кэш).
    nav_node_id = get_nav_node(target_id, world_model)

    if nav_node_id is None:
        logger.warning("Я не нашёл узла графа для target_id=%d, сбрасываю цель", target_id)
        # Команду НЕ отправляю — вызывающий код перейдёт к исследованию.
        return False, True, False, None

    path = compute_path(world_model.road_graph, agent_node_id, nav_node_id)

    if not path:
        logger.warning(
            "Я не могу построить путь к target_id=%d (nav_node=%d), сбрасываю цель",
            target_id, nav_node_id,
        )
        # Команду НЕ отправляю — вызывающий код перейдёт к исследованию.
        return False, True, False, None

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
        skip_blockades=skip_blockades,
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
) -> tuple[bool, bool, bool, int | None]:
    if agent_type == AgentType.AMBULANCE_TEAM:
        return _ambulance_at_target(client, agent_state, tick, target_id, entity, world_model)

    if agent_type == AgentType.FIRE_BRIGADE:
        if entity is None:
            # Я не нашёл сущность — помечаю как недостижимую, чтобы не
            # зацикливаться. Команду НЕ отправляю: вызывающий код
            # перейдёт к исследованию и отправит AKMove.
            return False, True, False, None

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
                return False, True, False, None
            if buriedness is None or buriedness <= 0:
                logger.info(
                    "Я пропускаю спасение: buriedness=%s у target_id=%d, tick=%d",
                    buriedness, target_id, tick,
                )
                # Откопан — передаю цель медикам (убираю из своих задач).
                world_model.remove_task(target_id)
                return False, True, False, None
            client.send_rescue(tick, target_id)
            logger.info(
                "Я (пожарный) отправил AKRescue: target_id=%d, buriedness=%d, tick=%d",
                target_id, buriedness, tick,
            )
            return True, False, True, None

        fieryness = entity.raw_sensor_data.fieryness
        if fieryness is not None and fieryness not in {1, 2, 3}:
            logger.info(
                "Я пропускаю тушение: здание target_id=%d не горит (fieryness=%s), tick=%d",
                target_id, fieryness, tick,
            )
            # Команду НЕ отправляю — вызывающий код перейдёт к
            # исследованию и отправит AKMove вместо бесполезного AKRest.
            # unreachable=True заблокирует повторный выбор этой цели.
            return False, True, False, None
        if fieryness is None:
            logger.info(
                "Я не знаю fieryness здания target_id=%d, пропускаю тушение, tick=%d",
                target_id, tick,
            )
            # Данные могут появиться на следующем такте; помечаю как
            # недостижимое с коротким сроком блокировки.
            return False, True, False, None
        water = min(MAX_WATER_DISCHARGE, agent_state.resources.water_quantity)
        if water <= 0:
            logger.info(
                "Я пропускаю тушение: water=0 у target_id=%d, tick=%d",
                target_id, tick,
            )
            return False, True, False, None
        client.send_extinguish(tick, target_id, water=water)
        logger.info("Я отправил AKExtinguish: target_id=%d, water=%d, tick=%d", target_id, water, tick)
        return True, False, True, None

    if agent_type == AgentType.POLICE_FORCE:
        if entity is None:
            logger.info(
                "Я пропускаю расчистку: завал target_id=%d пропал из кэша, tick=%d",
                target_id, tick,
            )
            return False, True, False, None
        repair_cost = entity.raw_sensor_data.repair_cost
        if repair_cost is not None and repair_cost <= 0:
            logger.info(
                "Я пропускаю расчистку: завал target_id=%d уже расчищен (repair_cost=%s), tick=%d",
                target_id, repair_cost, tick,
            )
            return False, True, False, None
        # Проверяю геометрическую близость: ClearSimulator.isValid
        # отклоняет AKClear, если расстояние агент→полигон завала больше
        # clear.repair.distance (10 м). Если мы «далеко» по координатам
        # (узел крупный, полигон в другом его конце), переключаюсь на
        # AKClearArea — он работает по геометрии и не зависит от id.
        ax, ay = agent_state.position.x, agent_state.position.y
        bx, by = _resolve_entity_coords(target_id, world_model)
        use_clear_area = False
        if bx != 0 or by != 0:
            dist = math.hypot(bx - ax, by - ay)
            if dist > POLICE_CLEAR_MAX_DISTANCE:
                use_clear_area = True

        if use_clear_area:
            client.send_clear_area(tick, bx, by)
            logger.info(
                "Я отправил AKClearArea: target_id=%d, repair_cost=%s, "
                "dist=%.0f>%.0f, dest=(%d,%d), tick=%d",
                target_id, repair_cost, dist, POLICE_CLEAR_MAX_DISTANCE,
                bx, by, tick,
            )
            # По AKClearArea нельзя надёжно сопоставить уменьшение
            # repair_cost конкретному id — возвращаю None, трекер не
            # будет инкрементировать stale для этой команды.
            return True, False, True, None

        client.send_clear(tick, target_id)
        logger.info(
            "Я отправил AKClear: target_id=%d, repair_cost=%s, tick=%d",
            target_id, repair_cost, tick,
        )
        return True, False, True, target_id

    # Неизвестный тип агента — не отправляю команду, вызывающий код
    # перейдёт к исследованию.
    return False, True, False, None


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
    skip_blockades: Container[int] | None = None,
) -> tuple[bool, bool, bool, int | None]:
    dest_x, dest_y = tx, ty

    # Полицейский перед движением расчищает завал на ТЕКУЩЕМ узле, если он
    # там есть. Иначе TrafficAgent.step() на сервере увидит
    # insideBlockade()=true и выставит setMobile(false) — AKMove будет
    # проигнорирован, агент «зависает». Веса рёбер графа уже гонят A* в
    # обход известных завалов (refresh_blockade_weights), поэтому чистить
    # промежуточные завалы по пути НЕ требуется — их там быть не должно.
    # Завалы из skip_blockades пропускаются — трекер в loop.py определил,
    # что repair_cost не снижается (AKClear не проходит).
    if agent_type == AgentType.POLICE_FORCE:
        local_blockades = world_model.blockades_by_node.get(agent_node_id)
        if local_blockades:
            ax, ay = agent_state.position.x, agent_state.position.y
            any_clearable = False  # завалы есть и не все в skip
            for blockade_id in local_blockades:
                if skip_blockades and blockade_id in skip_blockades:
                    continue
                local_entity = world_model.tasks.get(blockade_id)
                if local_entity is None:
                    continue
                local_cost = local_entity.raw_sensor_data.repair_cost
                if local_cost is None or local_cost <= 0:
                    continue
                any_clearable = True

                # Геометрическая проверка: ClearSimulator.isValid требует,
                # чтобы расстояние от агента до полигона завала было меньше
                # clear.repair.distance. Если блокада ближе другого узла
                # того же графа — шлю AKClear по id; иначе AKClearArea.
                bx = local_entity.entity_x
                by = local_entity.entity_y
                if bx is not None and by is not None and (bx != 0 or by != 0):
                    dist = math.hypot(bx - ax, by - ay)
                    if dist > POLICE_CLEAR_MAX_DISTANCE:
                        client.send_clear_area(tick, int(bx), int(by))
                        logger.info(
                            "Я (полиция) отправил AKClearArea на завал "
                            "blockade_id=%d (dist=%.0f>%.0f) на пути к "
                            "target_id=%d, node=%d, tick=%d",
                            blockade_id, dist, POLICE_CLEAR_MAX_DISTANCE,
                            target_id, agent_node_id, tick,
                        )
                        return True, False, True, None

                client.send_clear(tick, blockade_id)
                logger.info(
                    "Я (полиция) расчищаю завал на пути к target_id=%d: "
                    "blockade_id=%d, repair_cost=%d, node=%d, tick=%d",
                    target_id, blockade_id, local_cost, agent_node_id, tick,
                )
                return True, False, True, blockade_id

            # Все локальные завалы в skip_blockades. AKMove сервер
            # проигнорирует (setMobile(false) внутри insideBlockade()),
            # поэтому вместо него шлю AKClearArea в сторону цели — это
            # действие ClearSimulator обрабатывает геометрически и оно
            # не зависит от id, что разрывает зависание.
            if not any_clearable and local_blockades:
                client.send_clear_area(tick, dest_x, dest_y)
                logger.info(
                    "Я (полиция) все локальные завалы в skip — отправляю "
                    "AKClearArea вместо AKMove: target_id=%d, dest=(%d,%d), "
                    "node=%d, tick=%d",
                    target_id, dest_x, dest_y, agent_node_id, tick,
                )
                return True, False, True, None

    client.send_move(tick, path, dest_x=dest_x, dest_y=dest_y)
    logger.info(
        "Я отправил AKMove: target_id=%d, nav=%d, path_len=%d, dest=(%d,%d), tick=%d",
        target_id, nav_node_id, len(path), dest_x, dest_y, tick,
    )
    return True, False, False, None


def try_clear_local_blockade(
    client: RCRSClient,
    tick: int,
    agent_node_id: int,
    world_model: WorldModel,
    agent_state: AgentState | None = None,
    skip_blockades: Container[int] | None = None,
) -> tuple[bool, int | None]:
    """Расчищает ближайший завал на ТЕКУЩЕМ узле полицейского.

    Проверяет только текущий узел (не соседние), чтобы не зацикливаться
    на удалённых завалах. Пропускает завалы с repair_cost=0 или None,
    а также завалы из skip_blockades (прогресс расчистки отсутствует).
    Если координаты завала известны и он дальше POLICE_CLEAR_MAX_DISTANCE
    от агента — отправляет AKClearArea вместо AKClear (иначе сервер
    отклонит команду как «не adjacent»).

    Возвращает (action_sent, attempted_blockade_id): флаг отправки
    команды и id завала, по которому послан именно AKClear (для трекера
    прогресса). При AKClearArea id=None.
    """
    blockade_set = world_model.blockades_by_node.get(agent_node_id)
    if not blockade_set:
        return False, None

    ax = ay = 0
    if agent_state is not None:
        ax, ay = agent_state.position.x, agent_state.position.y

    for blockade_id in blockade_set:
        if skip_blockades and blockade_id in skip_blockades:
            continue
        entity = world_model.tasks.get(blockade_id)
        if entity is None:
            continue
        repair_cost = entity.raw_sensor_data.repair_cost
        if repair_cost is None or repair_cost <= 0:
            continue

        bx = entity.entity_x
        by = entity.entity_y
        if (
            agent_state is not None
            and bx is not None and by is not None
            and (bx != 0 or by != 0)
        ):
            dist = math.hypot(bx - ax, by - ay)
            if dist > POLICE_CLEAR_MAX_DISTANCE:
                client.send_clear_area(tick, int(bx), int(by))
                logger.info(
                    "Я отправил AKClearArea (локальный завал далёк): "
                    "blockade_id=%d, dist=%.0f>%.0f, dest=(%d,%d), "
                    "node=%d, tick=%d",
                    blockade_id, dist, POLICE_CLEAR_MAX_DISTANCE,
                    int(bx), int(by), agent_node_id, tick,
                )
                return True, None

        client.send_clear(tick, blockade_id)
        logger.info(
            "Я расчищаю ближайший завал (AKClear): blockade_id=%d, "
            "repair_cost=%d, node=%d, tick=%d",
            blockade_id, repair_cost, agent_node_id, tick,
        )
        return True, blockade_id

    # На узле есть завалы, но все в skip_blockades — шлю AKClearArea
    # по средним координатам, чтобы разорвать зависание (AKMove без
    # расчистки тут невозможен: setMobile(false) на стороне сервера).
    sum_x = sum_y = n = 0
    for blockade_id in blockade_set:
        ent = world_model.tasks.get(blockade_id)
        if ent is None or ent.entity_x is None or ent.entity_y is None:
            continue
        if ent.entity_x == 0 and ent.entity_y == 0:
            continue
        sum_x += ent.entity_x
        sum_y += ent.entity_y
        n += 1
    if n > 0:
        cx, cy = sum_x // n, sum_y // n
        client.send_clear_area(tick, cx, cy)
        logger.info(
            "Я (полиция) все локальные завалы в skip — шлю AKClearArea "
            "по центроиду: dest=(%d,%d), node=%d, tick=%d",
            cx, cy, agent_node_id, tick,
        )
        return True, None

    return False, None


__all__ = [
    "get_nav_node",
    "dispatch_action",
    "try_clear_local_blockade",
]
