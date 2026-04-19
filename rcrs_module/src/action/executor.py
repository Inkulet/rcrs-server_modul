from __future__ import annotations

import logging
import math
from typing import Optional

from config import (
    FIRE_EXTINGUISH_MAX_DISTANCE,
    MAX_WATER_DISCHARGE,
    POLICE_CLEAR_MAX_DISTANCE,
)
from action.navigation import compute_path, find_nearest_node
from action.police_geometry import (
    intersects_area_edge,
    intersects_blockade,
    nearest_apex,
    scale_clear_vector,
)
from network.client import RCRSClient
from world.cache import WorldModel
from world.entities import AgentState, AgentType, EntityType


# Количество подряд одинаковых целей расчистки, после которого полицейский
# форсированно идёт пешком в направлении клика (anti-stuck). Аналог
# forced_move в DefaultExtendActionClear._get_neighbour_position_action.
_SAME_CLEAR_FORCE_MOVE: int = 3

# Порог «одинаковости» точек расчистки (мм). Совпадение двух clear_xy
# в пределах этого радиуса считается повтором.
_SAME_CLEAR_EPS: int = 1000

# Состояние anti-stuck на агента: agent_id → (last_clear_xy, count).
# Модульного уровня — агент-процесс ровно один на инстанс executor-модуля
# (см. main.py), поэтому конфликтов между агентами не возникает.
_police_same_clear_state: dict[int, tuple[tuple[int, int], int]] = {}


logger = logging.getLogger(__name__)


def _resolve_entity_coords(
    entity_id: int,
    world_model: WorldModel,
) -> tuple[int, int]:
    entity = world_model.tasks.get(entity_id)

    if (
        entity is not None
        and entity.entity_x is not None and entity.entity_y is not None
        and (entity.entity_x != 0 or entity.entity_y != 0)
    ):
        return entity.entity_x, entity.entity_y

    if entity is not None and entity.raw_sensor_data.position_on_edge is not None:
        pos_edge = entity.raw_sensor_data.position_on_edge
        edge_attrs = world_model.road_graph.nodes.get(pos_edge, {})
        ex: int = edge_attrs.get("x", 0)
        ey: int = edge_attrs.get("y", 0)
        if ex != 0 or ey != 0:
            return ex, ey

    return 0, 0


def _node_xy(world_model: WorldModel, node_id: int) -> Optional[tuple[int, int]]:
    attrs = world_model.road_graph.nodes.get(node_id, {})
    x = attrs.get("x")
    y = attrs.get("y")
    if x is None or y is None:
        return None
    return int(x), int(y)


def _select_blockade_apex_clear(
    world_model: WorldModel,
    agent_state: AgentState,
    blockade_ids: set[int] | list[int],
    aim_xy: Optional[tuple[int, int]],
) -> Optional[tuple[int, int, int, int]]:
    """Выбирает завал и вычисляет точку расчистки (apex-таргетинг).

    Правила (аналог `_get_area_clear_action` / `_get_neighbour_position_action`
    из `default_extend_action_clear.py`):
    1. repair_cost должен быть > 0.
    2. Ближайший apex завала должен лежать в пределах POLICE_CLEAR_MAX_DISTANCE
       от агента.
    3. Если `aim_xy` задана (движемся к следующему узлу) — линия
       (agent → aim_xy) должна пересекать полигон завала. Без aim_xy
       (режим «нет цели, расчищаю всё что рядом») этот шаг пропускается.

    Возвращает (blockade_id, repair_cost, clear_x, clear_y), где clear_x/y —
    точка на конусе расчистки в направлении ближайшего apex.
    """
    ax, ay = agent_state.position.x, agent_state.position.y
    best: Optional[tuple[float, int, int, int, int]] = None  # (dist, bid, cost, cx, cy)

    for blockade_id in blockade_ids:
        entity = world_model.tasks.get(blockade_id)
        if entity is None:
            continue
        cost = entity.raw_sensor_data.repair_cost
        if cost is None or cost <= 0:
            continue

        apexes = entity.raw_sensor_data.apexes
        # Fallback: если apex-ов нет (не пришли в ChangeSet), беру центроид —
        # исторический режим. Лучше «расчистить приблизительно», чем
        # пропустить завал.
        if apexes is None or len(apexes) < 6:
            bx, by = _resolve_entity_coords(blockade_id, world_model)
            if bx == 0 and by == 0:
                continue
            dist = math.hypot(bx - ax, by - ay)
            if dist > POLICE_CLEAR_MAX_DISTANCE:
                continue
            clear_x, clear_y = scale_clear_vector(
                (ax, ay), (bx, by), POLICE_CLEAR_MAX_DISTANCE,
            )
        else:
            apex_result = nearest_apex((ax, ay), apexes)
            if apex_result is None:
                continue
            apex_x, apex_y, apex_dist = apex_result
            if apex_dist > POLICE_CLEAR_MAX_DISTANCE:
                continue
            if aim_xy is not None and not intersects_blockade(
                (ax, ay), aim_xy, apexes,
            ):
                # Завал не на пути к следующему узлу — пропускаем.
                continue
            clear_x, clear_y = scale_clear_vector(
                (ax, ay), (apex_x, apex_y), POLICE_CLEAR_MAX_DISTANCE,
            )
            dist = apex_dist

        if best is None or dist < best[0]:
            best = (dist, blockade_id, cost, clear_x, clear_y)

    if best is None:
        return None
    _d, bid, cost, cx, cy = best
    return bid, cost, cx, cy


def _police_send_clear_area_xy(
    client: RCRSClient,
    tick: int,
    agent_state: AgentState,
    clear_x: int,
    clear_y: int,
) -> None:
    ax, ay = agent_state.position.x, agent_state.position.y
    if clear_x == ax and clear_y == ay:
        # Страховка от нулевого вектора (scale_clear_vector сам это
        # обрабатывает, но защищаемся дополнительно).
        clear_x = ax + 1000
        clear_y = ay
    client.send_clear_area(tick, clear_x, clear_y)


def _register_and_check_same_clear(
    agent_id: int,
    clear_xy: tuple[int, int],
) -> bool:
    """Anti-stuck: регистрирует точку расчистки и возвращает True,
    если агент подряд >= _SAME_CLEAR_FORCE_MOVE раз целится в ту же точку.
    При срабатывании сбрасывает счётчик.
    """
    prev = _police_same_clear_state.get(agent_id)
    if prev is None:
        _police_same_clear_state[agent_id] = (clear_xy, 1)
        return False
    prev_xy, cnt = prev
    dx = clear_xy[0] - prev_xy[0]
    dy = clear_xy[1] - prev_xy[1]
    if abs(dx) <= _SAME_CLEAR_EPS and abs(dy) <= _SAME_CLEAR_EPS:
        cnt += 1
        if cnt >= _SAME_CLEAR_FORCE_MOVE:
            _police_same_clear_state[agent_id] = (clear_xy, 0)
            return True
        _police_same_clear_state[agent_id] = (prev_xy, cnt)
    else:
        _police_same_clear_state[agent_id] = (clear_xy, 1)
    return False


def reset_clear_state(agent_id: int) -> None:
    _police_same_clear_state.pop(agent_id, None)


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
) -> tuple[bool, bool, bool, int | None]:
    nav_node_id = get_nav_node(target_id, world_model)

    if nav_node_id is None:
        logger.warning("Я не нашёл узла графа для target_id=%d, сбрасываю цель", target_id)
        return False, True, False, None

    path = compute_path(world_model.road_graph, agent_node_id, nav_node_id)

    if not path:
        logger.warning(
            "Я не могу построить путь к target_id=%d (nav_node=%d), сбрасываю цель",
            target_id, nav_node_id,
        )
        return False, True, False, None

    entity = world_model.tasks.get(target_id)
    tx, ty = _resolve_entity_coords(target_id, world_model)

    if tx == 0 and ty == 0:
        fallback_attrs = world_model.road_graph.nodes.get(nav_node_id, {})
        tx = fallback_attrs.get("x", 0)
        ty = fallback_attrs.get("y", 0)

    ax, ay = agent_state.position.x, agent_state.position.y
    eucl_dist = math.hypot(tx - ax, ty - ay)

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
) -> tuple[bool, bool, bool, int | None]:
    if agent_type == AgentType.AMBULANCE_TEAM:
        return _ambulance_at_target(client, agent_state, tick, target_id, entity, world_model)

    if agent_type == AgentType.FIRE_BRIGADE:
        if entity is None:
            return False, True, False, None

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

        # Apex-таргетинг по ЕДИНСТВЕННОМУ заданному завалу — без ограничения
        # на aim_xy (мы уже рядом с целью, линия пересечения не нужна).
        pick = _select_blockade_apex_clear(
            world_model, agent_state, [target_id], aim_xy=None,
        )
        if pick is None:
            # Apex-расчёт не смог прицелиться (нет apex-ов И нет координат)
            # — fallback на старый центроидный режим.
            ax, ay = agent_state.position.x, agent_state.position.y
            cx, cy = (tx, ty) if (tx != 0 or ty != 0) else (ax + 1000, ay)
            _police_send_clear_area_xy(client, tick, agent_state, cx, cy)
            logger.info(
                "Я (полиция) AKClearArea fallback-центроид: target_id=%d, "
                "dest=(%d,%d), tick=%d", target_id, cx, cy, tick,
            )
            return True, False, True, target_id

        bid, _cost, cx, cy = pick
        forced = _register_and_check_same_clear(agent_state.id, (cx, cy))
        if forced:
            # Anti-stuck: вместо N-й AKClearArea на ту же точку идём туда
            # пешком (на один такт). Сбросив счётчик, следующий такт
            # вернётся к расчистке.
            client.send_move(tick, [agent_state.position.entity_id], dest_x=cx, dest_y=cy)
            logger.info(
                "Я (полиция) anti-stuck AKMove: target_id=%d одинаковая точка "
                "расчистки %d раз, иду к (%d,%d), tick=%d",
                target_id, _SAME_CLEAR_FORCE_MOVE, cx, cy, tick,
            )
            return True, False, True, None

        _police_send_clear_area_xy(client, tick, agent_state, cx, cy)
        logger.info(
            "Я (полиция) AKClearArea apex-target: target_id=%d (real=%d), "
            "clear=(%d,%d), repair_cost=%s, tick=%d",
            target_id, bid, cx, cy, repair_cost, tick,
        )
        return True, False, True, bid

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

    if entity_type == EntityType.HUMAN or entity.is_ally:  # type: ignore[union-attr]
        # Шаг 12: AKLoad валиден ТОЛЬКО на CIVILIAN. Живой союзник (пожарный/
        # медик/полицейский, который был заваленным) сам встанет и уедет —
        # в машину его грузить нельзя. Снимаю цель и переключаюсь.
        world_model.remove_task(target_id)
        logger.info(
            "Я снял цель target_id=%d: союзник/HUMAN — AKLoad не применим, "
            "tick=%d", target_id, tick,
        )
        return False, True, False

    if entity_type != EntityType.CIVILIAN:
        logger.warning(
            "Я пропускаю AKLoad: target_id=%d не CIVILIAN (type=%s), tick=%d",
            target_id, entity_type.value, tick,
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
) -> tuple[bool, bool, bool, int | None]:
    dest_x, dest_y = tx, ty

    # Полицейский перед движением расчищает завал на ТЕКУЩЕМ узле ТОЛЬКО
    # если завал реально перекрывает путь:
    #   1) линия (agent → next_node) пересекает полигон завала (apex-проверка),
    #   2) следовательно TrafficAgent.step() выставит insideBlockade()=true.
    # Если завал «на встречной полосе» и линия пути его не задевает —
    # отправляю обычный AKMove, расчистка не нужна.
    if agent_type == AgentType.POLICE_FORCE:
        blockade_set = world_model.blockades_by_node.get(agent_node_id)
        if blockade_set:
            # Целюсь в СЛЕДУЮЩИЙ узел пути, а не в финиш: если дорога
            # поворачивает, прямой конус на dest_x/y улетает в здания
            # и не задевает завал. path[1] всегда в направлении движения.
            aim_x, aim_y = dest_x, dest_y
            if path and len(path) > 1:
                nxt = _node_xy(world_model, path[1])
                if nxt is not None:
                    aim_x, aim_y = nxt

            pick = _select_blockade_apex_clear(
                world_model, agent_state, blockade_set,
                aim_xy=(aim_x, aim_y),
            )
            if pick is not None:
                bid, local_cost, cx, cy = pick
                forced = _register_and_check_same_clear(
                    agent_state.id, (cx, cy),
                )
                if forced:
                    # Anti-stuck: форсированный AKMove в сторону clear-точки.
                    client.send_move(
                        tick, [agent_node_id], dest_x=cx, dest_y=cy,
                    )
                    logger.info(
                        "Я (полиция) anti-stuck AKMove на пути к target_id=%d: "
                        "одинаковая расчистка %dx, иду к (%d,%d), tick=%d",
                        target_id, _SAME_CLEAR_FORCE_MOVE, cx, cy, tick,
                    )
                    return True, False, True, None
                _police_send_clear_area_xy(client, tick, agent_state, cx, cy)
                logger.info(
                    "Я (полиция) AKClearArea apex на пути к target_id=%d: "
                    "blockade=%d, cost=%d, clear=(%d,%d), aim=(%d,%d), tick=%d",
                    target_id, bid, local_cost, cx, cy, aim_x, aim_y, tick,
                )
                return True, False, True, bid

    # Шаг 9: FIRE_BRIGADE/AMBULANCE_TEAM — если по пути (не в конечной
    # точке) встречается узел с живым завалом, сокращаем путь до
    # ПРЕДЫДУЩЕГО узла и шлём AKSay с id завала — сигнал полиции.
    # Так агент «останавливается в 1-2 узлах от завала, оставляя место
    # для полиции», как в ADF-ноутбуке Kobe-теста.
    if agent_type in (AgentType.FIRE_BRIGADE, AgentType.AMBULANCE_TEAM):
        blocking_idx = _find_blockade_on_path(path, world_model)
        # blocking_idx == len(path)-1 — завал в самой целевой Area
        # (например, мы идём к гражданскому, который там лежит). В этом
        # случае НЕ сокращаем — пусть агент доедет до цели, а полиция
        # его отдельно разблокирует.
        if blocking_idx is not None and blocking_idx < len(path) - 1 and blocking_idx >= 1:
            truncated = path[:blocking_idx]
            if len(truncated) >= 2:
                blocked_node = path[blocking_idx]
                blk_ids = world_model.blockades_by_node.get(blocked_node, set())
                blk_for_say = next(iter(blk_ids), 0)
                stop_node_attrs = world_model.road_graph.nodes.get(
                    truncated[-1], {},
                )
                stop_x = int(stop_node_attrs.get("x", dest_x))
                stop_y = int(stop_node_attrs.get("y", dest_y))
                client.send_move(tick, truncated, dest_x=stop_x, dest_y=stop_y)
                try:
                    import struct as _struct
                    if blk_for_say > 0:
                        client.send_say(
                            tick, _struct.pack(">i", blk_for_say),
                        )
                except (ConnectionError, OSError, Exception) as exc:  # noqa: BLE001
                    logger.debug("Я не смог AKSay о завале: %s", exc)
                logger.info(
                    "Я (%s) торможу перед завалом на пути к target_id=%d: "
                    "узел=%d заблокирован blockade_id=%d, стою в node=%d, "
                    "tick=%d", agent_type.value, target_id,
                    blocked_node, blk_for_say, truncated[-1], tick,
                )
                return True, False, False, None

    client.send_move(tick, path, dest_x=dest_x, dest_y=dest_y)
    logger.info(
        "Я отправил AKMove: target_id=%d, nav=%d, path_len=%d, dest=(%d,%d), tick=%d",
        target_id, nav_node_id, len(path), dest_x, dest_y, tick,
    )
    return True, False, False, None


def _find_blockade_on_path(
    path: list[int], world_model: WorldModel,
) -> Optional[int]:
    """Индекс первого узла пути, содержащего живой завал (repair_cost > 0)."""
    for i in range(1, len(path)):
        node_id = path[i]
        blk_ids = world_model.blockades_by_node.get(node_id)
        if not blk_ids:
            continue
        for blk_id in blk_ids:
            ent = world_model.tasks.get(blk_id)
            if ent is None:
                continue
            cost = ent.raw_sensor_data.repair_cost
            if cost is None or cost > 0:
                return i
    return None


def try_clear_local_blockade(
    client: RCRSClient,
    tick: int,
    agent_node_id: int,
    world_model: WorldModel,
    agent_state: AgentState,
) -> tuple[bool, int | None]:
    """Расчищает ближайший завал на ТЕКУЩЕМ узле полицейского.

    Apex-таргетинг: выбирает завал с ближайшим apex в пределах
    POLICE_CLEAR_MAX_DISTANCE, направляет конус по единичному вектору
    агент → apex. `aim_xy=None`: без цели движения фильтр по пересечению
    линии не применяем — лишь бы апекс был в радиусе расчистки.
    """
    blockade_set = world_model.blockades_by_node.get(agent_node_id)
    if not blockade_set:
        return False, None

    pick = _select_blockade_apex_clear(
        world_model, agent_state, blockade_set, aim_xy=None,
    )
    if pick is None:
        return False, None

    bid, cost, cx, cy = pick
    forced = _register_and_check_same_clear(agent_state.id, (cx, cy))
    if forced:
        client.send_move(tick, [agent_node_id], dest_x=cx, dest_y=cy)
        logger.info(
            "Я (полиция) anti-stuck AKMove без цели: одинаковая точка "
            "расчистки %dx, иду к (%d,%d), tick=%d",
            _SAME_CLEAR_FORCE_MOVE, cx, cy, tick,
        )
        return True, None

    _police_send_clear_area_xy(client, tick, agent_state, cx, cy)
    logger.info(
        "Я расчищаю ближайший завал (AKClearArea apex): blockade=%d, "
        "cost=%d, clear=(%d,%d), node=%d, tick=%d",
        bid, cost, cx, cy, agent_node_id, tick,
    )
    return True, bid


__all__ = [
    "get_nav_node",
    "dispatch_action",
    "try_clear_local_blockade",
    "reset_clear_state",
]
