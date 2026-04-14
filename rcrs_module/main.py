from __future__ import annotations

import argparse
import logging
import math
import random
import struct
import signal
import sys
import time
from pathlib import Path


SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from action.navigation import (  # noqa: E402
    compute_path,
    fill_path_distances,
    nearest_refuge_path,
    pick_exploration_target,
    plan_exploration_path,
)
from action.selection import TargetSelector  # noqa: E402
from decision.filters.pre_filter import NeedRefugeException, PreFilterDispatcher  # noqa: E402
from decision.utility.aggregator import UtilityAggregator  # noqa: E402
from network.client import RCRSClient  # noqa: E402
from world.cache import WorldModel  # noqa: E402
from world.entities import AgentState, AgentType, EntityType, Position  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_debug.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# Параметры подключения
KERNEL_HOST: str  = "127.0.0.1"
KERNEL_PORT: int  = 7000
KERNEL_TIMEOUT: float = 30.0

DEFAULT_AGENT_TYPE: AgentType = AgentType.FIRE_BRIGADE
DEFAULT_AGENT_NAME: str = "diploma-agent"

# Параметры модели полезности
AVERAGE_SPEED: float = 70_000.0
SOCIAL_RADIUS: float = 30_000.0
MAX_WATER_DISCHARGE: int = 3_000
FIRE_EXTINGUISH_MAX_DISTANCE: float = 30_000.0
POLICE_CLEAR_MAX_DISTANCE: float = 10_000.0

# Длина случайного маршрута (количество узлов графа) — оставляю для отладочного fallback.
RANDOM_WALK_LENGTH: int = 50

# Длина маршрута исследования — сколько узлов агент отсылает ядру за раз.
# Ядро пересчитывает движение каждый такт, поэтому длинные маршруты избыточны.
EXPLORATION_PATH_LENGTH: int = 40

# Штраф координации
CLAIMED_TARGET_PENALTY: float = 0.3

# Я объявляю цель «застрявшей», если агент провёл возле неё STUCK_TICKS тактов без
# продвижения по графу. Причина — вероятнее всего завал на подходе: блокируем цель
# на STUCK_BLACKLIST_TICKS, чтобы не гонять агента в тупик.
STUCK_TICKS: int = 8
STUCK_BLACKLIST_TICKS: int = 60

# Если compute_path вернул пустой список — путь вообще не строится; я блокирую цель
# на меньший срок: возможно, граф ещё дополнится или полиция расчистит дорогу.
UNREACHABLE_BLACKLIST_TICKS: int = 30

# Глобальный флаг остановки
_shutdown_requested: bool = False


def _sigterm_handler(signum: int, frame: object) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Я получил SIGTERM (signal=%d), завершаю работу после текущего такта", signum)


signal.signal(signal.SIGTERM, _sigterm_handler)


def _random_walk(
    graph: "nx.Graph",
    start_node: int,
    length: int = RANDOM_WALK_LENGTH,
    visited: set[int] | None = None,
) -> list[int]:

    import networkx as nx

    if not graph.has_node(start_node):
        logger.warning("Я не нашёл start_node=%d в графе для random walk", start_node)
        return [start_node]

    if visited is not None:
        visited.add(start_node)

    path: list[int] = [start_node]
    current = start_node

    for _ in range(length):
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break

        if visited is not None:
            unvisited = [n for n in neighbors if n not in visited]
            next_node = random.choice(unvisited) if unvisited else random.choice(neighbors)
        else:
            next_node = random.choice(neighbors)

        path.append(next_node)
        current = next_node
        if visited is not None:
            visited.add(next_node)

    logger.debug(
        "Я построил random walk: start=%d, длина=%d узлов, visited=%d",
        start_node, len(path), len(visited) if visited is not None else 0,
    )
    return path


def _get_nav_node(target_id: int, world_model: WorldModel) -> int | None:
    entity = world_model.tasks.get(target_id)
    if entity is not None and entity.raw_sensor_data.position_on_edge is not None:
        pos = entity.raw_sensor_data.position_on_edge
        if world_model.road_graph.has_node(pos):
            return pos

    if world_model.road_graph.has_node(target_id):
        return target_id

    if entity is not None and entity.entity_x is not None and entity.entity_y is not None:
        best_node = None
        best_dist = float('inf')
        for node_id, data in world_model.road_graph.nodes(data=True):
            nx = data.get("x", 0)
            ny = data.get("y", 0)
            dist = math.hypot(entity.entity_x - nx, entity.entity_y - ny)
            if dist < best_dist:
                best_dist = dist
                best_node = node_id
        if best_node is not None:
            return best_node

    logger.warning(
        "Я не нашёл узла графа для target_id=%d (position_on_edge тоже не в графе и координаты отсутствуют)", target_id,
    )
    return None


def _dispatch_action(
    client: RCRSClient,
    agent_type: AgentType,
    agent_state: AgentState,
    tick: int,
    target_id: int,
    agent_node_id: int,
    world_model: WorldModel,
) -> tuple[bool, bool]:
    # Я возвращаю кортеж (target_valid, unreachable). target_valid=False означает,
    # что цель надо сбросить; unreachable=True — цель недостижима (нет пути в графе),
    # и её имеет смысл занести в чёрный список, чтобы не выбирать снова.
    nav_node_id = _get_nav_node(target_id, world_model)

    if nav_node_id is None:
        logger.warning(
            "Я не нашёл узла графа для target_id=%d, сбрасываю цель", target_id,
        )
        client.send_rest(tick)
        return False, True

    path = compute_path(world_model.road_graph, agent_node_id, nav_node_id)

    if not path:
        logger.warning(
            "Я не могу построить путь к target_id=%d (nav_node=%d), сбрасываю цель",
            target_id, nav_node_id,
        )
        client.send_rest(tick)
        return False, True

    entity = world_model.tasks.get(target_id)
    tx: int = 0
    ty: int = 0
    if entity is not None and entity.entity_x is not None and entity.entity_y is not None:
        tx, ty = entity.entity_x, entity.entity_y

    else:
        fallback_node = nav_node_id
        if entity is not None and entity.raw_sensor_data.position_on_edge is not None:
            fallback_node = entity.raw_sensor_data.position_on_edge
        target_attrs = world_model.road_graph.nodes.get(fallback_node, {})
        tx = target_attrs.get("x", 0)
        ty = target_attrs.get("y", 0)

    ax, ay = agent_state.position.x, agent_state.position.y
    eucl_dist = math.hypot(tx - ax, ty - ay)

    if agent_type in (AgentType.FIRE_BRIGADE, AgentType.POLICE_FORCE):
        max_dist = FIRE_EXTINGUISH_MAX_DISTANCE if agent_type == AgentType.FIRE_BRIGADE else POLICE_CLEAR_MAX_DISTANCE
        at_target = eucl_dist <= max_dist

    else:
        at_target = (agent_node_id == nav_node_id)

    if at_target:
        if agent_type == AgentType.AMBULANCE_TEAM:
            if entity is None:
                logger.warning(
                    "Я не нашёл сущность target_id=%d в кэше, сбрасываю цель (tick=%d)",
                    target_id, tick,
                )

                client.send_rest(tick)
                return False, False

            buriedness = entity.raw_sensor_data.buriedness

            logger.info(
                "ДИАГ_AT [AMBULANCE] at_target=True: target_id=%d, buriedness=%s, type=%s, tick=%d",
                target_id, buriedness, entity.type.value, tick,
            )

            # Я считаю None как «не завален» — сервер мог не отправить свойство
            if buriedness is None or buriedness == 0:
                if entity.type == EntityType.HUMAN:
                    world_model.tasks.pop(target_id, None)
                    logger.info(
                        "Я снял цель с агента target_id=%d: союзник уже откопан, tick=%d",
                        target_id, tick,
                    )
                    return False, False

                client.send_load(tick, target_id)

                world_model.tasks.pop(target_id, None)

                logger.info("Я отправил AKLoad: target_id=%d, tick=%d", target_id, tick)
                return False, False

            else:
                client.send_rescue(tick, target_id)
                logger.info("Я отправил AKRescue: target_id=%d, buriedness=%s, tick=%d", target_id, buriedness, tick)

        elif agent_type == AgentType.FIRE_BRIGADE:
            client.send_extinguish(tick, target_id, water=MAX_WATER_DISCHARGE)
            logger.info("Я отправил AKExtinguish: target_id=%d, tick=%d", target_id, tick)

        elif agent_type == AgentType.POLICE_FORCE:
            if entity is None:
                logger.info(
                    "Я пропускаю AKClear: завал target_id=%d пропал из кэша "
                    "(вероятно расчищен), tick=%d",
                    target_id, tick,
                )
                client.send_rest(tick)
                return False, False

            client.send_clear(tick, target_id)
            logger.info(
                "Я отправил AKClear: target_id=%d, dest=(%d,%d), "
                "repair_cost=%s, tick=%d",
                target_id, tx, ty,
                entity.raw_sensor_data.repair_cost, tick,
            )

    else:
        dest_x: int = tx
        dest_y: int = ty
        if agent_type in (AgentType.FIRE_BRIGADE, AgentType.POLICE_FORCE):

            if entity is not None and entity.entity_x is not None and entity.entity_y is not None:
                dest_x, dest_y = entity.entity_x, entity.entity_y

            elif entity is not None and entity.raw_sensor_data.position_on_edge is not None:
                pos_edge = entity.raw_sensor_data.position_on_edge
                edge_attrs = world_model.road_graph.nodes.get(pos_edge, {})
                fallback_x = edge_attrs.get("x", 0)
                fallback_y = edge_attrs.get("y", 0)

                if fallback_x != 0 or fallback_y != 0:
                    dest_x, dest_y = fallback_x, fallback_y
                    logger.debug(
                        "Я использую fallback координаты position_on_edge=%d: dest=(%d,%d)",
                        pos_edge, dest_x, dest_y,
                    )

        # Если полиция стоит на графе в нужной вершине, но евклидово далеко от завала
        # (например, завал в середине длинной дороги), AKMove с путём из одного узла
        # ничего не даст. Я пытаюсь расчистить зоной по координатам — у AKClearArea
        # больше шансов дотянуться, чем у стоячего AKMove.
        if (
            agent_type == AgentType.POLICE_FORCE
            and len(path) <= 1
            and entity is not None
        ):
            client.send_clear_area(tick, dest_x=dest_x, dest_y=dest_y)
            logger.info(
                "Я застрял у завала target_id=%d на узле=%d и отправил AKClearArea dest=(%d,%d), tick=%d",
                target_id, agent_node_id, dest_x, dest_y, tick,
            )
            return True, False

        client.send_move(tick, path, dest_x=dest_x, dest_y=dest_y)

        logger.debug(
            "Я отправил AKMove: nav_node=%d, path_len=%d, dest=(%d,%d)",
            nav_node_id, len(path), dest_x, dest_y,
        )

    return True, False


_CENTER_AGENT_TYPES: frozenset[AgentType] = frozenset({
    AgentType.FIRE_STATION,
    AgentType.AMBULANCE_CENTRE,
    AgentType.POLICE_OFFICE,
})


def _run_center_agent(client: RCRSClient, agent_type: AgentType) -> None:

    logger.info("Я запускаю центральный агент: type=%s", agent_type.value)

    try:
        while not _shutdown_requested:
            try:
                packet = client.receive_sense()
            except TimeoutError:
                if _shutdown_requested:
                    break
                continue
            except (ConnectionError, OSError) as exc:
                logger.error("Я потерял соединение центрального агента: %s", exc)
                break

            client.send_rest(packet.tick)
            logger.debug(
                "Центральный агент %s: AKRest, такт=%d",
                agent_type.value, packet.tick,
            )
    except KeyboardInterrupt:
        logger.info("Я завершаю центральный агент %s по сигналу", agent_type.value)
    finally:
        client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="RCRS diploma agent")

    parser.add_argument(
        "--agent-type",
        choices=[t.value for t in AgentType],
        default=DEFAULT_AGENT_TYPE.value,
        help="Тип агента RCRS (FIRE_BRIGADE / AMBULANCE_TEAM / POLICE_FORCE / FIRE_STATION / AMBULANCE_CENTRE / POLICE_OFFICE)",
    )

    parser.add_argument("--host", default=KERNEL_HOST, help="Адрес ядра RCRS Kernel")
    parser.add_argument("--port", type=int, default=KERNEL_PORT, help="Порт ядра RCRS Kernel")
    parser.add_argument("--name", default=DEFAULT_AGENT_NAME, help="Имя агента для рукопожатия")
    args = parser.parse_args()

    agent_type: AgentType = AgentType(args.agent_type)
    agent_name: str       = args.name

    client      = RCRSClient(host=args.host, port=args.port, timeout=KERNEL_TIMEOUT)
    world_model = WorldModel()

    dispatcher  = PreFilterDispatcher(work_rate=1.0, average_speed=AVERAGE_SPEED)
    aggregator  = UtilityAggregator(w_c=0.4, w_d=0.2, w_e=0.2, w_n=0.2)
    selector    = TargetSelector(c_switch=0.1)

    current_target_id: int | None = None

    # Я считаю неудачные попытки найти убежище, чтобы не спамить одинаковой ошибкой каждый такт
    NO_REFUGE_MAX_RETRIES: int = 5
    no_refuge_counter: int = 0

    visited_nodes: set[int] = set()

    # Я храню цель исследования между тактами: агент идёт к дальнему узлу карты и
    # только при достижении (или длительном застое) выбирает новую. Это разводит
    # агентов по разным районам, вместо того чтобы каждый такт случайно ходить по соседям.
    exploration_target_node: int | None = None
    exploration_start_tick: int = 0
    EXPLORATION_MAX_TICKS: int = 80  # Я меняю цель, если за это время до неё не дошёл — например, путь заблокирован.

    # Я отслеживаю прогресс к активной цели, чтобы занести «непроходимые» цели в чёрный список.
    stuck_target_id: int | None = None
    stuck_start_node: int | None = None
    stuck_counter: int = 0

    # target_id -> tick, до которого игнорируем цель (в чёрном списке).
    blacklisted_until: dict[int, int] = {}

    MAX_CONNECT_RETRIES: int = 20
    for attempt in range(1, MAX_CONNECT_RETRIES + 1):
        try:
            client.connect()
            break
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            if attempt == MAX_CONNECT_RETRIES:
                logger.error("Я исчерпал %d попыток подключения, завершаю работу: %s", MAX_CONNECT_RETRIES, exc)
                return
            logger.info("Я жду готовности ядра (попытка %d/%d): %s", attempt, MAX_CONNECT_RETRIES, exc)
            time.sleep(3)

    try:
        agent_id = client.handshake(agent_name, agent_type)
        logger.info("Я завершил рукопожатие: agent_id=%d, agent_type=%s", agent_id, agent_type.value)

    except (ConnectionError, TimeoutError, OSError) as exc:
        logger.error("Я не смог провести рукопожатие: %s", exc)
        client.disconnect()
        return

    if agent_type in _CENTER_AGENT_TYPES:
        _run_center_agent(client, agent_type)
        return

    try:
        while not _shutdown_requested:
            try:
                packet = client.receive_sense()
            except TimeoutError:
                if _shutdown_requested:
                    break

                logger.warning("Я не получил KASense в срок, продолжаю ожидание (такт)")

                continue

            except (ConnectionError, ConnectionRefusedError, OSError) as exc:
                logger.error("Я потерял соединение при получении данных: %s", exc)

                break

            tick_start = time.perf_counter()

            world_model.apply_perception(packet)

            agent_state   = packet.own_state
            agent_node_id = agent_state.position.entity_id
            visited_nodes.add(agent_node_id)

            if agent_node_id == 0:
                logger.warning(
                    "Я не нашёл позиции агента в такте %d (entity_id=0), отправляю AKRest",
                    packet.tick,
                )

                client.send_rest(packet.tick)
                continue

            if agent_state.hp is not None and agent_state.hp <= 0:
                logger.info("Я не могу действовать: агент мёртв, такт=%d", packet.tick)
                client.send_rest(packet.tick)
                current_target_id = None
                continue

            if agent_state.buriedness is not None and agent_state.buriedness > 0:
                logger.info(
                    "Я завален и зову на помощь! buriedness=%d, такт=%d",
                    agent_state.buriedness,
                    packet.tick,
                )
                client.send_rest(packet.tick)
                try:
                    say_data = struct.pack(">i", agent_state.id)
                    client.send_say(packet.tick, say_data)
                except (ConnectionError, OSError, struct.error) as exc:
                    logger.warning("Не смог крикнуть о помощи: %s", exc)
                current_target_id = None
                continue

            if agent_state.resources.is_transporting:
                logger.info(
                    "ДИАГ_TRANSPORT: agent_node=%d, refuge_ids=%s, graph_nodes=%d, graph_edges=%d, tick=%d",
                    agent_node_id,
                    world_model.refuge_ids[:5],
                    world_model.road_graph.number_of_nodes(),
                    world_model.road_graph.number_of_edges(),
                    packet.tick,
                )
                refuge_path = nearest_refuge_path(
                    world_model.road_graph,
                    agent_node_id,
                    world_model.refuge_ids,
                )
                if refuge_path:
                    no_refuge_counter = 0
                    refuge_node = refuge_path[-1]
                    ref_attrs = world_model.road_graph.nodes.get(refuge_node, {})
                    ref_x = ref_attrs.get("x", 0)
                    ref_y = ref_attrs.get("y", 0)
                    if agent_node_id == refuge_node:
                        client.send_unload(packet.tick)

                        logger.info(
                            "Я отправил AKUnload в убежище refuge_id=%d, такт=%d",
                            refuge_node, packet.tick,
                        )

                        current_target_id = None
                    else:
                        client.send_move(packet.tick, refuge_path, dest_x=ref_x, dest_y=ref_y)

                        logger.info(
                            "Я везу гражданского к убежищу refuge_id=%d, такт=%d",
                            refuge_node, packet.tick,
                        )

                        current_target_id = None
                else:
                    no_refuge_counter += 1
                    if no_refuge_counter >= NO_REFUGE_MAX_RETRIES:
                        logger.error(
                            "Я %d тактов подряд не могу найти убежище для выгрузки, такт=%d",
                            no_refuge_counter, packet.tick,
                        )
                        no_refuge_counter = 0
                    else:
                        logger.warning(
                            "Я не нашёл убежища для выгрузки гражданского (попытка %d/%d), такт=%d",
                            no_refuge_counter, NO_REFUGE_MAX_RETRIES, packet.tick,
                        )
                    client.send_rest(packet.tick)

                continue

            allowed_types = dispatcher._allowed_entity_types(agent_type)
            type_relevant = [
                e for e in world_model.tasks.values()
                if e.type in allowed_types
                and e.raw_sensor_data.position_on_edge not in world_model.refuge_ids
            ]
            fill_path_distances(
                world_model.road_graph,
                agent_node_id,
                type_relevant,
            )

            if packet.tick % 10 == 0 or len(type_relevant) > 0:
                logger.info(
                    "ДИАГ [%s] такт=%d: кэш=%d задач, type_relevant(%s)=%d",
                    agent_type.value, packet.tick,
                    len(world_model.tasks),
                    ",".join(t.value for t in sorted(allowed_types, key=lambda t: t.value)),
                    len(type_relevant),
                )

                if type_relevant and agent_type == AgentType.FIRE_BRIGADE:
                    sample = type_relevant[:5]
                    fieryness_info = [(e.id, e.raw_sensor_data.fieryness) for e in sample]

                    logger.info("ДИАГ fieryness: %s", fieryness_info)

            try:
                filtered_tasks = dispatcher.filter_tasks(agent_state, type_relevant)
            except NeedRefugeException:
                logger.info("Я отправляю пожарного в убежище из-за отсутствия воды")
                refuge_path = nearest_refuge_path(
                    world_model.road_graph,
                    agent_node_id,
                    world_model.refuge_ids,
                )

                if refuge_path:
                    client.send_move(packet.tick, refuge_path)
                    logger.info(
                        "Я отправил AKMove к убежищу refuge_id=%d (такт %d)",
                        refuge_path[-1], packet.tick,
                    )

                else:
                    client.send_rest(packet.tick)
                continue

            if filtered_tasks:
                logger.info(
                    "ДИАГ [%s] такт=%d: после фильтра=%d задач",
                    agent_type.value, packet.tick, len(filtered_tasks),
                )

            # Я чищу чёрный список от устаревших записей — блокировка имеет ограниченный срок.
            expired = [eid for eid, until in blacklisted_until.items() if until <= packet.tick]
            for eid in expired:
                blacklisted_until.pop(eid, None)
                logger.info("Я снял блокировку с target_id=%d на такте %d", eid, packet.tick)

            utilities: dict[int, float] = {}
            if filtered_tasks:

                for entity in filtered_tasks:
                    if entity.id in blacklisted_until:
                        logger.debug(
                            "Я пропускаю target_id=%d: цель в чёрном списке до такта %d",
                            entity.id, blacklisted_until[entity.id],
                        )
                        continue
                    try:
                        t_travel = entity.computed_metrics.path_distance / AVERAGE_SPEED

                    except ZeroDivisionError:
                        logger.warning("Я поймал деление на ноль при вычислении t_travel для entity_id=%s", entity.id)
                        t_travel = 0.0
                    buriedness = entity.raw_sensor_data.buriedness

                    try:
                        t_work = 0.0 if buriedness is None else buriedness / dispatcher.work_rate

                    except ZeroDivisionError:
                        logger.warning("Я поймал деление на ноль при вычислении t_work для entity_id=%s", entity.id)
                        t_work = 0.0

                    nav_id = entity.id

                    if entity.raw_sensor_data.position_on_edge is not None:
                        nav_id = entity.raw_sensor_data.position_on_edge
                    node_attrs = world_model.road_graph.nodes.get(nav_id, {})
                    target_position = Position(
                        entity_id=nav_id,
                        x=int(node_attrs.get("x", 0)),
                        y=int(node_attrs.get("y", 0)),
                    )

                    utility = aggregator.calculate_utility(
                        agent_state=agent_state,
                        entity=entity,
                        world_model=world_model,
                        target_position=target_position,
                        t_travel=t_travel,
                        t_work=t_work,
                        task_distance=entity.computed_metrics.path_distance,
                        social_radius=SOCIAL_RADIUS,
                    )
                    utilities[entity.id] = utility
                    logger.debug(
                        "Я рассчитал U_%d=%.4f для entity_id=%d",
                        agent_state.id,
                        utility,
                        entity.id,
                    )
            else:
                logger.info("Я не нашёл релевантных задач после фильтрации на такте %d", packet.tick)

            heard = packet.heard_target_ids
            if heard and utilities:
                for tid in heard:
                    if tid in utilities and tid != current_target_id:
                        old_u = utilities[tid]
                        utilities[tid] = old_u * CLAIMED_TARGET_PENALTY
                        logger.debug(
                            "Я снизил U для target_id=%d: %.4f → %.4f (занята другим агентом)",
                            tid, old_u, utilities[tid],
                        )

            selected_target_id = selector.select_best_target(current_target_id, utilities)

            if selected_target_id is None:
                logger.info("Я перевожу агента в режим ожидания на такте %d", packet.tick)
            elif selected_target_id != current_target_id:
                logger.info(
                    "Я выбрал новую цель: target_id=%s (такт %d)", selected_target_id, packet.tick
                )
            else:
                logger.info(
                    "Я сохраняю текущую цель: target_id=%s (такт %d)", current_target_id, packet.tick
                )

            current_target_id = selected_target_id

            # Я обновляю трекер застоя: если цель сменилась или агент продвинулся
            # по графу, сбрасываю счётчик; иначе считаю такты без прогресса.
            if current_target_id != stuck_target_id:
                stuck_target_id = current_target_id
                stuck_start_node = agent_node_id
                stuck_counter = 0
            elif current_target_id is not None:
                if stuck_start_node != agent_node_id:
                    stuck_start_node = agent_node_id
                    stuck_counter = 0
                else:
                    stuck_counter += 1

            try:
                if current_target_id is None:
                    # Я выбираю цель исследования один раз и иду к ней несколько тактов —
                    # так агенты расходятся по разным районам карты, а не кружат у дома.
                    need_new_target = (
                        exploration_target_node is None
                        or exploration_target_node == agent_node_id
                        or not world_model.road_graph.has_node(exploration_target_node)
                        or (packet.tick - exploration_start_tick) >= EXPLORATION_MAX_TICKS
                    )
                    if need_new_target:
                        # Я использую id агента как сид, чтобы разные агенты выбирали разные направления.
                        rng = random.Random(agent_state.id * 1_000_003 + packet.tick)
                        exploration_target_node = pick_exploration_target(
                            world_model.road_graph,
                            agent_node_id,
                            visited=visited_nodes,
                            rng=rng,
                        )
                        exploration_start_tick = packet.tick
                        if exploration_target_node is not None:
                            logger.info(
                                "Я выбрал новую цель исследования node=%d (из узла=%d), такт=%d",
                                exploration_target_node, agent_node_id, packet.tick,
                            )

                    exp_path: list[int] = []
                    if exploration_target_node is not None:
                        exp_path = plan_exploration_path(
                            world_model.road_graph,
                            agent_node_id,
                            exploration_target_node,
                            max_steps=EXPLORATION_PATH_LENGTH,
                        )

                    if exp_path and len(exp_path) > 1:
                        client.send_move(packet.tick, exp_path)
                        logger.info(
                            "Я иду к цели исследования node=%d: %d шагов, такт=%d",
                            exploration_target_node, len(exp_path), packet.tick,
                        )
                    else:
                        # Я откатываюсь к random walk, если не смог построить путь к далёкой цели.
                        exploration_target_node = None
                        rw_path = _random_walk(world_model.road_graph, agent_node_id, visited=visited_nodes)
                        if len(rw_path) > 1:
                            client.send_move(packet.tick, rw_path)
                            logger.info(
                                "Я исследую карту random walk: %d узлов, такт=%d",
                                len(rw_path), packet.tick,
                            )
                        else:
                            client.send_rest(packet.tick)
                            logger.warning(
                                "Я не могу исследовать карту — тупик node=%d, такт=%d",
                                agent_node_id, packet.tick,
                            )
                else:
                    target_valid, unreachable = _dispatch_action(
                        client=client,
                        agent_type=agent_type,
                        agent_state=agent_state,
                        tick=packet.tick,
                        target_id=current_target_id,
                        agent_node_id=agent_node_id,
                        world_model=world_model,
                    )

                    # Я заношу цель в чёрный список, если агент застрял рядом с ней без прогресса.
                    # Скорее всего путь физически блокирован (завал у входа в здание).
                    if target_valid and stuck_counter >= STUCK_TICKS:
                        logger.warning(
                            "Я застрял у target_id=%d на узле=%d %d тактов — заношу в чёрный список на %d тактов",
                            current_target_id, agent_node_id, stuck_counter, STUCK_BLACKLIST_TICKS,
                        )
                        blacklisted_until[current_target_id] = packet.tick + STUCK_BLACKLIST_TICKS
                        current_target_id = None
                        stuck_target_id = None
                        stuck_counter = 0
                    elif unreachable and current_target_id is not None:
                        # Нет пути по графу — блокирую цель на короткий срок, возможно путь появится.
                        logger.info(
                            "Я заношу недостижимую цель target_id=%d в чёрный список на %d тактов",
                            current_target_id, UNREACHABLE_BLACKLIST_TICKS,
                        )
                        blacklisted_until[current_target_id] = packet.tick + UNREACHABLE_BLACKLIST_TICKS
                        current_target_id = None
                        stuck_target_id = None
                        stuck_counter = 0
                    elif not target_valid:
                        if current_target_id is not None:
                            world_model.tasks.pop(current_target_id, None)
                        current_target_id = None
                        stuck_target_id = None
                        stuck_counter = 0
            except (ConnectionError, ConnectionRefusedError, TimeoutError, OSError) as exc:
                logger.error("Я потерял соединение при отправке команды: %s", exc)
                break

            if current_target_id is not None:
                try:
                    say_data = struct.pack(">i", current_target_id)
                    client.send_say(packet.tick, say_data)
                except (ConnectionError, OSError) as exc:
                    logger.warning("Я не смог отправить AKSay: %s", exc)

            tick_elapsed = time.perf_counter() - tick_start
            if tick_elapsed > 0.1:
                logger.warning(
                    "Я превысил бюджет времени тика %d: %.3f с", packet.tick, tick_elapsed
                )

    except KeyboardInterrupt:
        logger.info("Я получил сигнал остановки и завершаю работу")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
