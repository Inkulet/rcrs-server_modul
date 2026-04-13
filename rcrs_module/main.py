from __future__ import annotations

"""В этом модуле я запускаю главный цикл агента и конвейер принятия решений.

Я реализую строгий конвейер: Perception → World → Navigation → Filter → Utility → Action.
Каждый такт агент получает свежий пакет восприятия, обновляет модель мира,
пересчитывает реальные дистанции по графу и принимает решение строго
на основе функции полезности U_ij.
"""

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

from action.navigation import compute_path, fill_path_distances, nearest_refuge_path  # noqa: E402
from action.selection import TargetSelector  # noqa: E402
from decision.filters.pre_filter import NeedRefugeException, PreFilterDispatcher  # noqa: E402
from decision.utility.aggregator import UtilityAggregator  # noqa: E402
from network.client import RCRSClient  # noqa: E402
from world.cache import WorldModel  # noqa: E402
from world.entities import AgentState, AgentType, Position  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Параметры подключения ---
KERNEL_HOST: str  = "127.0.0.1"
KERNEL_PORT: int  = 7000
KERNEL_TIMEOUT: float = 30.0

# --- Тип агента и имя для рукопожатия (дефолты; переопределяются через CLI) ---
# Я задаю дефолтный тип агента. При запуске нескольких процессов используйте
# флаг --agent-type, чтобы запустить разные типы агентов независимо.
DEFAULT_AGENT_TYPE: AgentType = AgentType.FIRE_BRIGADE
DEFAULT_AGENT_NAME: str = "diploma-agent"

# --- Параметры модели полезности ---
# Я задаю среднюю скорость движения агента в мм/такт.
# Реальное значение для RCRS: ~70 000 мм/такт (70 м/такт при шаге симуляции ~1 с).
# Подлежит калибровке по данным конкретной карты в ходе интеграционных испытаний.
AVERAGE_SPEED: float = 70_000.0

# Я задаю радиус для социального фактора f_social в единицах карты (мм).
SOCIAL_RADIUS: float = 30_000.0

# Я задаю максимальный объём воды для команды AKExtinguish.
# Значение 10 000 соответствует вместимости стандартного пожарного агента в RCRS.
# Ядро само ограничит фактически отправляемое количество воды текущим запасом агента.
MAX_WATER_DISCHARGE: int = 10_000

# Я задаю максимальный радиус действия команды AKExtinguish (мм).
# В RCRS параметр fire.extinguish.max-distance обычно равен 30 000 мм (30 м).
# Ядро проигнорирует AKExtinguish, если евклидово расстояние до цели превышает этот радиус.
FIRE_EXTINGUISH_MAX_DISTANCE: float = 30_000.0

# Я задаю максимальный радиус действия команды AKClear для полиции (мм).
# В RCRS параметр clear.repair.distance обычно равен 10 000 мм (10 м).
# Ядро проигнорирует AKClear, если евклидово расстояние до завала превышает этот радиус.
POLICE_CLEAR_MAX_DISTANCE: float = 10_000.0

# --- Длина случайного маршрута (количество узлов графа) ---
# Я моделирую поведение Java AbstractSampleAgent.randomWalk():
# агент строит случайный маршрут через RANDOM_WALK_LENGTH соседних узлов графа,
# чтобы исследовать карту и обнаружить пожары, завалы и гражданских.
# Без этого агент стоит на месте, когда в зоне видимости нет задач.
RANDOM_WALK_LENGTH: int = 50

# --- Штраф координации: множитель полезности для задач, уже занятых другими агентами ---
# Я применяю этот коэффициент к U_ij, если target_id был услышан от другого агента
# через AKSay. Значение 0.3 означает, что занятая цель сохраняет 30% полезности —
# агент предпочтёт свободную цель, но в крайнем случае пойдёт к занятой.
CLAIMED_TARGET_PENALTY: float = 0.3

# --- Глобальный флаг остановки (используется обработчиком SIGTERM) ---
_shutdown_requested: bool = False


def _sigterm_handler(signum: int, frame: object) -> None:
    """Я устанавливаю флаг остановки по сигналу SIGTERM, чтобы выйти чисто."""
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
    """Здесь я строю случайный маршрут с памятью посещённых узлов.

    Я реализую улучшенный аналог Java AbstractSampleAgent.randomWalk():
    начиная от текущей позиции агента, я приоритетно выбираю ещё не
    посещённые узлы среди соседей. Если все соседи уже посещены — выбираю
    случайного, чтобы не застрять. Множество visited обновляется in-place
    и сохраняется между тактами, предотвращая повторное исследование
    одних и тех же районов карты.

    Если у узла нет соседей (тупик) — маршрут прерывается досрочно.
    Маршрут всегда начинается с start_node, как того требует AKMove.
    """
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
            # Я попал в тупик — возвращаю то, что построил.
            break

        if visited is not None:
            # Я приоритетно выбираю непосещённые узлы, чтобы агент
            # исследовал новые районы карты, а не ходил по кругу.
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
    """Здесь я определяю узел графа для навигации к цели.

    Для гражданских (CIVILIAN) entity_id — это сам объект гражданского, которого
    нет в дорожном графе. Я навигирую к зданию/дороге, на которой находится
    гражданский (PROP_POSITION → position_on_edge). Для зданий и завалов
    entity_id совпадает с узлом графа напрямую.

    Возвращаю None, если ни position_on_edge, ни target_id не являются
    узлами графа — вызывающий код должен сбросить цель, чтобы агент
    не застрял навсегда на AKRest.
    """
    entity = world_model.tasks.get(target_id)
    if entity is not None and entity.raw_sensor_data.position_on_edge is not None:
        pos = entity.raw_sensor_data.position_on_edge
        # Я проверяю, что узел существует в графе — position_on_edge может быть
        # устаревшим или указывать на объект вне зоны карты.
        if world_model.road_graph.has_node(pos):
            return pos
    # Я проверяю, является ли target_id узлом графа напрямую (здания, дороги).
    # Для гражданских и завалов target_id отсутствует в графе — без position_on_edge
    # навигация невозможна и цель нужно сбросить.
    if world_model.road_graph.has_node(target_id):
        return target_id
    logger.warning(
        "Я не нашёл узла графа для target_id=%d (position_on_edge тоже не в графе)", target_id,
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
) -> bool:
    """Здесь я отправляю типизированную команду в зависимости от типа агента.

    Логика:
    - Я разделяю nav_node_id (узел графа для навигации) и target_id (ID объекта действия).
      Для гражданского: nav_node_id = здание, где он находится; target_id = ID гражданского.
      Для здания/завала: nav_node_id = target_id (здание само является узлом графа).
    - Агент «у цели», когда евклидово расстояние от реальной позиции агента до
      координат сущности не превышает радиус действия (fire/clear distance).
    - При достижении: выполняю целевое действие по типу агента (UC-7).
    - Иначе: отправляю AKMove по кратчайшему пути к nav_node_id.
      Для полиции/пожарных на той же дороге — AKMove с dest_x,dest_y к сущности.

    Возвращаю True, если цель остаётся валидной; False — если цель нужно сбросить
    (нет узла графа или нет пути), чтобы агент не застрял на AKRest навсегда.
    """
    # Я определяю узел навигации — для гражданских это здание, где они находятся.
    nav_node_id = _get_nav_node(target_id, world_model)

    if nav_node_id is None:
        logger.warning(
            "Я не нашёл узла графа для target_id=%d, сбрасываю цель", target_id,
        )
        client.send_rest(tick)
        return False

    path = compute_path(world_model.road_graph, agent_node_id, nav_node_id)

    if not path:
        logger.warning(
            "Я не могу построить путь к target_id=%d (nav_node=%d), сбрасываю цель",
            target_id, nav_node_id,
        )
        client.send_rest(tick)
        return False

    # Я получаю координаты цели: приоритет у entity_x/entity_y (реальная позиция
    # сущности на карте), fallback — центр навигационного узла графа.
    entity = world_model.tasks.get(target_id)
    tx: int = 0
    ty: int = 0
    if entity is not None and entity.entity_x is not None and entity.entity_y is not None:
        tx, ty = entity.entity_x, entity.entity_y
    else:
        # Я использую fallback: для завалов entity_x/entity_y часто None,
        # беру координаты из position_on_edge (дорога, на которой лежит объект),
        # а если и его нет — центр навигационного узла графа.
        fallback_node = nav_node_id
        if entity is not None and entity.raw_sensor_data.position_on_edge is not None:
            fallback_node = entity.raw_sensor_data.position_on_edge
        target_attrs = world_model.road_graph.nodes.get(fallback_node, {})
        tx = target_attrs.get("x", 0)
        ty = target_attrs.get("y", 0)

    # Я использую реальную позицию агента из perception (agent_state.position),
    # а не координаты центра узла графа. Это критически важно: агент посреди
    # длинной дороги (100–200 м) имеет координаты, отличные от центра дороги.
    ax, ay = agent_state.position.x, agent_state.position.y
    eucl_dist = math.hypot(tx - ax, ty - ay)

    if agent_type in (AgentType.FIRE_BRIGADE, AgentType.POLICE_FORCE):
        # Я проверяю евклидово расстояние для пожарных и полицейских,
        # потому что AKExtinguish и AKClear имеют жёсткий максимальный радиус.
        max_dist = FIRE_EXTINGUISH_MAX_DISTANCE if agent_type == AgentType.FIRE_BRIGADE else POLICE_CLEAR_MAX_DISTANCE
        at_target = eucl_dist <= max_dist
    else:
        at_target = (agent_node_id == nav_node_id)

    if at_target:
        # Я выполняю действие по типу агента, передавая оригинальный target_id (не nav_node).
        if agent_type == AgentType.AMBULANCE_TEAM:
            # Я реализую полный цикл спасения: раскопка → погрузка.
            # Пока гражданский ещё завален (buriedness > 0) — продолжаю AKRescue.
            # Когда buriedness == 0 — гражданский свободен, отправляю AKLoad.
            if entity is None:
                # Сущность исчезла из кэша (удалена ядром или вышла из зоны) —
                # отправлять AKLoad без данных опасно: ядро может отклонить команду.
                # Я сбрасываю цель, чтобы на следующем такте выбрать новую.
                logger.warning(
                    "Я не нашёл сущность target_id=%d в кэше, сбрасываю цель (tick=%d)",
                    target_id, tick,
                )
                client.send_rest(tick)
                return False
            buriedness = entity.raw_sensor_data.buriedness
            if buriedness == 0:
                # Я гружу только когда точно уверен, что завалов нет (== 0).
                client.send_load(tick, target_id)
                # Я удаляю загруженного гражданского из кэша задач, чтобы ни этот,
                # ни другие агенты не выбрали его повторно как цель.
                world_model.tasks.pop(target_id, None)
                logger.info("Я отправил AKLoad: target_id=%d, tick=%d", target_id, tick)
                # Я возвращаю False, чтобы main-цикл сбросил current_target_id.
                # После AKLoad агент переходит в режим транспортировки (is_transporting),
                # и на следующем такте ветка транспортировки возьмёт управление.
                return False
            else:
                # Если buriedness > 0 ИЛИ None (сенсоры не обновились) —
                # продолжаю раскопку для страховки. При None ядро проигнорирует
                # AKLoad завалённого гражданского, и агент потеряет такт.
                client.send_rescue(tick, target_id)
                logger.info("Я отправил AKRescue: target_id=%d, buriedness=%s, tick=%d", target_id, buriedness, tick)
        elif agent_type == AgentType.FIRE_BRIGADE:
            # Я задаю полный объём воды — ядро само ограничит его доступным количеством.
            client.send_extinguish(tick, target_id, water=MAX_WATER_DISCHARGE)
            logger.info("Я отправил AKExtinguish: target_id=%d, tick=%d", target_id, tick)
        elif agent_type == AgentType.POLICE_FORCE:
            # Я использую AKClearArea вместо AKClear: AKClearArea расчищает
            # конусообразную область от агента в направлении точки (tx, ty),
            # что эффективнее, чем AKClear по конкретному target_id.
            # ClearSimulator в RCRS обрабатывает AKClearArea: удаляет завалы,
            # пересекающие конус, и уменьшает repair_cost.
            client.send_clear_area(tick, tx, ty)
            logger.info("Я отправил AKClearArea: dest=(%d,%d), tick=%d", tx, ty, tick)
            # Я удаляю завал из кэша после отправки AKClearArea, потому что ядро
            # RCRS не всегда шлёт ChangeSet.deletes при расчистке завала.
            # Без этого полицейский зависает: завал остаётся в кэше с repair_cost > 0,
            # агент снова выбирает его как цель и шлёт AKClearArea в пустоту.
            # После удаления полицейский на следующем такте выберет новый завал
            # или перейдёт в random walk для поиска новых целей.
            world_model.tasks.pop(target_id, None)
            return False
    else:
        # Я движусь по маршруту — ядро переместит агента на максимально возможное расстояние.
        # Я передаю dest_x,dest_y с координатами цели для ВСЕХ типов агентов —
        # это позволяет ядру перемещать агента напрямую к точке по диагонали,
        # а не только до центра узла графа. Без dest_x/dest_y агент идёт
        # «по двум катетам» (через центры промежуточных дорог).
        dest_x: int = tx  # Я использую ранее вычисленные координаты цели
        dest_y: int = ty
        if agent_type in (AgentType.FIRE_BRIGADE, AgentType.POLICE_FORCE):
            # Я уточняю координаты для пожарных/полиции: entity_x/entity_y точнее tx/ty.
            if entity is not None and entity.entity_x is not None and entity.entity_y is not None:
                dest_x, dest_y = entity.entity_x, entity.entity_y
            elif entity is not None and entity.raw_sensor_data.position_on_edge is not None:
                # Я использую fallback для завалов (BLOCKADE): у них entity_x/entity_y
                # часто None, потому что ядро не всегда шлёт PROP_X/PROP_Y.
                # Я беру координаты центра дороги, на которой лежит завал (position_on_edge),
                # чтобы полиция не отправляла AKMove с dest=(-1,-1) и не стояла на месте.
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
        client.send_move(tick, path, dest_x=dest_x, dest_y=dest_y)
        logger.debug(
            "Я отправил AKMove: nav_node=%d, path_len=%d, dest=(%d,%d)",
            nav_node_id, len(path), dest_x, dest_y,
        )
    return True


# Я определяю множество центральных типов агентов для быстрой проверки.
# Центральные агенты не перемещаются и не выполняют действий — они только
# ретранслируют сообщения между каналами связи (в Java: SampleCenter.java).
_CENTER_AGENT_TYPES: frozenset[AgentType] = frozenset({
    AgentType.FIRE_STATION,
    AgentType.AMBULANCE_CENTRE,
    AgentType.POLICE_OFFICE,
})


def _run_center_agent(client: RCRSClient, agent_type: AgentType) -> None:
    """Здесь я запускаю цикл центрального агента (FIRE_STATION / AMBULANCE_CENTRE / POLICE_OFFICE).

    Центральные агенты в RCRS — это стационарные объекты (здания),
    которые служат ретрансляторами сообщений между полевыми агентами.
    В базовой реализации без межагентной коммуникации они просто
    отправляют AKRest каждый такт, чтобы ядро не ожидало их команды бесконечно.
    Без подключения центральных агентов ядро может зависнуть, ожидая
    контроллеры для всех 93 сущностей (90 полевых + 3 центральных).
    """
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

            # Я отправляю AKRest — центральный агент не выполняет действий.
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
    """В этой функции я запускаю основной цикл симуляции и обрабатываю ошибки."""

    # --- Разбор аргументов командной строки (UC-6: запуск нескольких типов агентов) ---
    # Я использую argparse, чтобы один исполняемый файл мог запускать агентов
    # любого типа — запуская несколько процессов с разными флагами --agent-type.
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
    # Я передаю AVERAGE_SPEED в фильтр, чтобы он приводил path_distance (мм) к тактам
    # при проверке дедлайна: t_travel = path_distance / AVERAGE_SPEED.
    dispatcher  = PreFilterDispatcher(work_rate=1.0, average_speed=AVERAGE_SPEED)
    aggregator  = UtilityAggregator(w_c=0.4, w_d=0.2, w_e=0.2, w_n=0.2)
    selector    = TargetSelector(c_switch=0.1)

    current_target_id: int | None = None
    # Я добавляю память для статуса транспортировки (см. комментарии в цикле).
    is_transporting_memory: bool = False

    # Я храню множество посещённых узлов графа между тактами, чтобы random walk
    # приоритетно выбирал непосещённые районы карты и агенты не ходили по кругу.
    visited_nodes: set[int] = set()

    # --- Шаг 0: Подключение к ядру с повторными попытками ---
    # Я жду готовности ядра до MAX_CONNECT_RETRIES попыток с интервалом 3 с.
    # Ядро RCRS загружает GIS, симуляторы и viewer перед открытием порта —
    # на карте Kobe это занимает 15–30 с после запуска start.sh.
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

    # --- Шаг 0.5: Рукопожатие AKConnect → KAConnectOK → AKAcknowledge ---
    # Я провожу рукопожатие один раз после connect() — ядро возвращает agent_id
    # и топологию карты, которую я сохраню в WorldModel на первом такте.
    try:
        agent_id = client.handshake(agent_name, agent_type)
        logger.info("Я завершил рукопожатие: agent_id=%d, agent_type=%s", agent_id, agent_type.value)
    except (ConnectionError, TimeoutError, OSError) as exc:
        logger.error("Я не смог провести рукопожатие: %s", exc)
        client.disconnect()
        return

    # Я проверяю, является ли агент центральным (FIRE_STATION / AMBULANCE_CENTRE / POLICE_OFFICE).
    # Центральные агенты не используют конвейер решений — они просто отправляют AKRest.
    if agent_type in _CENTER_AGENT_TYPES:
        _run_center_agent(client, agent_type)
        return

    try:
        while not _shutdown_requested:
            # --- Шаг 1: Восприятие (Perception) ---
            try:
                packet = client.receive_sense()
            except TimeoutError:
                # Я получил таймаут от сокета — ядро задерживает KASense.
                # Если запрошена остановка — завершаю, иначе продолжаю ожидание.
                if _shutdown_requested:
                    break
                logger.warning("Я не получил KASense в срок, продолжаю ожидание (такт)")
                continue
            except (ConnectionError, ConnectionRefusedError, OSError) as exc:
                logger.error("Я потерял соединение при получении данных: %s", exc)
                break

            # Я начинаю отсчёт времени только после получения пакета восприятия,
            # чтобы не включать время ожидания следующего такта ядра (~1 с).
            tick_start = time.perf_counter()

            # --- Шаг 2: Обновление модели мира (World Update) ---
            # Я применяю весь пакет сразу: граф (если такт 0), союзники, задачи.
            world_model.apply_perception(packet)

            # Я использую собственное состояние из пакета — оно всегда актуально.
            agent_state   = packet.own_state
            agent_node_id = agent_state.position.entity_id

            # Я сохраняю статус транспортировки между тактами, если ядро перестало
            # присылать этот флаг в дельта-обновлениях.
            if agent_state.resources.is_transporting:
                is_transporting_memory = True
            
            # Я принудительно перезаписываю статус в модели агента для текущего такта.
            agent_state.resources.is_transporting = is_transporting_memory

            # Я запоминаю текущую позицию агента, чтобы random walk
            # приоритетно исследовал новые районы карты.
            visited_nodes.add(agent_node_id)

            # Я защищаюсь от ситуации, когда ядро не прислало позицию агента:
            # entity_id=0 означает «не найдено» — в этом случае граф-запросы вернут
            # MAX_MAP_DISTANCE для всех сущностей и агент застынет на месте.
            if agent_node_id == 0:
                logger.warning(
                    "Я не нашёл позиции агента в такте %d (entity_id=0), отправляю AKRest",
                    packet.tick,
                )
                client.send_rest(packet.tick)
                continue

            # --- Шаг 3: Быстрая фильтрация по типу + пересчёт дистанций (UC-6) ---
            # Я сначала отсекаю нерелевантные типы сущностей (O(M) без копирования),
            # а затем вычисляю path_distance только для оставшихся — это снижает
            # число обработанных сущностей на 2/3 и убирает O(M) глубоких копий.
            allowed_type = dispatcher._allowed_entity_type(agent_type)
            type_relevant = [
                e for e in world_model.tasks.values() if e.type == allowed_type
            ]
            fill_path_distances(
                world_model.road_graph,
                agent_node_id,
                type_relevant,
            )

            # Я логирую диагностику задач на уровне INFO, чтобы при первых запусках
            # видеть, что именно содержится в кэше и почему фильтр отсекает задачи.
            if packet.tick % 10 == 0 or len(type_relevant) > 0:
                logger.info(
                    "ДИАГ [%s] такт=%d: кэш=%d задач, type_relevant(%s)=%d",
                    agent_type.value, packet.tick,
                    len(world_model.tasks), allowed_type.value, len(type_relevant),
                )
                if type_relevant and agent_type == AgentType.FIRE_BRIGADE:
                    # Я вывожу fieryness первых 5 зданий для быстрой диагностики.
                    sample = type_relevant[:5]
                    fieryness_info = [(e.id, e.raw_sensor_data.fieryness) for e in sample]
                    logger.info("ДИАГ fieryness: %s", fieryness_info)

            # --- Шаг 4: Предварительная фильтрация (Pre-Filter, UC-2) ---
            # Контракт: type_relevant содержат реальные path_distance (обновлены in-place выше).
            try:
                filtered_tasks = dispatcher.filter_tasks(agent_state, type_relevant)
            except NeedRefugeException:
                logger.info("Я отправляю пожарного в убежище из-за отсутствия воды")
                # Я строю маршрут до ближайшего убежища и сразу отправляю AKMove.
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

            # Я логирую результат фильтрации для диагностики.
            if filtered_tasks:
                logger.info(
                    "ДИАГ [%s] такт=%d: после фильтра=%d задач",
                    agent_type.value, packet.tick, len(filtered_tasks),
                )

            # --- Шаг 5: Расчёт полезности (Utility Calculation, UC-3/UC-4) ---
            utilities: dict[int, float] = {}
            if filtered_tasks:
                # Я передаю расстояние до конкретной задачи в urgency_for_police через
                # параметр task_distance — каждая задача получает свою нормировку.

                for entity in filtered_tasks:
                    # Я вычисляю t_travel и t_work по формулам диплома:
                    # t_travel = d_ij / v_avg [такты]; t_work = Buriedness / Rate [такты].
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

                    # Я определяю позицию цели на карте для евклидовых вычислений (f_social).
                    # Для CIVILIAN entity.id — это ID гражданского, которого нет в графе дорог.
                    # Я использую position_on_edge (здание/дорога, где находится гражданский),
                    # чтобы получить реальные координаты из графа.
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

            # --- Шаг 5.5: Координация — штраф за занятые цели (UC-8) ---
            # Я снижаю полезность целей, о которых уже сообщили соседние агенты
            # через AKSay, чтобы избежать дублирования работы.
            heard = packet.heard_target_ids
            if heard and utilities:
                for tid in heard:
                    if tid in utilities and tid != current_target_id:
                        # Я не штрафую собственную текущую цель — гистерезис важнее.
                        old_u = utilities[tid]
                        utilities[tid] = old_u * CLAIMED_TARGET_PENALTY
                        logger.debug(
                            "Я снизил U для target_id=%d: %.4f → %.4f (занята другим агентом)",
                            tid, old_u, utilities[tid],
                        )

            # --- Шаг 6: Выбор цели с гистерезисом (Selection, UC-5) ---
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

            # --- Шаг 7: Отправка типизированной команды (Action, UC-7) ---
            try:
                if agent_state.resources.is_transporting:
                    # Я определяю узел навигации из координат агента, так как
                    # agent_node_id может указывать на ребро, а алгоритм pathfinding
                    # ожидает узел графа.
                    nav_start_node = world_model.get_nearest_node(
                        agent_state.position.x,
                        agent_state.position.y
                    )
                    
                    refuge_path = nearest_refuge_path(
                        world_model.road_graph,
                        nav_start_node,
                        world_model.refuge_ids,
                    )
                    if refuge_path:
                        # Я проверяю евклидово расстояние до убежища — сравнение
                        # node_id не работает, если агент находится на ребре.
                        refuge_node = refuge_path[-1]
                        ref_attrs = world_model.road_graph.nodes.get(refuge_node, {})
                        ref_x = ref_attrs.get("x", 0)
                        ref_y = ref_attrs.get("y", 0)
                        
                        # Использую порог 5000 мм (5 метров) для определения "в убежище".
                        dist_to_refuge = math.hypot(ref_x - agent_state.position.x, ref_y - agent_state.position.y)
                        
                        if dist_to_refuge < 5000:
                            # Я нахожусь в убежище — выгружаю гражданского.
                            client.send_unload(packet.tick)
                            logger.info(
                                "Я отправил AKUnload в убежище refuge_id=%d, такт=%d",
                                refuge_node, packet.tick,
                            )
                            # Я сбрасываю цель: после выгрузки начинаю поиск новой задачи.
                            current_target_id = None
                            # Я сбрасываю память после успешной выгрузки!
                            is_transporting_memory = False
                        else:
                            # Я передаю координаты убежища как dest_x/dest_y для
                            # плавного перемещения — ядро направит агента к точке,
                            # а не только до центра промежуточного узла графа.
                            client.send_move(packet.tick, refuge_path, dest_x=ref_x, dest_y=ref_y)
                            logger.info(
                                "Я везу гражданского к убежищу refuge_id=%d, такт=%d",
                                refuge_node, packet.tick,
                            )
                    else:
                        client.send_rest(packet.tick)
                        logger.warning("Я не нашёл убежища для выгрузки гражданского, такт=%d", packet.tick)
                elif current_target_id is None:
                    # Я исследую карту случайным маршрутом вместо бездействия.
                    # Без random walk агент стоит на месте и никогда не обнаруживает
                    # задачи за пределами начальной зоны видимости (Line-of-Sight).
                    # Это аналог Java AbstractSampleAgent.randomWalk().
                    rw_path = _random_walk(world_model.road_graph, agent_node_id, visited=visited_nodes)
                    if len(rw_path) > 1:
                        client.send_move(packet.tick, rw_path)
                        logger.info(
                            "Я исследую карту random walk: %d узлов, такт=%d",
                            len(rw_path), packet.tick,
                        )
                    else:
                        # Я стою в тупике без соседей — вынужден ждать.
                        client.send_rest(packet.tick)
                        logger.warning(
                            "Я не могу исследовать карту — тупик node=%d, такт=%d",
                            agent_node_id, packet.tick,
                        )
                else:
                    target_valid = _dispatch_action(
                        client=client,
                        agent_type=agent_type,
                        agent_state=agent_state,
                        tick=packet.tick,
                        target_id=current_target_id,
                        agent_node_id=agent_node_id,
                        world_model=world_model,
                    )
                    # Я сбрасываю цель, если _dispatch_action обнаружил, что к ней
                    # невозможно построить маршрут — это предотвращает зависание агента
                    # на AKRest, когда гражданский не имеет валидного position_on_edge.
                    if not target_valid:
                        # Я удаляю недостижимую цель из кэша, чтобы TargetSelector
                        # не выбрал её повторно на следующем такте — иначе агент
                        # зависнет на AKRest до конца симуляции.
                        if current_target_id is not None:
                            world_model.tasks.pop(current_target_id, None)
                        current_target_id = None
            except (ConnectionError, ConnectionRefusedError, TimeoutError, OSError) as exc:
                logger.error("Я потерял соединение при отправке команды: %s", exc)
                break

            # --- Шаг 8: Координация — отправка AKSay с текущей целью (UC-8) ---
            # Я сообщаю соседним агентам свою текущую цель, чтобы они учитывали
            # её при расчёте полезности и не дублировали работу.
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
