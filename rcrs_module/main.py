from __future__ import annotations

"""В этом модуле я запускаю главный цикл агента и конвейер принятия решений.

Я реализую строгий конвейер: Perception → World → Navigation → Filter → Utility → Action.
Каждый такт агент получает свежий пакет восприятия, обновляет модель мира,
пересчитывает реальные дистанции по графу и принимает решение строго
на основе функции полезности U_ij.
"""

import argparse
import logging
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
from world.entities import AgentType, Position  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Параметры подключения ---
KERNEL_HOST: str  = "127.0.0.1"
KERNEL_PORT: int  = 7000
KERNEL_TIMEOUT: float = 5.0

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

# --- Глобальный флаг остановки (используется обработчиком SIGTERM) ---
_shutdown_requested: bool = False


def _sigterm_handler(signum: int, frame: object) -> None:
    """Я устанавливаю флаг остановки по сигналу SIGTERM, чтобы выйти чисто."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Я получил SIGTERM (signal=%d), завершаю работу после текущего такта", signum)


signal.signal(signal.SIGTERM, _sigterm_handler)


def _get_nav_node(target_id: int, world_model: WorldModel) -> int:
    """Здесь я определяю узел графа для навигации к цели.

    Для гражданских (CIVILIAN) entity_id — это сам объект гражданского, которого
    нет в дорожном графе. Я навигирую к зданию/дороге, на которой находится
    гражданский (PROP_POSITION → position_on_edge). Для зданий и завалов
    entity_id совпадает с узлом графа напрямую.
    """
    entity = world_model.tasks.get(target_id)
    if entity is not None and entity.raw_sensor_data.position_on_edge is not None:
        pos = entity.raw_sensor_data.position_on_edge
        # Я проверяю, что узел существует в графе — position_on_edge может быть
        # устаревшим или указывать на объект вне зоны карты.
        if world_model.road_graph.has_node(pos):
            return pos
    return target_id


def _dispatch_action(
    client: RCRSClient,
    agent_type: AgentType,
    tick: int,
    target_id: int,
    agent_node_id: int,
    world_model: WorldModel,
) -> None:
    """Здесь я отправляю типизированную команду в зависимости от типа агента.

    Логика:
    - Я разделяю nav_node_id (узел графа для навигации) и target_id (ID объекта действия).
      Для гражданского: nav_node_id = здание, где он находится; target_id = ID гражданского.
      Для здания/завала: nav_node_id = target_id (здание само является узлом графа).
    - Агент «у цели», когда его позиция совпадает с nav_node_id.
    - При достижении: выполняю целевое действие по типу агента (UC-7).
    - Иначе: отправляю AKMove по кратчайшему пути к nav_node_id.
    """
    # Я определяю узел навигации — для гражданских это здание, где они находятся.
    nav_node_id = _get_nav_node(target_id, world_model)

    path = compute_path(world_model.road_graph, agent_node_id, nav_node_id)

    if not path:
        logger.warning(
            "Я не могу построить путь к target_id=%d (nav_node=%d), отправляю AKRest",
            target_id, nav_node_id,
        )
        client.send_rest(tick)
        return

    # Я считаю агента «у цели», если его текущий узел совпадает с навигационным узлом.
    # Это единственный корректный критерий: len(path)==1 эквивалентен from_id==to_id.
    at_target = (agent_node_id == nav_node_id)

    if at_target:
        # Я выполняю действие по типу агента, передавая оригинальный target_id (не nav_node).
        if agent_type == AgentType.AMBULANCE_TEAM:
            # Я реализую полный цикл спасения: раскопка → погрузка.
            # Пока гражданский ещё завален (buriedness > 0) — продолжаю AKRescue.
            # Когда buriedness == 0 — гражданский свободен, отправляю AKLoad.
            entity = world_model.tasks.get(target_id)
            buriedness = entity.raw_sensor_data.buriedness if entity is not None else None
            if buriedness is not None and buriedness > 0:
                client.send_rescue(tick, target_id)
                logger.info("Я отправил AKRescue: target_id=%d, buriedness=%d, tick=%d", target_id, buriedness, tick)
            else:
                # Гражданский раскопан (buriedness=0) или данные устарели — гружу.
                client.send_load(tick, target_id)
                logger.info("Я отправил AKLoad: target_id=%d, tick=%d", target_id, tick)
        elif agent_type == AgentType.FIRE_BRIGADE:
            # Я задаю полный объём воды — ядро само ограничит его доступным количеством.
            client.send_extinguish(tick, target_id, water=10_000)
            logger.info("Я отправил AKExtinguish: target_id=%d, tick=%d", target_id, tick)
        elif agent_type == AgentType.POLICE_FORCE:
            client.send_clear(tick, target_id)
            logger.info("Я отправил AKClear: target_id=%d, tick=%d", target_id, tick)
    else:
        # Я движусь по маршруту — ядро переместит агента на максимально возможное расстояние.
        client.send_move(tick, path)
        logger.debug(
            "Я отправил AKMove: nav_node=%d, path_len=%d, first=%d, last=%d",
            nav_node_id, len(path), path[0], path[-1],
        )


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
        help="Тип агента RCRS (FIRE_BRIGADE / AMBULANCE_TEAM / POLICE_FORCE)",
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

    # --- Шаг 0: Подключение к ядру ---
    try:
        client.connect()
    except (ConnectionRefusedError, TimeoutError, OSError) as exc:
        logger.error("Я не смог установить соединение и завершаю работу: %s", exc)
        return

    # --- Шаг 0.5: Рукопожатие AKConnect → KAConnectOK → AKAcknowledge ---
    # Я провожу рукопожатие один раз после connect() — ядро возвращает agent_id
    # и топологию карты, которую я сохраню в WorldModel на первом такте.
    try:
        agent_id = client.handshake(agent_name, agent_type)
        logger.info("Я завершил рукопожатие: agent_id=%d, agent_type=%s", agent_id, agent_type.value)
    except (ConnectionError, OSError) as exc:
        logger.error("Я не смог провести рукопожатие: %s", exc)
        client.disconnect()
        return

    try:
        while not _shutdown_requested:
            tick_start = time.perf_counter()

            # --- Шаг 1: Восприятие (Perception) ---
            try:
                packet = client.receive_sense()
            except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                logger.error("Я потерял соединение при получении данных: %s", exc)
                break

            # --- Шаг 2: Обновление модели мира (World Update) ---
            # Я применяю весь пакет сразу: граф (если такт 0), союзники, задачи.
            world_model.apply_perception(packet)

            # Я использую собственное состояние из пакета — оно всегда актуально.
            agent_state   = packet.own_state
            agent_node_id = agent_state.position.entity_id

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

            # --- Шаг 3: Пересчёт реальных дистанций по графу (Navigation, UC-6) ---
            # Я обновляю path_distance у всех известных сущностей (исторический кэш +
            # свежие данные такта), а не только у видимых в текущем такте.
            # Это позволяет агенту продолжать работу с целями, вышедшими из зоны видимости.
            enriched_entities = fill_path_distances(
                world_model.road_graph,
                agent_node_id,
                list(world_model.tasks.values()),
            )

            # --- Шаг 4: Предварительная фильтрация (Pre-Filter, UC-2) ---
            # Контракт: enriched_entities содержат реальные path_distance (уже заполнены выше).
            try:
                filtered_tasks = dispatcher.filter_tasks(agent_state, enriched_entities)
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

            # --- Шаг 5: Расчёт полезности (Utility Calculation, UC-3/UC-4) ---
            utilities: dict[int, float] = {}
            if filtered_tasks:
                # Я вычисляю min_distance_to_targets для полиции: минимальная дистанция
                # до любого завала в filtered_tasks. urgency_for_police использует это
                # значение для нормировки — без него f_urgency для полиции всегда 0.0.
                min_distance_to_targets: float | None = None
                if agent_state.type.value == "POLICE_FORCE":
                    distances = [e.computed_metrics.path_distance for e in filtered_tasks]
                    min_distance_to_targets = min(distances) if distances else None

                for entity in filtered_tasks:
                    # Я вычисляю t_travel и t_work по формулам диплома:
                    # t_travel = d_ij / v_avg [такты]; t_work = Buriedness / Rate [такты].
                    t_travel = entity.computed_metrics.path_distance / AVERAGE_SPEED
                    buriedness = entity.raw_sensor_data.buriedness
                    t_work = 0.0 if buriedness is None else buriedness / dispatcher.work_rate

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
                        min_distance_to_targets=min_distance_to_targets,
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
                    # Я везу гражданского — нужно добраться до ближайшего убежища и выгрузить.
                    # pre_filter уже вернул [] при is_transporting=True, поэтому current_target
                    # мог стать None — обрабатываю транспортировку здесь явно.
                    refuge_path = nearest_refuge_path(
                        world_model.road_graph,
                        agent_node_id,
                        world_model.refuge_ids,
                    )
                    if refuge_path:
                        if agent_node_id == refuge_path[-1]:
                            # Я нахожусь в убежище — выгружаю гражданского.
                            client.send_unload(packet.tick)
                            logger.info(
                                "Я отправил AKUnload в убежище refuge_id=%d, такт=%d",
                                refuge_path[-1], packet.tick,
                            )
                            # Я сбрасываю цель: после выгрузки начинаю поиск новой задачи.
                            current_target_id = None
                        else:
                            client.send_move(packet.tick, refuge_path)
                            logger.info(
                                "Я везу гражданского к убежищу refuge_id=%d, такт=%d",
                                refuge_path[-1], packet.tick,
                            )
                    else:
                        client.send_rest(packet.tick)
                        logger.warning("Я не нашёл убежища для выгрузки гражданского, такт=%d", packet.tick)
                elif current_target_id is None:
                    client.send_rest(packet.tick)
                else:
                    _dispatch_action(
                        client=client,
                        agent_type=agent_type,
                        tick=packet.tick,
                        target_id=current_target_id,
                        agent_node_id=agent_node_id,
                        world_model=world_model,
                    )
            except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                logger.error("Я потерял соединение при отправке команды: %s", exc)
                break

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
