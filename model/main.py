from __future__ import annotations

"""В этом модуле я запускаю главный цикл агента и конвейер принятия решений."""

import logging
import sys
import time
from pathlib import Path


SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from action.selection import TargetSelector  # noqa: E402
from decision.filters.pre_filter import NeedRefugeException, PreFilterDispatcher  # noqa: E402
from decision.utility.aggregator import UtilityAggregator  # noqa: E402
from network.client import RCRSClient  # noqa: E402
from world.cache import WorldModel  # noqa: E402
from world.entities import AgentState, AgentType, Position, Resources  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

AVERAGE_SPEED = 1.0


def main() -> None:
    """В этой функции я запускаю основной цикл симуляции и обрабатываю ошибки."""

    client = RCRSClient(host="127.0.0.1", port=7000, timeout=5.0)
    world_model = WorldModel()
    dispatcher = PreFilterDispatcher(work_rate=1.0)
    aggregator = UtilityAggregator(w_c=0.4, w_d=0.2, w_e=0.2, w_n=0.2)
    selector = TargetSelector(c_switch=0.1)

    agent_state = AgentState(
        id=1,
        type=AgentType.FIRE_BRIGADE,
        position=Position(entity_id=1, x=0, y=0),
        resources=Resources(water_quantity=5000, is_transporting=False),
    )

    current_target_id = None

    try:
        client.connect()
    except (ConnectionRefusedError, TimeoutError, OSError) as exc:
        logger.error("Я не смог установить соединение и завершаю работу: %s", exc)
        return

    try:
        tick = 0
        while True:
            tick_start = time.perf_counter()

            try:
                visible_entities = client.receive_sense()
            except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                logger.error("Я потерял соединение при получении данных: %s", exc)
                break

            world_model.update_perception(visible_entities)

            try:
                filtered_tasks = dispatcher.filter_tasks(agent_state, visible_entities)
            except NeedRefugeException:
                logger.info("Я отправляю пожарного в убежище из-за отсутствия воды")
                filtered_tasks = []

            utilities = {}
            if filtered_tasks:
                for entity in filtered_tasks:
                    t_travel = entity.computed_metrics.path_distance / AVERAGE_SPEED
                    buriedness = entity.raw_sensor_data.buriedness
                    t_work = 0.0 if buriedness is None else buriedness / dispatcher.work_rate
                    utility = aggregator.calculate_utility(
                        agent_state=agent_state,
                        entity=entity,
                        world_model=world_model,
                        target_position=Position(entity_id=entity.id, x=0, y=0),
                        t_travel=t_travel,
                        t_work=t_work,
                        social_radius=2000.0,
                    )
                    utilities[entity.id] = utility
            else:
                logger.info("Я не нашел релевантных задач после фильтрации")

            selected_target_id = selector.select_best_target(current_target_id, utilities)

            if selected_target_id is None:
                logger.info("Я перевожу агента в режим ожидания")
            elif selected_target_id != current_target_id:
                logger.info("Я выбрал новую цель: target_id=%s", selected_target_id)
            else:
                logger.info("Я сохраняю текущую цель: target_id=%s", current_target_id)

            current_target_id = selected_target_id

            try:
                command_payload = "IDLE" if current_target_id is None else f"TARGET {current_target_id}"
                client.send_command(payload=f"TICK {tick} {command_payload}")
            except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                logger.error("Я потерял соединение при отправке команды: %s", exc)
                break

            tick_elapsed = time.perf_counter() - tick_start
            if tick_elapsed > 0.1:
                logger.warning("Я превысил бюджет времени тика: %.3f c", tick_elapsed)

            tick += 1
    except KeyboardInterrupt:
        logger.info("Я получил сигнал остановки и завершаю работу")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
