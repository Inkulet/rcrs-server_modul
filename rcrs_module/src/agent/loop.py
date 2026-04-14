from __future__ import annotations

import logging
import random
import struct
import time

from config import (
    AVERAGE_SPEED,
    CLAIMED_TARGET_PENALTY,
    EXPLORATION_MAX_TICKS,
    EXPLORATION_PATH_LENGTH,
    EXPLORATION_SEED_PRIME,
    LOG_DIAG_PERIOD,
    NO_REFUGE_MAX_RETRIES,
    SOCIAL_RADIUS,
    TICK_BUDGET_SECONDS,
)
from action.executor import dispatch_action
from action.navigation import (
    fill_path_distances,
    nearest_refuge_path,
    pick_exploration_target,
    plan_exploration_path,
    random_walk,
)
from action.selection import TargetSelector
from decision.filters.pre_filter import NeedRefugeException, PreFilterDispatcher
from decision.utility.aggregator import UtilityAggregator
from network.client import RCRSClient
from world.cache import WorldModel
from world.entities import AgentType, Position


logger = logging.getLogger(__name__)


def run_field_agent(
    client: RCRSClient,
    world_model: WorldModel,
    agent_type: AgentType,
    dispatcher: PreFilterDispatcher,
    aggregator: UtilityAggregator,
    selector: TargetSelector,
) -> None:
    current_target_id: int | None = None
    visited_nodes: set[int] = set()

    no_refuge_counter: int = 0

    exploration_target_node: int | None = None
    exploration_start_tick: int = 0

    try:
        while True:
            try:
                packet = client.receive_sense()
            except TimeoutError:
                logger.warning("Я не получил KASense в срок, продолжаю ожидание")
                continue
            except (ConnectionError, OSError) as exc:
                logger.error("Я потерял соединение при получении данных: %s", exc)
                break

            tick_start = time.perf_counter()

            world_model.apply_perception(packet)

            agent_state   = packet.own_state
            agent_node_id = agent_state.position.entity_id
            visited_nodes.add(agent_node_id)


            if agent_node_id == 0:
                logger.warning(
                    "Я не нашёл позиции агента (entity_id=0), такт=%d → AKRest", packet.tick,
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
                    agent_state.buriedness, packet.tick,
                )
                client.send_rest(packet.tick)
                try:
                    say_data = struct.pack(">i", agent_state.id)
                    client.send_say(packet.tick, say_data)
                except (ConnectionError, OSError, struct.error) as exc:
                    logger.warning("Я не смог крикнуть о помощи: %s", exc)
                current_target_id = None
                continue

            if agent_state.resources.is_transporting:
                no_refuge_counter = _handle_transport(
                    client, packet, agent_node_id, world_model, no_refuge_counter,
                )
                current_target_id = None
                continue

            allowed_types = dispatcher._allowed_entity_types(agent_type)
            type_relevant = [
                e for e in world_model.tasks.values()
                if e.type in allowed_types
                and e.raw_sensor_data.position_on_edge not in world_model.refuge_ids
            ]
            fill_path_distances(world_model.road_graph, agent_node_id, type_relevant)

            if packet.tick % LOG_DIAG_PERIOD == 0 or type_relevant:
                logger.info(
                    "ДИАГ [%s] такт=%d: кэш=%d задач, type_relevant=%d",
                    agent_type.value, packet.tick,
                    len(world_model.tasks), len(type_relevant),
                )

            try:
                filtered_tasks = dispatcher.filter_tasks(agent_state, type_relevant)
            except NeedRefugeException:
                logger.info("Я отправляю пожарного в убежище: нет воды")
                refuge_path = nearest_refuge_path(
                    world_model.road_graph, agent_node_id, world_model.refuge_ids,
                )
                if refuge_path:
                    client.send_move(packet.tick, refuge_path)
                else:
                    client.send_rest(packet.tick)
                continue

            utilities: dict[int, float] = {}
            for entity in filtered_tasks:
                try:
                    t_travel = entity.computed_metrics.path_distance / AVERAGE_SPEED
                except ZeroDivisionError:
                    t_travel = 0.0

                buriedness = entity.raw_sensor_data.buriedness
                try:
                    t_work = 0.0 if buriedness is None else buriedness / dispatcher.work_rate
                except ZeroDivisionError:
                    t_work = 0.0

                nav_id = (
                    entity.raw_sensor_data.position_on_edge
                    if entity.raw_sensor_data.position_on_edge is not None
                    else entity.id
                )
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
                    agent_state.id, utility, entity.id,
                )

            for tid in packet.heard_target_ids:
                if tid in utilities and tid != current_target_id:
                    old_u = utilities[tid]
                    utilities[tid] = old_u * CLAIMED_TARGET_PENALTY
                    logger.debug(
                        "Я снизил U для target_id=%d: %.4f → %.4f (занята другим)",
                        tid, old_u, utilities[tid],
                    )

            is_stuck = selector.report_progress(current_target_id, agent_node_id, packet.tick)
            if is_stuck:
                current_target_id = None

            current_target_id = selector.select_best_target(
                current_target_id, utilities, current_tick=packet.tick,
            )

            if current_target_id is None:
                logger.info("Я не нашёл задачи — перехожу в режим исследования, такт=%d", packet.tick)
            elif not is_stuck:
                logger.info("Я выбрал/сохраняю цель target_id=%s, такт=%d", current_target_id, packet.tick)

            try:
                if current_target_id is None:
                    exploration_target_node, exploration_start_tick = _explore_and_update(
                        client, packet, agent_state, agent_node_id,
                        world_model, visited_nodes,
                        exploration_target_node, exploration_start_tick,
                    )
                else:
                    target_valid, unreachable, working = dispatch_action(
                        client=client,
                        agent_type=agent_type,
                        agent_state=agent_state,
                        tick=packet.tick,
                        target_id=current_target_id,
                        agent_node_id=agent_node_id,
                        world_model=world_model,
                    )
                    if unreachable:
                        selector.blacklist_unreachable(current_target_id, packet.tick)
                        current_target_id = None
                        selector.reset_stuck()
                    elif not target_valid:
                        world_model.remove_task(current_target_id)
                        current_target_id = None
                        selector.reset_stuck()
                    elif working:
                        selector.reset_stuck()

            except (ConnectionError, OSError) as exc:
                logger.error("Я потерял соединение при отправке команды: %s", exc)
                break

            if current_target_id is not None:
                try:
                    say_data = struct.pack(">i", current_target_id)
                    client.send_say(packet.tick, say_data)
                except (ConnectionError, OSError) as exc:
                    logger.warning("Я не смог отправить AKSay: %s", exc)

            tick_elapsed = time.perf_counter() - tick_start
            if tick_elapsed > TICK_BUDGET_SECONDS:
                logger.warning(
                    "Я превысил бюджет такта %d: %.3f с", packet.tick, tick_elapsed,
                )

    except KeyboardInterrupt:
        logger.info("Я получил сигнал остановки и завершаю работу")
    finally:
        client.disconnect()


def _handle_transport(
    client: RCRSClient,
    packet: object,
    agent_node_id: int,
    world_model: WorldModel,
    no_refuge_counter: int,
) -> int:
    from world.entities import PerceptionPacket
    pkt: PerceptionPacket = packet  # type: ignore[assignment]

    refuge_path = nearest_refuge_path(
        world_model.road_graph, agent_node_id, world_model.refuge_ids,
    )
    if refuge_path:
        refuge_node = refuge_path[-1]
        ref_attrs = world_model.road_graph.nodes.get(refuge_node, {})
        if agent_node_id == refuge_node:
            client.send_unload(pkt.tick)
            logger.info("Я выгрузил гражданского в убежище refuge_id=%d, такт=%d", refuge_node, pkt.tick)
        else:
            ref_x = ref_attrs.get("x", 0)
            ref_y = ref_attrs.get("y", 0)
            client.send_move(pkt.tick, refuge_path, dest_x=ref_x, dest_y=ref_y)
            logger.info("Я везу гражданского к убежищу refuge_id=%d, такт=%d", refuge_node, pkt.tick)
        return 0
    else:
        no_refuge_counter += 1
        if no_refuge_counter >= NO_REFUGE_MAX_RETRIES:
            logger.error(
                "Я %d тактов не могу найти убежище, такт=%d",
                no_refuge_counter, pkt.tick,
            )
            return 0
        logger.warning("Я не нашёл убежища (попытка %d), такт=%d", no_refuge_counter, pkt.tick)
        client.send_rest(pkt.tick)
        return no_refuge_counter


def _explore_and_update(
    client: RCRSClient,
    packet: object,
    agent_state: object,
    agent_node_id: int,
    world_model: WorldModel,
    visited_nodes: set[int],
    exploration_target_node: int | None,
    exploration_start_tick: int,
) -> tuple[int | None, int]:
    from world.entities import AgentState, PerceptionPacket
    pkt: PerceptionPacket = packet  # type: ignore[assignment]
    state: AgentState = agent_state  # type: ignore[assignment]

    need_new = (
        exploration_target_node is None
        or exploration_target_node == agent_node_id
        or not world_model.road_graph.has_node(exploration_target_node)
        or (pkt.tick - exploration_start_tick) >= EXPLORATION_MAX_TICKS
    )

    if need_new:
        rng = random.Random(state.id * EXPLORATION_SEED_PRIME + pkt.tick)
        exploration_target_node = pick_exploration_target(
            world_model.road_graph, agent_node_id, visited=visited_nodes, rng=rng,
        )
        exploration_start_tick = pkt.tick
        if exploration_target_node is not None:
            logger.info(
                "Я выбрал новую цель исследования node=%d (из узла=%d), такт=%d",
                exploration_target_node, agent_node_id, pkt.tick,
            )

    exp_path: list[int] = []
    if exploration_target_node is not None:
        exp_path = plan_exploration_path(
            world_model.road_graph, agent_node_id,
            exploration_target_node, max_steps=EXPLORATION_PATH_LENGTH,
        )

    if exp_path and len(exp_path) > 1:
        client.send_move(pkt.tick, exp_path)
        logger.info(
            "Я иду к цели исследования node=%d: %d шагов, такт=%d",
            exploration_target_node, len(exp_path), pkt.tick,
        )
    else:
        exploration_target_node = None
        rw_path = random_walk(world_model.road_graph, agent_node_id, visited=visited_nodes)
        if len(rw_path) > 1:
            client.send_move(pkt.tick, rw_path)
            logger.info("Я исследую random walk: %d узлов, такт=%d", len(rw_path), pkt.tick)
        else:
            client.send_rest(pkt.tick)
            logger.warning("Я не могу исследовать — тупик node=%d, такт=%d", agent_node_id, pkt.tick)

    return exploration_target_node, exploration_start_tick


__all__ = ["run_field_agent"]
