from __future__ import annotations

import logging
import math
import random
import struct
from pathlib import Path

from config import (
    ALLY_TARGET_LONGTERM_PENALTY,
    AVERAGE_SPEED,
    CLAIMED_TARGET_PENALTY,
    EXPLORATION_MAX_TICKS,
    EXPLORATION_PATH_LENGTH,
    EXPLORATION_SEED_PRIME,
    LOG_DIAG_PERIOD,
    METRICS_BUDGET_MS,
    METRICS_ENABLED,
    METRICS_FILE,
    METRICS_REPORT_PERIOD,
    NO_REFUGE_MAX_RETRIES,
    POLICE_URGENCY_DISTANCE_SCALE,
    RECENT_ALLY_TARGET_TICKS,
    SOCIAL_RADIUS,
    TICK_BUDGET_SECONDS,
)
from metrics import MetricsCollector
from action.executor import dispatch_action, try_clear_local_blockade
from action.navigation import (
    choose_refuge_with_exit,
    fill_path_distances,
    nearest_refuge_path,
    pick_search_target,
    pick_exploration_target,
    plan_exploration_path,
    random_walk,
)
from action.selection import TargetSelector
from decision.filters.pre_filter import NeedRefugeException, PreFilterDispatcher
from decision.utility.aggregator import UtilityAggregator
from network.client import RCRSClient
from world.cache import WorldModel
from world.entities import AgentType, EntityType, Position, VisibleEntity


logger = logging.getLogger(__name__)

METRICS_REPORT_PATH: Path = Path(__file__).resolve().parents[2] / METRICS_FILE


def _entity_world_position(entity: VisibleEntity, world_model: WorldModel) -> tuple[int, int] | None:
    if entity.entity_x is not None and entity.entity_y is not None:
        if entity.entity_x != 0 or entity.entity_y != 0:
            return int(entity.entity_x), int(entity.entity_y)

    pos_edge = entity.raw_sensor_data.position_on_edge
    if pos_edge is not None:
        attrs = world_model.road_graph.nodes.get(pos_edge, {})
        x = attrs.get("x")
        y = attrs.get("y")
        if x is not None and y is not None and (x != 0 or y != 0):
            return int(x), int(y)

    return None


def _min_distance_blockade_to_important(
    blockade: VisibleEntity,
    world_model: WorldModel,
) -> float:
    bpos = _entity_world_position(blockade, world_model)
    if bpos is None:
        return float("inf")
    bx, by = bpos

    min_d: float = float("inf")

    for task in world_model.tasks.values():
        if task.type == EntityType.BLOCKADE:
            continue

        if task.type in (EntityType.CIVILIAN, EntityType.HUMAN):
            hp = task.raw_sensor_data.hp
            if hp is not None and hp == 0:
                continue

        if task.type == EntityType.BUILDING:
            fieryness = task.raw_sensor_data.fieryness
            if fieryness is None or fieryness not in {1, 2, 3}:
                continue

        tpos = _entity_world_position(task, world_model)
        if tpos is None:
            continue
        d = math.hypot(bx - tpos[0], by - tpos[1])
        if d < min_d:
            min_d = d

    for refuge_id in world_model.refuge_ids:
        attrs = world_model.road_graph.nodes.get(refuge_id, {})
        rx = attrs.get("x")
        ry = attrs.get("y")
        if rx is None or ry is None or (rx == 0 and ry == 0):
            continue
        d = math.hypot(bx - int(rx), by - int(ry))
        if d < min_d:
            min_d = d

    for ally in world_model.agents.values():
        if ally.type not in (AgentType.FIRE_BRIGADE, AgentType.AMBULANCE_TEAM):
            continue
        ax, ay = ally.position.x, ally.position.y
        if ax == 0 and ay == 0:
            continue
        d = math.hypot(bx - ax, by - ay)
        if d < min_d:
            min_d = d

    return min_d


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

    last_processed_tick: int = -1

    metrics = MetricsCollector(
        enabled=METRICS_ENABLED, budget_ms=METRICS_BUDGET_MS,
    )
    client.set_metrics(metrics)

    recent_allies_targets: dict[int, int] = {}

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

            metrics.start_tick()

            if last_processed_tick >= 0 and packet.tick - last_processed_tick > 1:
                logger.warning(
                    "Я пропустил %d такт(ов): прошлый=%d, текущий=%d "
                    "(ядро не дождалось моей команды)",
                    packet.tick - last_processed_tick - 1,
                    last_processed_tick, packet.tick,
                )
            last_processed_tick = packet.tick

            with metrics.phase("perception"):
                world_model.apply_perception(packet)

            agent_state   = packet.own_state
            agent_node_id = agent_state.position.entity_id
            visited_nodes.add(agent_node_id)

            metrics.gauge("cache_size", len(world_model.tasks))
            metrics.gauge("visible_entities", len(packet.visible_entities))
            metrics.gauge("ally_states", len(packet.ally_states))
            metrics.gauge("blockades_on_map", sum(
                1 for e in world_model.tasks.values()
                if e.type == EntityType.BLOCKADE
            ))


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

            with metrics.phase("dijkstra"):
                allowed_types = dispatcher._allowed_entity_types(agent_type)
                type_relevant = [
                    e for e in world_model.tasks.values()
                    if e.type in allowed_types
                    and e.raw_sensor_data.position_on_edge not in world_model.refuge_ids
                ]
                fill_path_distances(
                    world_model.road_graph, agent_node_id, type_relevant,
                    blockades_by_node=world_model.blockades_by_node,
                )

            metrics.gauge("type_relevant", len(type_relevant))

            if packet.tick % LOG_DIAG_PERIOD == 0 or type_relevant:
                logger.info(
                    "ДИАГ [%s] такт=%d: кэш=%d задач, type_relevant=%d",
                    agent_type.value, packet.tick,
                    len(world_model.tasks), len(type_relevant),
                )

            try:
                with metrics.phase("pre_filter"):
                    filtered_tasks = dispatcher.filter_tasks(
                        agent_state, type_relevant, world_model=world_model,
                    )
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

            metrics.gauge("filtered_tasks", len(filtered_tasks))

            if agent_type == AgentType.FIRE_BRIGADE:
                rescue_tasks = [
                    entity for entity in filtered_tasks
                    if entity.type in (EntityType.CIVILIAN, EntityType.HUMAN)
                ]
                if rescue_tasks:
                    filtered_tasks = rescue_tasks

            with metrics.phase("utility"):
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

                if agent_type == AgentType.POLICE_FORCE:
                    raw_min_d = _min_distance_blockade_to_important(
                        entity, world_model,
                    )
                    if raw_min_d == float("inf"):
                        task_distance = 1.0e9
                    else:
                        task_distance = raw_min_d / POLICE_URGENCY_DISTANCE_SCALE
                else:
                    task_distance = entity.computed_metrics.path_distance

                utility = aggregator.calculate_utility(
                    agent_state=agent_state,
                    entity=entity,
                    world_model=world_model,
                    target_position=target_position,
                    t_travel=t_travel,
                    t_work=t_work,
                    task_distance=task_distance,
                    social_radius=SOCIAL_RADIUS,
                    max_map_distance=world_model.max_map_distance,
                )
                utilities[entity.id] = utility
                logger.debug(
                    "Я рассчитал U_%d=%.4f для entity_id=%d",
                    agent_state.id, utility, entity.id,
                )

            for tid in packet.heard_target_ids:
                recent_allies_targets[tid] = packet.tick
                if tid in utilities and tid != current_target_id:
                    old_u = utilities[tid]
                    utilities[tid] = old_u - CLAIMED_TARGET_PENALTY
                    logger.debug(
                        "Я снизил U для target_id=%d: %.4f → %.4f (занята другим)",
                        tid, old_u, utilities[tid],
                    )

            if agent_type == AgentType.FIRE_BRIGADE:
                expired = [
                    t for t, last in recent_allies_targets.items()
                    if packet.tick - last > RECENT_ALLY_TARGET_TICKS
                ]
                for t in expired:
                    recent_allies_targets.pop(t, None)
                for tid in list(recent_allies_targets.keys()):
                    if tid in utilities and tid != current_target_id:
                        utilities[tid] -= ALLY_TARGET_LONGTERM_PENALTY

            prev_target = current_target_id
            with metrics.phase("selection"):
                is_stuck = selector.report_progress(
                    current_target_id, agent_node_id, packet.tick,
                )
                if is_stuck:
                    current_target_id = None
                    metrics.inc("stuck_detected")

                current_target_id = selector.select_best_target(
                    current_target_id, utilities, current_tick=packet.tick,
                )
            if current_target_id is not None and current_target_id != prev_target:
                metrics.inc("target_selected")
                if prev_target is not None and prev_target != current_target_id:
                    metrics.inc("target_switched")

            if current_target_id is None:
                logger.info("Я не нашёл задачи — перехожу в режим исследования, такт=%d", packet.tick)
            elif not is_stuck:
                logger.info("Я выбрал/сохраняю цель target_id=%s, такт=%d", current_target_id, packet.tick)

            try:
                action_sent = False

                if current_target_id is not None:
                    with metrics.phase("dispatch"):
                        target_valid, unreachable, working, _attempted = dispatch_action(
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
                        metrics.inc("target_blacklisted")
                        metrics.inc("unreachable_target")
                        current_target_id = None
                        selector.reset_stuck()
                    elif not target_valid:
                        world_model.remove_task(current_target_id)
                        current_target_id = None
                        selector.reset_stuck()
                    else:
                        action_sent = True
                        if working:
                            selector.reset_stuck()

                if not action_sent:
                    if agent_type == AgentType.POLICE_FORCE:
                        with metrics.phase("dispatch"):
                            sent, _ = try_clear_local_blockade(
                                client, packet.tick, agent_node_id, world_model,
                                agent_state=agent_state,
                            )
                        if sent:
                            action_sent = True

                if not action_sent:
                    with metrics.phase("explore"):
                        exploration_target_node, exploration_start_tick = _explore_and_update(
                            client, packet, agent_state, agent_node_id,
                            world_model, visited_nodes,
                            exploration_target_node, exploration_start_tick,
                        )

            except (ConnectionError, OSError) as exc:
                logger.error("Я потерял соединение при отправке команды: %s", exc)
                break

            if current_target_id is not None:
                with metrics.phase("say"):
                    try:
                        say_data = struct.pack(">i", current_target_id)
                        client.send_say(packet.tick, say_data)
                    except (ConnectionError, OSError) as exc:
                        logger.warning("Я не смог отправить AKSay: %s", exc)

            metrics.stop_tick(packet.tick)

            tick_ms = metrics.last_tick_ms()
            if tick_ms > TICK_BUDGET_SECONDS * 1000.0:
                logger.warning(
                    "Я превысил бюджет такта %d: %.1f мс (tasks=%d)",
                    packet.tick, tick_ms, len(type_relevant),
                )

            if packet.tick % METRICS_REPORT_PERIOD == 0:
                metrics.write_report(
                    METRICS_REPORT_PATH, agent_state.id, agent_type.value,
                )

    except KeyboardInterrupt:
        logger.info("Я получил сигнал остановки и завершаю работу")
    finally:
        try:
            metrics.write_report(
                METRICS_REPORT_PATH,
                agent_state.id if "agent_state" in locals() else -1,
                agent_type.value,
            )
        except (OSError, NameError) as exc:
            logger.warning("Я не смог записать финальный отчёт метрик: %s", exc)
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

    my_id = pkt.own_state.id
    for ent in world_model.tasks.values():
        if ent.type != EntityType.CIVILIAN:
            continue
        if ent.raw_sensor_data.position_on_edge != my_id:
            continue
        hp = ent.raw_sensor_data.hp
        if hp is not None and hp == 0:
            client.send_unload(pkt.tick)
            logger.info(
                "Я выгружаю погибшего гражданского entity_id=%d на месте, "
                "tick=%d (не везу в убежище)", ent.id, pkt.tick,
            )
            world_model.remove_task(ent.id)
            return 0
        break

    next_target_node: int | None = None
    if pkt.own_state.type == AgentType.AMBULANCE_TEAM:
        for ent in world_model.tasks.values():
            if ent.type != EntityType.CIVILIAN:
                continue
            raw = ent.raw_sensor_data
            if raw.hp is not None and raw.hp == 0:
                continue
            if raw.buriedness is not None and raw.buriedness > 0:
                continue
            if raw.damage is None or raw.damage <= 0:
                continue
            pos = raw.position_on_edge
            if pos is None or pos in world_model.refuge_ids:
                continue
            if world_model.road_graph.has_node(pos):
                next_target_node = pos
                break

    refuge_path = choose_refuge_with_exit(
        world_model.road_graph, agent_node_id, world_model.refuge_ids,
        next_target_node,
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
            client.send_rest(pkt.tick)
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

    if state.type in (AgentType.FIRE_BRIGADE, AgentType.AMBULANCE_TEAM):
        search_target = exploration_target_node
        if (
            search_target is None
            or search_target == agent_node_id
            or not world_model.road_graph.has_node(search_target)
            or world_model.road_graph.nodes[search_target].get("area_type") != "BUILDING"
        ):
            search_target = pick_search_target(
                world_model.road_graph, agent_node_id, visited=visited_nodes,
            )
        if search_target is not None:
            search_path = plan_exploration_path(
                world_model.road_graph, agent_node_id, search_target,
                max_steps=EXPLORATION_PATH_LENGTH,
            )
            if search_path and len(search_path) > 1:
                client.send_move(pkt.tick, search_path)
                logger.info(
                    "Я иду к зданию для поиска пострадавших node=%d: %d шагов, такт=%d",
                    search_target, len(search_path), pkt.tick,
                )
                return search_target, pkt.tick

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
