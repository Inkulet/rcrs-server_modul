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
    EVENTS_CSV_ENABLED,
    EXPLORATION_MAX_TICKS,
    EXPLORATION_PATH_LENGTH,
    EXPLORATION_SEED_PRIME,
    LOG_DIAG_PERIOD,
    LOG_DIR,
    METRICS_BUDGET_MS,
    METRICS_ENABLED,
    METRICS_FILE,
    METRICS_REPORT_PERIOD,
    NO_REFUGE_MAX_RETRIES,
    POLICE_URGENCY_DISTANCE_SCALE,
    RECENT_ALLY_TARGET_TICKS,
    SAME_ROLE_CLAIM_PENALTY,
    SOCIAL_RADIUS,
    SUMMARY_JSON_ENABLED,
    TICK_BUDGET_SECONDS,
    TICK_CSV_ENABLED,
)
from decision.utility.distance import distance_factor_precomputed
from decision.utility.effort import compute_effort
from decision.utility.social import social_factor
from decision.utility.urgency import compute_urgency
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
from network.codec import (
    AGENT_TYPE_TO_SAY_ROLE,
    SAY_KIND_BURIED_HELP,
    SAY_KIND_SEARCH_CLAIM,
    SAY_KIND_TARGET_CLAIM,
    SAY_ROLE_UNKNOWN,
    encode_say_payload,
)
from world.cache import WorldModel
from world.entities import AgentType, EntityType, Position, VisibleEntity


logger = logging.getLogger(__name__)

def _metrics_report_path(agent_type: AgentType) -> Path:
    import os
    stem = Path(METRICS_FILE).stem
    suffix = Path(METRICS_FILE).suffix or ".md"
    return LOG_DIR / f"{stem}_{agent_type.value}_{os.getpid()}{suffix}"


METRICS_REPORT_PATH: Path = LOG_DIR / METRICS_FILE  # legacy, может перезаписываться одним из агентов


def _build_metrics_paths(agent_type: AgentType) -> tuple[Path | None, Path | None, Path | None]:
    import os
    pid = os.getpid()
    tag = f"{agent_type.value}_{pid}"
    tick_path = (LOG_DIR / f"tick_{tag}.csv") if TICK_CSV_ENABLED else None
    events_path = (LOG_DIR / f"events_{tag}.csv") if EVENTS_CSV_ENABLED else None
    summary_path = (LOG_DIR / f"summary_{tag}.json") if SUMMARY_JSON_ENABLED else None
    return tick_path, events_path, summary_path


def _count_filter_breakdown(
    type_relevant: list[VisibleEntity],
    dispatcher: PreFilterDispatcher,
    agent_state,
) -> tuple[int, int, int]:
    # Побочный пересчёт только для метрик воронки задач. Алгоритм принятия
    # решений использует pre_filter отдельно и на этот подсчёт не смотрит.
    n_ttl = 0
    n_fier = 0
    n_already = 0
    for e in type_relevant:
        raw = e.raw_sensor_data
        if e.type in (EntityType.CIVILIAN, EntityType.HUMAN):
            if raw.hp is not None and raw.hp == 0:
                n_already += 1
                continue
            if (raw.damage == 0 and raw.buriedness == 0):
                n_already += 1
                continue
            damage = raw.damage
            if damage is not None and damage > 0 and raw.hp is not None:
                ttl = raw.hp / damage
                try:
                    t_travel = e.computed_metrics.path_distance / dispatcher.average_speed
                except ZeroDivisionError:
                    t_travel = 0.0
                try:
                    t_work = 0.0 if raw.buriedness is None else raw.buriedness / dispatcher.work_rate
                except ZeroDivisionError:
                    t_work = 0.0
                if ttl <= t_travel + t_work:
                    n_ttl += 1
        elif e.type == EntityType.BUILDING:
            if raw.fieryness is not None and raw.fieryness not in {1, 2, 3}:
                n_fier += 1
    return n_ttl, n_fier, n_already


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


def _search_partition(
    agent_state,
    world_model: WorldModel,
) -> tuple[int | None, int | None]:
    same_role_ids = {agent_state.id}
    for ally in world_model.agents.values():
        if ally.type == agent_state.type:
            same_role_ids.add(ally.id)

    ordered_ids = sorted(same_role_ids)
    if len(ordered_ids) <= 1:
        return None, None

    return ordered_ids.index(agent_state.id), len(ordered_ids)


def _is_transport_ready_civilian(
    entity: VisibleEntity,
    refuge_ids: list[int],
) -> bool:
    if entity.type != EntityType.CIVILIAN or entity.is_ally:
        return False

    raw = entity.raw_sensor_data
    if raw.hp is not None and raw.hp <= 0:
        return False
    if raw.position_on_edge is None or raw.position_on_edge in refuge_ids:
        return False
    if raw.buriedness is not None and raw.buriedness > 0:
        return False
    if raw.damage is None or raw.damage <= 0:
        return False

    return True


def _prioritize_role_tasks(
    agent_type: AgentType,
    tasks: list[VisibleEntity],
    refuge_ids: list[int],
) -> list[VisibleEntity]:
    if agent_type == AgentType.FIRE_BRIGADE:
        rescue_tasks = [
            entity for entity in tasks
            if entity.type in (EntityType.CIVILIAN, EntityType.HUMAN)
        ]
        if rescue_tasks:
            return rescue_tasks

    if agent_type == AgentType.AMBULANCE_TEAM:
        transport_ready = [
            entity for entity in tasks
            if _is_transport_ready_civilian(entity, refuge_ids)
        ]
        if transport_ready:
            return transport_ready

    return tasks


def _same_role_claim_exclusions(
    claims: dict[int, tuple[int, int, int]],
    own_role: int,
    agent_id: int,
) -> set[int]:
    excluded: set[int] = set()
    for target_id, (_tick, role_code, speaker_id) in claims.items():
        if role_code != own_role:
            continue
        if speaker_id <= 0 or speaker_id >= agent_id:
            continue
        excluded.add(target_id)
    return excluded


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

    tick_csv_path, events_csv_path, summary_json_path = _build_metrics_paths(agent_type)
    collector_enabled = (
        METRICS_ENABLED or TICK_CSV_ENABLED
        or EVENTS_CSV_ENABLED or SUMMARY_JSON_ENABLED
    )
    metrics = MetricsCollector(
        enabled=collector_enabled, budget_ms=METRICS_BUDGET_MS,
        tick_csv_path=tick_csv_path,
        events_csv_path=events_csv_path,
        summary_json_path=summary_json_path,
        budget_total_ms=TICK_BUDGET_SECONDS * 1000.0,
    )
    client.set_metrics(metrics)

    # Фиксирую текущие веса агрегатора один раз в session — они идут в
    # per-tick CSV (для ablation) и в summary.json (ablation_config).
    metrics.set_tick_fields(
        w_c=aggregator.w_c, w_d=aggregator.w_d,
        w_e=aggregator.w_e, w_n=aggregator.w_n,
    )

    _session_weights = (aggregator.w_c, aggregator.w_d, aggregator.w_e, aggregator.w_n)

    recent_target_claims: dict[int, tuple[int, int, int]] = {}
    recent_search_claims: dict[int, tuple[int, int, int]] = {}
    own_say_role = AGENT_TYPE_TO_SAY_ROLE.get(agent_type, SAY_ROLE_UNKNOWN)

    try:
        while True:
            try:
                packet = client.receive_sense()
            except TimeoutError:
                logger.warning("Loop: KASense не пришёл в срок, ожидание следующего такта")
                continue
            except (ConnectionError, OSError) as exc:
                logger.error("Loop: соединение с ядром потеряно при получении KASense — %s", exc)
                break

            metrics.start_tick()
            metrics.set_tick_fields(
                w_c=_session_weights[0], w_d=_session_weights[1],
                w_e=_session_weights[2], w_n=_session_weights[3],
            )

            if last_processed_tick >= 0 and packet.tick - last_processed_tick > 1:
                logger.warning(
                    "Loop: пропущены такты (ядро не дождалось команды агента) "
                    "[lost_ticks=%d, prev_tick=%d, current_tick=%d]",
                    packet.tick - last_processed_tick - 1,
                    last_processed_tick, packet.tick,
                )
            last_processed_tick = packet.tick

            with metrics.phase("perception"):
                world_model.apply_perception(packet)

            agent_state   = packet.own_state
            agent_node_id = agent_state.position.entity_id
            visited_nodes.add(agent_node_id)

            metrics.set_tick_fields(
                agent_id=agent_state.id,
                agent_type=agent_type.value,
                pos_entity_id=agent_node_id,
                pos_x=agent_state.position.x,
                pos_y=agent_state.position.y,
                hp=(agent_state.hp if agent_state.hp is not None else ""),
                water_level=agent_state.resources.water_quantity,
                is_transporting=int(agent_state.resources.is_transporting),
                n_visible=len(packet.visible_entities),
            )

            metrics.gauge("cache_size", len(world_model.tasks))
            metrics.gauge("visible_entities", len(packet.visible_entities))
            metrics.gauge("ally_states", len(packet.ally_states))
            metrics.gauge("blockades_on_map", sum(
                1 for e in world_model.tasks.values()
                if e.type == EntityType.BLOCKADE
            ))


            if agent_node_id == 0:
                logger.warning(
                    "Loop: позиция агента не определена (entity_id=0), отправляется AKRest [tick=%d]", packet.tick,
                )
                client.send_rest(packet.tick)
                continue

            if agent_state.hp is not None and agent_state.hp <= 0:
                logger.info("Loop: агент мёртв, действия заблокированы [tick=%d]", packet.tick)
                client.send_rest(packet.tick)
                current_target_id = None
                continue

            if agent_state.buriedness is not None and agent_state.buriedness > 0:
                logger.info(
                    "Loop: агент завален, отправляется сигнал о помощи [buriedness=%d, tick=%d]",
                    agent_state.buriedness, packet.tick,
                )
                client.send_rest(packet.tick)
                try:
                    say_data = encode_say_payload(
                        SAY_KIND_BURIED_HELP, agent_state.id, own_say_role,
                    )
                    client.send_say(packet.tick, say_data)
                except (ConnectionError, OSError, struct.error) as exc:
                    logger.warning("Loop: сигнал BURIED_HELP не отправлен — %s", exc)
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
            metrics.set_tick_field("n_type_relevant", len(type_relevant))

            if packet.tick % LOG_DIAG_PERIOD == 0 or type_relevant:
                logger.info(
                    "Loop: диагностика такта [agent_type=%s, tick=%d, cache_tasks=%d, type_relevant=%d]",
                    agent_type.value, packet.tick,
                    len(world_model.tasks), len(type_relevant),
                )

            try:
                with metrics.phase("pre_filter"):
                    filtered_tasks = dispatcher.filter_tasks(
                        agent_state, type_relevant, world_model=world_model,
                    )
            except NeedRefugeException:
                logger.info("Loop (fire): воды нет, агент направляется в убежище")
                refuge_path = nearest_refuge_path(
                    world_model.road_graph, agent_node_id, world_model.refuge_ids,
                )
                if refuge_path:
                    client.send_move(packet.tick, refuge_path)
                else:
                    client.send_rest(packet.tick)
                continue

            metrics.gauge("filtered_tasks", len(filtered_tasks))
            n_ttl_f, n_fier_f, n_already_f = _count_filter_breakdown(
                type_relevant, dispatcher, agent_state,
            )
            metrics.set_tick_fields(
                n_pre_filter_pass=len(filtered_tasks),
                n_ttl_filtered=n_ttl_f,
                n_fieryness_filtered=n_fier_f,
                n_already_rescued_filtered=n_already_f,
            )

            filtered_tasks = _prioritize_role_tasks(
                agent_type, filtered_tasks, world_model.refuge_ids,
            )

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
                    "Loop: utility вычислена [agent_id=%d, entity_id=%d, U=%.4f]",
                    agent_state.id, entity.id, utility,
                )

            for tid in packet.heard_target_ids:
                role_code = packet.heard_target_roles.get(tid, SAY_ROLE_UNKNOWN)
                speaker_id = packet.heard_target_speakers.get(tid, 0)
                recent_target_claims[tid] = (packet.tick, role_code, speaker_id)
            for tid in packet.heard_search_target_ids:
                role_code = packet.heard_search_target_roles.get(tid, SAY_ROLE_UNKNOWN)
                speaker_id = packet.heard_search_target_speakers.get(tid, 0)
                recent_search_claims[tid] = (packet.tick, role_code, speaker_id)

            expired = [
                tid for tid, (last_tick, _role_code, _speaker_id) in recent_target_claims.items()
                if packet.tick - last_tick > RECENT_ALLY_TARGET_TICKS
            ]
            for tid in expired:
                recent_target_claims.pop(tid, None)
            expired_search = [
                tid for tid, (last_tick, _role_code, _speaker_id) in recent_search_claims.items()
                if packet.tick - last_tick > RECENT_ALLY_TARGET_TICKS
            ]
            for tid in expired_search:
                recent_search_claims.pop(tid, None)

            if current_target_id is not None and current_target_id in recent_target_claims:
                _last_tick, role_code, speaker_id = recent_target_claims[current_target_id]
                if (
                    role_code == own_say_role
                    and speaker_id > 0
                    and speaker_id < agent_state.id
                ):
                    logger.info(
                        "Loop: текущая цель уступлена союзнику той же роли с меньшим agent_id "
                        "[target_id=%d, claimer_agent_id=%d]",
                        current_target_id, speaker_id,
                    )
                    current_target_id = None

            for tid, (_last_tick, role_code, _speaker_id) in recent_target_claims.items():
                if tid not in utilities or tid == current_target_id:
                    continue
                penalty = CLAIMED_TARGET_PENALTY
                if role_code == own_say_role:
                    penalty += SAME_ROLE_CLAIM_PENALTY
                    if agent_type == AgentType.FIRE_BRIGADE:
                        penalty += ALLY_TARGET_LONGTERM_PENALTY
                old_u = utilities[tid]
                utilities[tid] = old_u - penalty
                logger.debug(
                    "Loop: utility цели снижена по claim союзника [target_id=%d, role=%d, U: %.4f → %.4f]",
                    tid, role_code, old_u, utilities[tid],
                )

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

            target_changed = int(current_target_id != prev_target)
            target_utility = utilities.get(current_target_id) if current_target_id is not None else None
            prev_utility = utilities.get(prev_target) if prev_target is not None else None
            if target_changed and (prev_target is not None or current_target_id is not None):
                metrics.record_event(
                    "target_change",
                    tick=packet.tick, agent_id=agent_state.id,
                    old_target=(prev_target if prev_target is not None else -1),
                    new_target=(current_target_id if current_target_id is not None else -1),
                    old_U=("" if prev_utility is None else round(prev_utility, 6)),
                    new_U=("" if target_utility is None else round(target_utility, 6)),
                    reason=("stuck" if is_stuck else "hysteresis"),
                )
            if is_stuck:
                metrics.record_event(
                    "stuck_detected",
                    tick=packet.tick, agent_id=agent_state.id,
                    position=agent_node_id,
                )

            # Разложение utility для выбранной цели (для ablation / CSV).
            f_u = f_d = f_e = f_s = None
            tgt_distance = None
            if current_target_id is not None and current_target_id in utilities:
                sel_entity = world_model.tasks.get(current_target_id)
                if sel_entity is not None:
                    try:
                        t_travel_sel = sel_entity.computed_metrics.path_distance / AVERAGE_SPEED
                    except ZeroDivisionError:
                        t_travel_sel = 0.0
                    bur_sel = sel_entity.raw_sensor_data.buriedness
                    try:
                        t_work_sel = 0.0 if bur_sel is None else bur_sel / dispatcher.work_rate
                    except ZeroDivisionError:
                        t_work_sel = 0.0
                    if agent_type == AgentType.POLICE_FORCE:
                        raw_min_d = _min_distance_blockade_to_important(sel_entity, world_model)
                        td_sel = 1.0e9 if raw_min_d == float("inf") else raw_min_d / POLICE_URGENCY_DISTANCE_SCALE
                    else:
                        td_sel = sel_entity.computed_metrics.path_distance
                    try:
                        f_u = compute_urgency(
                            agent_state, entity=sel_entity,
                            t_travel=t_travel_sel, t_work=t_work_sel,
                            task_distance=td_sel,
                        )
                        f_e = compute_effort(agent_state, entity=sel_entity)
                        f_d = distance_factor_precomputed(
                            sel_entity.computed_metrics.path_distance,
                            max_map_distance=world_model.max_map_distance,
                        )
                        nav_sel = (
                            sel_entity.raw_sensor_data.position_on_edge
                            if sel_entity.raw_sensor_data.position_on_edge is not None
                            else sel_entity.id
                        )
                        attrs_sel = world_model.road_graph.nodes.get(nav_sel, {})
                        tgt_position_sel = Position(
                            entity_id=nav_sel,
                            x=int(attrs_sel.get("x", 0)),
                            y=int(attrs_sel.get("y", 0)),
                        )
                        f_s = social_factor(
                            world_model, tgt_position_sel, agent_state.type,
                            current_agent_id=agent_state.id, radius=SOCIAL_RADIUS,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Loop: разложение utility для метрик не выполнено — %s", exc)
                    tgt_distance = sel_entity.computed_metrics.path_distance

            metrics.set_tick_fields(
                target_id=(current_target_id if current_target_id is not None else -1),
                target_id_prev=(prev_target if prev_target is not None else -1),
                target_changed=target_changed,
                target_utility=("" if target_utility is None else round(target_utility, 6)),
                target_distance=("" if tgt_distance is None else round(tgt_distance, 2)),
                U_total=("" if target_utility is None else round(target_utility, 6)),
                f_urgency=("" if f_u is None else round(f_u, 6)),
                f_dist=("" if f_d is None else round(f_d, 6)),
                f_effort=("" if f_e is None else round(f_e, 6)),
                f_social=("" if f_s is None else round(f_s, 6)),
                is_stuck=int(bool(is_stuck)),
                is_idle=int(current_target_id is None),
                agent_status=("Transporting" if agent_state.resources.is_transporting
                              else ("Idle" if current_target_id is None else "Moving")),
            )

            if current_target_id is None:
                logger.info("Loop: доступных задач нет, агент переходит в режим exploration [tick=%d]", packet.tick)
            elif not is_stuck:
                logger.info("Loop: цель выбрана/удержана [target_id=%s, tick=%d]", current_target_id, packet.tick)

            try:
                action_sent = False

                action_kind_dispatched: str = ""
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
                            if agent_type == AgentType.FIRE_BRIGADE:
                                action_kind_dispatched = "EXTINGUISH"
                            elif agent_type == AgentType.AMBULANCE_TEAM:
                                action_kind_dispatched = "RESCUE"
                            else:
                                action_kind_dispatched = "CLEAR"
                        else:
                            action_kind_dispatched = "MOVE"

                if not action_sent:
                    if agent_type == AgentType.POLICE_FORCE:
                        with metrics.phase("dispatch"):
                            sent, _ = try_clear_local_blockade(
                                client, packet.tick, agent_node_id, world_model,
                                agent_state=agent_state,
                            )
                        if sent:
                            action_sent = True
                            action_kind_dispatched = "CLEAR"

                if not action_sent:
                    with metrics.phase("explore"):
                        exploration_target_node, exploration_start_tick = _explore_and_update(
                            client, packet, agent_state, agent_node_id,
                            world_model, visited_nodes,
                            exploration_target_node, exploration_start_tick,
                            recent_search_claims, own_say_role,
                        )
                    action_kind_dispatched = "MOVE"

                metrics.set_tick_field("action_dispatched", action_kind_dispatched)

            except (ConnectionError, OSError) as exc:
                logger.error("Loop: соединение с ядром потеряно при отправке команды — %s", exc)
                break

            if current_target_id is not None:
                with metrics.phase("say"):
                    try:
                        say_data = encode_say_payload(
                            SAY_KIND_TARGET_CLAIM, current_target_id, own_say_role,
                        )
                        client.send_say(packet.tick, say_data)
                        metrics.record_communication(
                            packet.tick, agent_state.id, len(say_data),
                            kind="coord", channel="say",
                        )
                    except (ConnectionError, OSError) as exc:
                        logger.warning("Loop: AKSay с TARGET_CLAIM не отправлен — %s", exc)

            metrics.stop_tick(packet.tick)

            tick_ms = metrics.last_tick_ms()
            if tick_ms > TICK_BUDGET_SECONDS * 1000.0:
                logger.warning(
                    "Loop: превышен бюджет времени такта [tick=%d, duration_ms=%.1f, tasks=%d]",
                    packet.tick, tick_ms, len(type_relevant),
                )

            if packet.tick % METRICS_REPORT_PERIOD == 0:
                metrics.write_report(
                    _metrics_report_path(agent_type), agent_state.id, agent_type.value,
                )

    except KeyboardInterrupt:
        logger.info("Loop: получен KeyboardInterrupt, завершение работы агента")
    finally:
        final_agent_id = agent_state.id if "agent_state" in locals() else -1
        try:
            metrics.write_report(
                _metrics_report_path(agent_type), final_agent_id, agent_type.value,
            )
        except (OSError, NameError) as exc:
            logger.warning("Loop: финальный отчёт метрик не записан — %s", exc)
        try:
            metrics.write_summary_json(final_agent_id, agent_type.value)
        except (OSError, NameError) as exc:
            logger.warning("Loop: summary.json не записан — %s", exc)
        metrics.close()
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
                "Loop (ambulance): погибший гражданский выгружен на месте, маршрут в убежище отменён "
                "[entity_id=%d, tick=%d]", ent.id, pkt.tick,
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
            logger.info("Loop (ambulance): гражданский выгружен в убежище [refuge_id=%d, tick=%d]", refuge_node, pkt.tick)
        else:
            ref_x = ref_attrs.get("x", 0)
            ref_y = ref_attrs.get("y", 0)
            client.send_move(pkt.tick, refuge_path, dest_x=ref_x, dest_y=ref_y)
            logger.info("Loop (ambulance): транспортировка гражданского к убежищу [refuge_id=%d, tick=%d]", refuge_node, pkt.tick)
        return 0
    else:
        no_refuge_counter += 1
        if no_refuge_counter >= NO_REFUGE_MAX_RETRIES:
            logger.error(
                "Loop (ambulance): убежище не найдено, лимит попыток исчерпан [retries=%d, tick=%d]",
                no_refuge_counter, pkt.tick,
            )
            client.send_rest(pkt.tick)
            return 0
        logger.warning("Loop (ambulance): убежище не найдено [attempt=%d, tick=%d]", no_refuge_counter, pkt.tick)
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
    recent_search_claims: dict[int, tuple[int, int, int]],
    own_say_role: int,
) -> tuple[int | None, int]:
    from world.entities import AgentState, PerceptionPacket
    pkt: PerceptionPacket = packet  # type: ignore[assignment]
    state: AgentState = agent_state  # type: ignore[assignment]

    if state.type in (AgentType.FIRE_BRIGADE, AgentType.AMBULANCE_TEAM):
        partition_index, partition_count = _search_partition(state, world_model)
        excluded_targets = _same_role_claim_exclusions(
            recent_search_claims, own_say_role, state.id,
        )
        search_target = exploration_target_node
        if (
            search_target is None
            or search_target == agent_node_id
            or not world_model.road_graph.has_node(search_target)
            or world_model.road_graph.nodes[search_target].get("area_type") != "BUILDING"
            or search_target in excluded_targets
        ):
            search_target = pick_search_target(
                world_model.road_graph,
                agent_node_id,
                visited=visited_nodes,
                partition_index=partition_index,
                partition_count=partition_count,
                excluded=excluded_targets,
            )
        if search_target is not None:
            search_path = plan_exploration_path(
                world_model.road_graph, agent_node_id, search_target,
                max_steps=EXPLORATION_PATH_LENGTH,
            )
            if search_path and len(search_path) > 1:
                client.send_move(pkt.tick, search_path)
                try:
                    client.send_say(
                        pkt.tick,
                        encode_say_payload(
                            SAY_KIND_SEARCH_CLAIM, search_target, own_say_role,
                        ),
                    )
                except (ConnectionError, OSError, struct.error) as exc:
                    logger.warning("Loop: AKSay с SEARCH_CLAIM не отправлен — %s", exc)
                logger.info(
                    "Loop: маршрут к зданию для поиска пострадавших построен [target_node=%d, path_len=%d, tick=%d]",
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
                "Loop: выбрана новая exploration-цель [target_node=%d, from_node=%d, tick=%d]",
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
            "Loop: маршрут к exploration-цели построен [target_node=%d, path_len=%d, tick=%d]",
            exploration_target_node, len(exp_path), pkt.tick,
        )
    else:
        exploration_target_node = None
        rw_path = random_walk(world_model.road_graph, agent_node_id, visited=visited_nodes)
        if len(rw_path) > 1:
            client.send_move(pkt.tick, rw_path)
            logger.info("Loop: fallback на random walk [path_len=%d, tick=%d]", len(rw_path), pkt.tick)
        else:
            client.send_rest(pkt.tick)
            logger.warning("Loop: exploration невозможна — узел-тупик без соседей [node=%d, tick=%d]", agent_node_id, pkt.tick)

    return exploration_target_node, exploration_start_tick


__all__ = [
    "run_field_agent",
    "_search_partition",
    "_is_transport_ready_civilian",
    "_prioritize_role_tasks",
    "_same_role_claim_exclusions",
]
