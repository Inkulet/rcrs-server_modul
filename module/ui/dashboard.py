from __future__ import annotations

import copy
import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from module.config_profile import model_config_from_dict, model_config_to_dict
from module.data_models import AgentType
from module.sample_data import (
    BASELINE_PROFILE_NAME,
    DEFAULT_LIVE_STATE_PATH,
    build_baseline_config,
    build_demo_engine,
    build_demo_scenario,
    load_live_state_snapshot,
)
from module.simulation import SimulationEngine


PROFILE_WEIGHT_KEYS = ["w_c", "w_d", "w_e", "w_n"]
PROFILE_AGENT_KEYS = ["FIRE_BRIGADE", "AMBULANCE_TEAM", "POLICE_FORCE"]
PROFILE_CONSTANT_KEYS = [
    "max_map_distance",
    "max_buriedness",
    "max_total_area",
    "max_repair_cost",
    "temperature_max",
    "social_radius",
    "ambulance_clear_rate",
    "travel_speed",
    "epsilon",
    "c_switch",
]


@dataclass
class DashboardState:
    """Унифицированная структура упрощает общий рендер Debug/Commission для Sample и Live источников."""

    source_mode: str
    tick: int
    agent_id: str
    agent_type: str
    agent_state: Dict[str, Any]
    decision: Dict[str, Any]
    utility_matrix: List[Dict[str, Any]]
    visible_entities: List[Dict[str, Any]]
    decision_log: List[Dict[str, Any]]
    warnings: List[str]


@dataclass
class AuditResult:
    """Результат аудита показывает комиссии, что реализация следует формулам 2.2 без ручных трактовок."""

    total_checks: int
    passed_checks: int
    failed_messages: List[str]
    checks_table: List[Dict[str, Any]]


def _apply_visual_theme() -> None:
    """Нестандартная тема разделяет режимы Debug/Комиссия и делает интерфейс презентационно-читаемым."""
    st.markdown(
        """
        <style>
            :root {
                --bg-grad-a: #f4f8ff;
                --bg-grad-b: #ecf9f1;
                --accent: #0f766e;
                --accent-2: #1d4ed8;
                --warning: #b45309;
                --danger: #b91c1c;
                --card: #ffffff;
                --muted: #475569;
            }
            .stApp {
                background: linear-gradient(135deg, var(--bg-grad-a) 0%, var(--bg-grad-b) 100%);
                font-family: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", sans-serif;
            }
            .hero {
                background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 50%, #0f766e 100%);
                color: #f8fafc;
                border-radius: 14px;
                padding: 1rem 1.2rem;
                margin-bottom: 0.8rem;
            }
            .hero h2 {
                margin: 0;
                font-size: 1.35rem;
                font-weight: 700;
            }
            .hero p {
                margin: 0.35rem 0 0 0;
                font-size: 0.96rem;
                color: #dbeafe;
            }
            .pill {
                display: inline-block;
                border-radius: 999px;
                padding: 0.12rem 0.6rem;
                font-size: 0.8rem;
                font-weight: 600;
                background: rgba(255, 255, 255, 0.2);
                margin-right: 0.35rem;
            }
            .runbook {
                background: var(--card);
                border: 1px solid #dbeafe;
                border-radius: 12px;
                padding: 0.95rem 1rem;
                margin-bottom: 0.75rem;
            }
            .kpi-note {
                color: var(--muted);
                font-size: 0.88rem;
                margin-top: -0.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_engine() -> SimulationEngine:
    """Sample engine хранится в session_state, чтобы профиль матмодели применялся детерминированно."""
    if "baseline_profile" not in st.session_state:
        visible_entities, refuges = build_demo_scenario()
        baseline_config = build_baseline_config(visible_entities, refuges)
        baseline_profile = model_config_to_dict(baseline_config)

        st.session_state.baseline_profile_name = BASELINE_PROFILE_NAME
        st.session_state.baseline_profile = baseline_profile
        st.session_state.active_profile = copy.deepcopy(baseline_profile)
        st.session_state.engine = build_demo_engine(config=baseline_config)
        _sync_form_with_profile(st.session_state.active_profile, force=True)

    if "stop_event" not in st.session_state:
        st.session_state.stop_event = threading.Event()
    if "bg_thread" not in st.session_state:
        st.session_state.bg_thread = None

    _sync_form_with_profile(st.session_state.active_profile, force=False)
    return st.session_state.engine


def _sync_form_with_profile(profile: Dict[str, Any], force: bool) -> None:
    """Синхронизация формы нужна, чтобы UI и фактическая конфигурация расчета не расходились."""
    for agent_key in PROFILE_AGENT_KEYS:
        for weight_key in PROFILE_WEIGHT_KEYS:
            state_key = f"profile_{agent_key}_{weight_key}"
            value = float(profile["weights_by_agent"][agent_key][weight_key])
            if force or state_key not in st.session_state:
                st.session_state[state_key] = value

    for constant_key in PROFILE_CONSTANT_KEYS:
        state_key = f"profile_const_{constant_key}"
        value = float(profile["constants"][constant_key])
        if force or state_key not in st.session_state:
            st.session_state[state_key] = value


def _collect_profile_from_form() -> Dict[str, Any]:
    """Единый сбор профиля исключает расхождение import/apply/reset поведения."""
    return {
        "weights_by_agent": {
            agent_key: {
                weight_key: float(st.session_state[f"profile_{agent_key}_{weight_key}"])
                for weight_key in PROFILE_WEIGHT_KEYS
            }
            for agent_key in PROFILE_AGENT_KEYS
        },
        "constants": {
            constant_key: float(st.session_state[f"profile_const_{constant_key}"])
            for constant_key in PROFILE_CONSTANT_KEYS
        },
    }


def _start_background(engine: SimulationEngine, interval_seconds: float) -> None:
    """Фоновый поток включается только для Sample Mode, где симуляция исполняется внутри Streamlit."""
    background_thread = st.session_state.bg_thread
    if background_thread is not None and background_thread.is_alive():
        return

    stop_event = st.session_state.stop_event
    stop_event.clear()
    thread = threading.Thread(
        target=engine.run_background,
        kwargs={"stop_event": stop_event, "interval_seconds": interval_seconds},
        daemon=True,
    )
    thread.start()
    st.session_state.bg_thread = thread


def _stop_background() -> None:
    """Остановка потока обязательна при смене режима, иначе метрики будут изменяться в фоне неконтролируемо."""
    stop_event = st.session_state.stop_event
    stop_event.set()

    background_thread = st.session_state.bg_thread
    if background_thread is not None and background_thread.is_alive():
        background_thread.join(timeout=1.5)
    st.session_state.bg_thread = None


def _apply_profile(profile: Dict[str, Any]) -> None:
    """Пересоздаем engine, чтобы все агенты гарантированно считали U_ij на одном и том же профиле."""
    _stop_background()
    config = model_config_from_dict(profile)
    st.session_state.engine = build_demo_engine(config=config)
    st.session_state.active_profile = copy.deepcopy(profile)
    _sync_form_with_profile(st.session_state.active_profile, force=True)


def _serialize_agent_state(state: Any) -> Dict[str, Any]:
    """Enum переводим в строки, чтобы статус агента в Debug читался без внутренних python-типов."""
    data = asdict(state)
    data["type"] = state.type.value
    data["state"] = state.state.value
    return data


def _safe_float(value: Any) -> Optional[float]:
    """Нормализация числа нужна для работы с `inf` и строковыми значениями из snapshot JSON."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "inf":
            return float("inf")
        if lowered == "-inf":
            return float("-inf")
        try:
            return float(lowered)
        except ValueError:
            return None
    return None


def _safe_int(value: Any) -> Optional[int]:
    """Валидация integer-полей исключает падения аудита на частично заполненных сенсорных данных."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_matrix_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Приводим Sample/Live матрицы к единому формату, чтобы рендер и аудит не дублировали ветки."""
    normalized: List[Dict[str, Any]] = []
    for raw_row in raw_rows:
        utility_raw = raw_row.get("U_ij", raw_row.get("utility_score"))
        utility_score = _safe_float(utility_raw)

        normalized.append(
            {
                "entity_id": _safe_int(raw_row.get("entity_id")),
                "entity_type": str(raw_row.get("entity_type", "UNKNOWN")),
                "f_urgency": _safe_float(raw_row.get("f_urgency")) or 0.0,
                "f_dist": _safe_float(raw_row.get("f_dist")) or 0.0,
                "f_effort": _safe_float(raw_row.get("f_effort")) or 0.0,
                "f_social": _safe_float(raw_row.get("f_social")) or 0.0,
                "utility_score": utility_score if utility_score is not None else float("-inf"),
                "passed_prefilter": bool(raw_row.get("passed_prefilter", False)),
                "prefilter_reason": str(raw_row.get("prefilter_reason", "")),
            }
        )

    return sorted(
        normalized,
        key=lambda row: (
            float("-inf") if not math.isfinite(row["utility_score"]) else row["utility_score"],
            -(row["entity_id"] or 0),
        ),
        reverse=True,
    )


def _render_matrix_table(rows: List[Dict[str, Any]], caption: Optional[str] = None) -> None:
    """Единый рендер матрицы гарантирует одинаковый формат в Debug/Commission и Sample/Live."""
    if caption:
        st.caption(caption)

    if not rows:
        st.info("Матрица полезности пуста для выбранного агента.")
        return

    dataframe = pd.DataFrame(
        [
            {
                "entity_id": row["entity_id"],
                "entity_type": row["entity_type"],
                "f_urgency": round(row["f_urgency"], 6),
                "f_dist": round(row["f_dist"], 6),
                "f_effort": round(row["f_effort"], 6),
                "f_social": round(row["f_social"], 6),
                "U_ij": row["utility_score"] if math.isfinite(row["utility_score"]) else "-inf",
                "passed_prefilter": row["passed_prefilter"],
                "prefilter_reason": row["prefilter_reason"],
            }
            for row in rows
        ]
    )
    st.dataframe(dataframe, width="stretch", hide_index=True)


def _build_entity_lookup(visible_entities: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Lookup по entity_id нужен аудиту pre-filter правил, чтобы связать matrix-row и сенсорные поля."""
    result: Dict[int, Dict[str, Any]] = {}
    for entity in visible_entities:
        entity_id = _safe_int(entity.get("id"))
        if entity_id is None:
            continue
        result[entity_id] = entity
    return result


def _extract_profile_for_agent(
    agent_type: str,
    source_mode: str,
) -> Tuple[Optional[Dict[str, float]], Dict[str, float], bool]:
    """Проверки, зависящие от констант/весов, разрешены только там, где профиль гарантированно известен."""
    if source_mode == "Sample Mode" and "active_profile" in st.session_state:
        profile = st.session_state.active_profile
        raw_weights = profile.get("weights_by_agent", {}).get(agent_type)
        raw_constants = profile.get("constants", {})
        if isinstance(raw_weights, dict):
            weights = {key: float(raw_weights.get(key, 0.0)) for key in PROFILE_WEIGHT_KEYS}
        else:
            weights = None

        constants = {
            "travel_speed": float(raw_constants.get("travel_speed", 1.0)),
            "ambulance_clear_rate": float(raw_constants.get("ambulance_clear_rate", 1.0)),
            "c_switch": float(raw_constants.get("c_switch", 0.0)),
        }
        return weights, constants, True

    # В Live режиме без профиля не проверяем формулу U и TTL-порог,
    # чтобы не получить ложные FAIL из-за неизвестных runtime-констант.
    return None, {"travel_speed": 1.0, "ambulance_clear_rate": 1.0, "c_switch": 0.0}, False


def _evaluate_audit(state: DashboardState) -> AuditResult:
    """Аудит подтверждает соответствие разделам 2.2 и 4 прямо на фактическом состоянии агента."""
    checks: List[Dict[str, Any]] = []
    failed: List[str] = []

    weights, constants, strict_constant_checks = _extract_profile_for_agent(state.agent_type, state.source_mode)
    entity_lookup = _build_entity_lookup(state.visible_entities)

    for row in state.utility_matrix:
        entity_id = row.get("entity_id")
        passed_prefilter = bool(row.get("passed_prefilter", False))

        for factor_name in ["f_urgency", "f_dist", "f_effort", "f_social"]:
            factor_value = _safe_float(row.get(factor_name))
            factor_ok = factor_value is not None and 0.0 <= float(factor_value) <= 1.0
            check_name = f"{factor_name} in [0,1] | entity={entity_id}"
            checks.append({"check": check_name, "passed": factor_ok})
            if not factor_ok:
                failed.append(f"{check_name}: получено {factor_value}")

        if passed_prefilter and weights is not None:
            utility_value = _safe_float(row.get("utility_score"))
            expected_utility = (
                weights["w_c"] * float(row.get("f_urgency", 0.0))
                - (
                    weights["w_d"] * float(row.get("f_dist", 0.0))
                    + weights["w_e"] * float(row.get("f_effort", 0.0))
                    + weights["w_n"] * float(row.get("f_social", 0.0))
                )
            )
            formula_ok = utility_value is not None and math.isfinite(utility_value) and abs(utility_value - expected_utility) <= 1e-6
            check_name = f"U_ij по формуле 2.2 | entity={entity_id}"
            checks.append({"check": check_name, "passed": formula_ok})
            if not formula_ok:
                failed.append(
                    f"{check_name}: фактическое={utility_value}, ожидаемое={expected_utility:.6f}"
                )

        entity = entity_lookup.get(int(entity_id)) if entity_id is not None else None
        if entity is None:
            continue

        raw_data = entity.get("raw_sensor_data", {}) or {}
        metrics = entity.get("computed_metrics", {}) or {}

        if state.agent_type == AgentType.FIRE_BRIGADE.value and passed_prefilter:
            fieryness = _safe_int(raw_data.get("fieryness"))
            fire_filter_ok = fieryness not in {4, 5, 6, 7, 8}
            check_name = f"PreFilter Fire (Fieryness != 4..8) | entity={entity_id}"
            checks.append({"check": check_name, "passed": fire_filter_ok})
            if not fire_filter_ok:
                failed.append(f"{check_name}: Fieryness={fieryness}")

        if state.agent_type == AgentType.AMBULANCE_TEAM.value and passed_prefilter:
            hp = _safe_int(raw_data.get("hp"))
            damage = _safe_int(raw_data.get("damage"))
            buriedness = _safe_int(raw_data.get("buriedness"))

            hp_ok = hp is None or hp > 0
            checks.append({"check": f"PreFilter Ambulance (HP>0) | entity={entity_id}", "passed": hp_ok})
            if not hp_ok:
                failed.append(f"PreFilter Ambulance (HP>0) | entity={entity_id}: HP={hp}")

            stable_ok = not (damage == 0 and (buriedness is None or buriedness == 0))
            checks.append({"check": f"PreFilter Ambulance (not stable) | entity={entity_id}", "passed": stable_ok})
            if not stable_ok:
                failed.append(
                    f"PreFilter Ambulance (not stable) | entity={entity_id}: Damage={damage}, Buriedness={buriedness}"
                )

            ttl_value: float
            if hp is None or hp <= 0:
                ttl_value = 0.0
            elif damage is None or damage == 0:
                ttl_value = float("inf")
            elif damage > 0:
                ttl_value = float(hp) / float(damage)
            else:
                ttl_value = float("inf")

            path_distance = _safe_float(metrics.get("path_distance"))
            travel_speed = max(float(constants.get("travel_speed", 1.0)), 1e-9)
            travel_time = float("inf") if path_distance is None else max(path_distance, 0.0) / travel_speed
            clear_rate = max(float(constants.get("ambulance_clear_rate", 1.0)), 1e-9)
            work_time = 0.0 if buriedness is None else max(float(buriedness), 0.0) / clear_rate

            if strict_constant_checks:
                ttl_ok = math.isinf(ttl_value) or ttl_value > (travel_time + work_time)
                checks.append({"check": f"PreFilter Ambulance (TTL > travel+work) | entity={entity_id}", "passed": ttl_ok})
                if not ttl_ok:
                    failed.append(
                        f"PreFilter Ambulance (TTL > travel+work) | entity={entity_id}: "
                        f"TTL={ttl_value:.4f}, travel+work={travel_time + work_time:.4f}"
                    )

        if state.agent_type == AgentType.POLICE_FORCE.value and passed_prefilter:
            repair_cost = _safe_int(raw_data.get("repair_cost"))
            repair_ok = repair_cost is not None and repair_cost > 0
            check_name = f"PreFilter Police (RepairCost>0) | entity={entity_id}"
            checks.append({"check": check_name, "passed": repair_ok})
            if not repair_ok:
                failed.append(f"{check_name}: RepairCost={repair_cost}")

    total = len(checks)
    passed = sum(1 for row in checks if row["passed"])
    return AuditResult(total_checks=total, passed_checks=passed, failed_messages=failed, checks_table=checks)


def _render_audit_tab(audit: AuditResult) -> None:
    """Таблица аудита превращает «верим в модель» в измеряемый и проверяемый артефакт для комиссии."""
    st.subheader("Аудит соответствия матмодели 2.2")

    if audit.total_checks == 0:
        st.info("Недостаточно данных для аудита (пустая матрица или нет видимых сущностей).")
        return

    quality = 100.0 * float(audit.passed_checks) / float(max(1, audit.total_checks))
    col_left, col_center, col_right = st.columns(3)
    col_left.metric("Проверок", audit.total_checks)
    col_center.metric("Успешно", audit.passed_checks)
    col_right.metric("Качество", f"{quality:.1f}%")

    check_df = pd.DataFrame(
        [
            {
                "check": row["check"],
                "status": "PASS" if row["passed"] else "FAIL",
            }
            for row in audit.checks_table
        ]
    )
    st.dataframe(check_df, width="stretch", hide_index=True)

    if audit.failed_messages:
        st.error("Обнаружены отклонения от формализации. Список:")
        for message in audit.failed_messages:
            st.write(f"- {message}")
    else:
        st.success("Отклонений не найдено: текущая выборка соответствует формализации 2.2 и pre-filter раздела 4.")


def _best_candidate(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Лучший кандидат нужен для верхних KPI и объяснения выбора цели пользователю."""
    valid_rows = [row for row in rows if bool(row.get("passed_prefilter")) and math.isfinite(row.get("utility_score", float("-inf")))]
    if not valid_rows:
        return None
    return sorted(valid_rows, key=lambda row: (row["utility_score"], -(row.get("entity_id") or 0)), reverse=True)[0]


def _render_debug_panel(state: DashboardState, audit: AuditResult) -> None:
    """Debug-режим показывает полный аналитический слой для автора диплома и проверки каждого шага."""
    best = _best_candidate(state.utility_matrix)
    passed_count = sum(1 for row in state.utility_matrix if row.get("passed_prefilter"))

    st.markdown(
        f"""
        <div class="hero">
            <h2>Debug Console: Utility Decision Trace</h2>
            <p>
                <span class="pill">source={state.source_mode}</span>
                <span class="pill">agent={state.agent_id}</span>
                <span class="pill">type={state.agent_type}</span>
                <span class="pill">tick={state.tick}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_col_1, kpi_col_2, kpi_col_3, kpi_col_4 = st.columns(4)
    kpi_col_1.metric("Кандидаты", len(state.utility_matrix))
    kpi_col_2.metric("После pre-filter", passed_count)
    kpi_col_3.metric("Проверки 2.2", f"{audit.passed_checks}/{audit.total_checks}")
    kpi_col_4.metric("Лучший U_ij", "N/A" if best is None else f"{best['utility_score']:.6f}")

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("Состояние агента")
        st.json(state.agent_state, expanded=False)

        st.subheader("Текущее решение")
        st.json(state.decision, expanded=False)

        if state.warnings:
            st.subheader("Валидационные предупреждения")
            warning_df = pd.DataFrame({"warning": state.warnings})
            st.dataframe(warning_df, width="stretch", hide_index=True)

    with right_col:
        st.subheader("Матрица полезности")
        _render_matrix_table(
            state.utility_matrix,
            caption="Раскладка факторов f_urgency/f_dist/f_effort/f_social и итогового U_ij.",
        )

    st.subheader("Лог решений")
    if state.decision_log:
        log_df = pd.DataFrame(state.decision_log)
        st.dataframe(log_df, width="stretch", hide_index=True)
    else:
        st.info("Лог решений пока пуст.")

    st.subheader("Видимые сущности (raw + computed)")
    if state.visible_entities:
        visible_df = pd.json_normalize(state.visible_entities)
        st.dataframe(visible_df, width="stretch", hide_index=True)
    else:
        st.info("Список видимых сущностей пуст.")


def _render_commission_panel(state: DashboardState, audit: AuditResult) -> None:
    """Режим для комиссии показывает только ключевые объяснения и измеримые результаты без перегруза деталями."""
    best = _best_candidate(state.utility_matrix)
    decision_action = str(state.decision.get("action", "N/A"))
    decision_target = state.decision.get("selected_target_id", "N/A")

    st.markdown(
        f"""
        <div class="hero">
            <h2>Многоагентная система принятия решений (RCRS)</h2>
            <p>
                Реальное время: <b>tick={state.tick}</b> | Агент: <b>{state.agent_type} #{state.agent_id}</b> |
                Режим данных: <b>{state.source_mode}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Текущее действие", decision_action)
    metric_2.metric("Целевая сущность", decision_target)
    metric_3.metric("Лучшая полезность", "N/A" if best is None else f"{best['utility_score']:.4f}")
    metric_4.metric("Соответствие 2.2", f"{audit.passed_checks}/{audit.total_checks}")

    st.markdown('<div class="kpi-note">Показатели обновляются каждый тик и формируются на основе функции полезности U_ij.</div>', unsafe_allow_html=True)

    if state.utility_matrix:
        top_rows = [
            row
            for row in state.utility_matrix
            if bool(row.get("passed_prefilter")) and math.isfinite(row.get("utility_score", float("-inf")))
        ][:5]
        if top_rows:
            chart_df = pd.DataFrame(
                {
                    "target": [f"{row['entity_type']}#{row['entity_id']}" for row in top_rows],
                    "utility": [row["utility_score"] for row in top_rows],
                }
            ).set_index("target")
            st.subheader("Топ-цели по функции полезности")
            st.bar_chart(chart_df, width="stretch")

            summary_df = pd.DataFrame(
                [
                    {
                        "target": f"{row['entity_type']}#{row['entity_id']}",
                        "U_ij": round(row["utility_score"], 6),
                        "f_urgency": round(row["f_urgency"], 6),
                        "f_dist": round(row["f_dist"], 6),
                        "f_effort": round(row["f_effort"], 6),
                        "f_social": round(row["f_social"], 6),
                    }
                    for row in top_rows
                ]
            )
            st.dataframe(summary_df, width="stretch", hide_index=True)
        else:
            st.info("После pre-filter нет валидных кандидатов на текущем тике.")

    st.subheader("Пояснение выбора")
    reason = str(state.decision.get("reason", "Причина не передана агентом."))
    st.write(reason)


def _render_runbook() -> None:
    """Runbook внутри UI закрывает запрос «как запустить и как проверить» без переключения в README."""
    st.subheader("How to Run")

    st.markdown('<div class="runbook"><b>1. Запуск всей системы (рекомендуется)</b><br/>Из корня проекта выполните одну команду.</div>', unsafe_allow_html=True)
    st.code("python launcher.py", language="bash")

    st.markdown('<div class="runbook"><b>2. Запуск без автооткрытия браузера</b><br/>Удобно для отладки на удаленной машине.</div>', unsafe_allow_html=True)
    st.code("python launcher.py --no-browser", language="bash")

    st.markdown('<div class="runbook"><b>3. Проверка матмодели тестами</b><br/>Тесты покрывают U_ij, TTL, pre-filter и adapter/protocol слой.</div>', unsafe_allow_html=True)
    st.code("python -m unittest discover -s tests -p 'test_*.py' -v", language="bash")

    st.markdown('<div class="runbook"><b>4. Проверка в UI</b><br/>Выберите агента, откройте вкладку "Аудит 2.2" и убедитесь, что нет FAIL.</div>', unsafe_allow_html=True)


def _render_profile_controls() -> None:
    """Профили оставлены как экспериментальный слой поверх фиксированной структуры формул 2.2."""
    st.subheader("Профиль параметров модели")
    st.caption("Формулы не меняются. Можно менять только веса/константы и вернуться к baseline-профилю.")

    col_apply, col_reset = st.columns(2)
    with col_apply:
        if st.button("Применить профиль", key="profile_apply_button"):
            try:
                custom_profile = _collect_profile_from_form()
                _apply_profile(custom_profile)
                st.success("Пользовательский профиль применен.")
            except Exception as error:  # noqa: BLE001
                st.error(f"Не удалось применить профиль: {error}")

    with col_reset:
        if st.button("Вернуться к baseline", key="profile_reset_button"):
            try:
                _apply_profile(copy.deepcopy(st.session_state.baseline_profile))
                st.success("Восстановлен baseline профиль дипломной модели.")
            except Exception as error:  # noqa: BLE001
                st.error(f"Не удалось восстановить baseline: {error}")

    uploaded_profile = st.file_uploader("Импорт профиля (JSON)", type=["json"])
    if uploaded_profile is not None and st.button("Загрузить JSON-профиль", key="profile_import_button"):
        try:
            imported_profile = json.load(uploaded_profile)
            _apply_profile(imported_profile)
            st.success("Профиль из JSON успешно применен.")
        except Exception as error:  # noqa: BLE001
            st.error(f"Ошибка импорта профиля: {error}")

    export_payload = json.dumps(st.session_state.active_profile, ensure_ascii=False, indent=2)
    st.download_button(
        label="Экспорт текущего профиля",
        data=export_payload,
        file_name="utility_model_profile.json",
        mime="application/json",
    )

    st.markdown("#### Веса Utility-функции")
    for agent_key in PROFILE_AGENT_KEYS:
        with st.expander(f"{agent_key}", expanded=False):
            for weight_key in PROFILE_WEIGHT_KEYS:
                st.number_input(
                    label=f"{weight_key}",
                    min_value=0.0,
                    value=float(st.session_state[f"profile_{agent_key}_{weight_key}"]),
                    step=0.05,
                    format="%.6f",
                    key=f"profile_{agent_key}_{weight_key}",
                )

    st.markdown("#### Константы нормализации")
    for constant_key in PROFILE_CONSTANT_KEYS:
        min_value = 0.0 if constant_key == "c_switch" else 1e-9
        st.number_input(
            label=constant_key,
            min_value=min_value,
            value=float(st.session_state[f"profile_const_{constant_key}"]),
            step=0.01,
            format="%.6f",
            key=f"profile_const_{constant_key}",
        )


def _build_sample_dashboard_state(engine: SimulationEngine, selected_agent_id: int) -> DashboardState:
    """Снимок Sample-режима собирается атомарно из engine API для консистентного рендера панели."""
    matrix_rows = _normalize_matrix_rows(engine.get_agent_utility_matrix(selected_agent_id))
    decision_log = engine.get_agent_decision_log(selected_agent_id, limit=50)
    decision = decision_log[-1] if decision_log else {}

    return DashboardState(
        source_mode="Sample Mode",
        tick=engine.tick,
        agent_id=str(selected_agent_id),
        agent_type=engine.get_agent_state(selected_agent_id).type.value,
        agent_state=_serialize_agent_state(engine.get_agent_state(selected_agent_id)),
        decision={
            "action": decision.get("action"),
            "selected_target_id": decision.get("target_id"),
            "reason": decision.get("reason"),
        }
        if decision
        else {},
        utility_matrix=matrix_rows,
        visible_entities=engine.get_visible_entities_snapshot(),
        decision_log=decision_log,
        warnings=[],
    )


def _build_live_dashboard_state(snapshot_path: Path, selected_agent_id: str) -> Optional[DashboardState]:
    """Live-состояние читается из snapshot, чтобы UI не зависел от внутренностей сетевого цикла агентов."""
    snapshot = load_live_state_snapshot(snapshot_path)
    if snapshot is None:
        return None

    agents = snapshot.get("agents", {})
    if not isinstance(agents, dict) or not agents:
        return None

    record = agents.get(selected_agent_id, {})
    if not isinstance(record, dict):
        return None

    payload = record.get("payload", {})
    if not isinstance(payload, dict):
        return None

    agent_state = payload.get("agent_state", {})
    decision = payload.get("decision", {})
    matrix_rows = payload.get("utility_matrix", [])
    visible_entities = payload.get("visible_entities", [])
    decision_log = [decision] if isinstance(decision, dict) and decision else []
    warnings = record.get("warnings", []) if isinstance(record.get("warnings"), list) else []

    agent_type = str(agent_state.get("type", "UNKNOWN"))
    tick = _safe_int(record.get("tick"))

    return DashboardState(
        source_mode="Live Mode",
        tick=0 if tick is None else tick,
        agent_id=str(selected_agent_id),
        agent_type=agent_type,
        agent_state=agent_state if isinstance(agent_state, dict) else {},
        decision=decision if isinstance(decision, dict) else {},
        utility_matrix=_normalize_matrix_rows(matrix_rows if isinstance(matrix_rows, list) else []),
        visible_entities=visible_entities if isinstance(visible_entities, list) else [],
        decision_log=decision_log,
        warnings=warnings,
    )


def _sidebar_controls(mode: str, engine: Optional[SimulationEngine]) -> Dict[str, Any]:
    """Sidebar централизует выбор источника данных и управление тиками в Sample-режиме."""
    sidebar_state: Dict[str, Any] = {}

    with st.sidebar:
        st.header("Управление")
        st.caption("Режимы представления и источник данных")

        audience_mode = st.radio(
            "Профиль интерфейса",
            options=["Debug (для разработчика)", "Комиссия (презентация)"],
            index=0,
            key="audience_mode",
        )
        sidebar_state["audience_mode"] = audience_mode

        if mode == "Sample Mode" and engine is not None:
            st.markdown("---")
            st.subheader("Симуляция (Sample)")
            st.metric("Текущий тик", engine.tick)
            interval = st.number_input(
                "Интервал фонового шага (сек)",
                min_value=0.2,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="sample_interval",
            )

            if st.button("Запустить фон", key="sample_start_bg"):
                _start_background(engine, interval_seconds=float(interval))
            if st.button("Остановить фон", key="sample_stop_bg"):
                _stop_background()
            if st.button("Сделать 1 шаг", key="sample_step_once"):
                engine.step()
            if st.button("Обновить экран", key="sample_refresh"):
                st.rerun()

        if mode == "Live Mode":
            st.markdown("---")
            st.subheader("Источник данных (Live)")
            live_path = st.text_input("Путь к live_state.json", value=str(DEFAULT_LIVE_STATE_PATH), key="live_snapshot_path")
            sidebar_state["live_snapshot_path"] = live_path
            if st.button("Обновить снимок", key="live_refresh"):
                st.rerun()

    return sidebar_state


def _render_header(mode: str, audience_mode: str) -> None:
    """Верхний блок фиксирует комбинацию режимов, чтобы не перепутать Debug/Commission и Sample/Live."""
    st.markdown(
        f"""
        <div class="hero">
            <h2>RCRS Utility Dashboard</h2>
            <p>
                <span class="pill">data={mode}</span>
                <span class="pill">view={audience_mode}</span>
                <span class="pill">model=documents/2.2.md</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _stop_background_if_exists() -> None:
    """При переходе в Live останавливаем локальный поток, чтобы не было параллельного обновления демо-мира."""
    if "stop_event" not in st.session_state:
        return
    _stop_background()


def run_dashboard() -> None:
    """Единая точка входа: два источника данных и два интерфейсных режима поверх одной матмодели."""
    st.set_page_config(page_title="RCRS Utility Dashboard", layout="wide")
    _apply_visual_theme()

    with st.sidebar:
        mode = st.radio("Режим данных", options=["Sample Mode", "Live Mode"], index=0)

    if mode == "Sample Mode":
        engine = _ensure_engine()
    else:
        engine = None
        _stop_background_if_exists()

    sidebar_state = _sidebar_controls(mode=mode, engine=engine)
    audience_mode = sidebar_state.get("audience_mode", "Debug (для разработчика)")

    _render_header(mode=mode, audience_mode=audience_mode)

    if mode == "Sample Mode":
        assert engine is not None
        agent_options = engine.get_agent_ids()
        if not agent_options:
            st.warning("В симуляции нет агентов.")
            return

        selected_agent_id = st.selectbox("Выберите агента", options=agent_options, index=0, key="sample_agent_select")
        dashboard_state = _build_sample_dashboard_state(engine, int(selected_agent_id))
    else:
        snapshot_path_raw = sidebar_state.get("live_snapshot_path", str(DEFAULT_LIVE_STATE_PATH))
        snapshot_path = Path(str(snapshot_path_raw)).expanduser().resolve()
        snapshot = load_live_state_snapshot(snapshot_path)

        if snapshot is None:
            st.warning("Не удалось прочитать snapshot. Проверьте путь и запущены ли Python-агенты.")
            return

        agents = snapshot.get("agents", {})
        if not isinstance(agents, dict) or not agents:
            st.info("Snapshot загружен, но активных агентов пока нет.")
            st.json(snapshot, expanded=False)
            return

        live_agent_ids = sorted(agents.keys(), key=lambda value: int(value) if str(value).isdigit() else str(value))
        selected_agent_id = st.selectbox("Выберите live-агента", options=live_agent_ids, index=0, key="live_agent_select")

        dashboard_state = _build_live_dashboard_state(snapshot_path=snapshot_path, selected_agent_id=str(selected_agent_id))
        if dashboard_state is None:
            st.warning("Не удалось собрать состояние выбранного live-агента.")
            return

    audit_result = _evaluate_audit(dashboard_state)

    tab_names = ["Панель", "Аудит 2.2", "How to Run"]
    if mode == "Sample Mode":
        tab_names.append("Профиль матмодели")

    tabs = st.tabs(tab_names)

    with tabs[0]:
        if audience_mode.startswith("Debug"):
            _render_debug_panel(dashboard_state, audit_result)
        else:
            _render_commission_panel(dashboard_state, audit_result)

    with tabs[1]:
        _render_audit_tab(audit_result)

    with tabs[2]:
        _render_runbook()

    if mode == "Sample Mode":
        with tabs[3]:
            _render_profile_controls()


if __name__ == "__main__":
    run_dashboard()
