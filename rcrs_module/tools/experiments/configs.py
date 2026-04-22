"""Описание наборов (пресетов) ablation-экспериментов.

Каждый пресет возвращает список RunSpec — план запусков для ExperimentRunner.
RunSpec содержит веса для трёх ролей; None означает «взять дефолт агента».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


Weights = tuple[float, float, float, float]  # (w_c, w_d, w_e, w_n)


DEFAULT_FIRE: Weights = (0.4, 0.2, 0.2, 0.2)
DEFAULT_AMBULANCE: Weights = (0.4, 0.2, 0.2, 0.2)
DEFAULT_POLICE: Weights = (0.4, 0.2, 0.2, 0.2)


@dataclass(frozen=True)
class RunSpec:
    name: str
    description: str = ""
    weights_fire: Optional[Weights] = None
    weights_ambulance: Optional[Weights] = None
    weights_police: Optional[Weights] = None
    radio_enabled: bool = True
    # Роль, по которой «едет» ablation (для группировки на графиках).
    ablation_role: str = "baseline"
    # Имя варьируемого веса (для графиков). Пример: "w_c".
    ablation_axis: str = ""
    ablation_value: float = 0.0


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _normalize(w: Weights) -> Weights:
    # Нормализация до суммы 1.0, чтобы Pydantic-проверка в агрегаторе не падала.
    total = sum(w)
    if total <= 0:
        return DEFAULT_FIRE
    return tuple(round(v / total, 6) for v in w)  # type: ignore[return-value]


def _sweep_axis(base: Weights, axis_idx: int, values: list[float]) -> list[tuple[Weights, float]]:
    # Вариация одного веса с нормализацией остальных трёх.
    out: list[tuple[Weights, float]] = []
    for v in values:
        rest = [w for i, w in enumerate(base) if i != axis_idx]
        rest_sum = sum(rest)
        if rest_sum <= 0:
            continue
        scale = (1.0 - v) / rest_sum
        w = list(base)
        w[axis_idx] = v
        for j in range(4):
            if j != axis_idx:
                w[j] = round(base[j] * scale, 6)
        out.append((tuple(w), v))  # type: ignore[arg-type]
    return out


# ---------------------------------------------------------------------------
# Пресеты
# ---------------------------------------------------------------------------

def preset_smoke() -> list[RunSpec]:
    # Два прогона: baseline и одна крайность — быстрая проверка что стенд работает.
    return [
        RunSpec(
            name="smoke_baseline",
            description="Дефолтные веса (0.4, 0.2, 0.2, 0.2)",
            ablation_role="baseline",
        ),
        RunSpec(
            name="smoke_no_radio",
            description="Baseline с отключённой радио-координацией",
            radio_enabled=False,
            ablation_role="no_radio",
        ),
    ]


def preset_weight_sweep_fire() -> list[RunSpec]:
    # Одномерная вариация каждого веса только для пожарных (FIRE_BRIGADE).
    runs: list[RunSpec] = [
        RunSpec(
            name="fire_baseline",
            description="Baseline fire",
            weights_fire=DEFAULT_FIRE,
            ablation_role="fire",
            ablation_axis="baseline",
        ),
    ]
    axis_map = {0: "w_c", 1: "w_d", 2: "w_e", 3: "w_n"}
    values = [0.1, 0.25, 0.4, 0.55, 0.7]
    for idx, axis in axis_map.items():
        for w, v in _sweep_axis(DEFAULT_FIRE, idx, values):
            if abs(v - DEFAULT_FIRE[idx]) < 1e-6:
                continue
            runs.append(
                RunSpec(
                    name=f"fire_{axis}_{v:.2f}",
                    description=f"fire: {axis}={v:.2f} (остальные нормализованы)",
                    weights_fire=w,
                    ablation_role="fire",
                    ablation_axis=axis,
                    ablation_value=v,
                )
            )
    return runs


def preset_weight_sweep_all() -> list[RunSpec]:
    # Полная ablation для трёх ролей: вариация w_c и w_d (важнейшие).
    runs: list[RunSpec] = [
        RunSpec(
            name="all_baseline",
            description="Baseline все роли",
            ablation_role="all",
            ablation_axis="baseline",
        ),
    ]
    for role, base in (
        ("fire", DEFAULT_FIRE),
        ("ambulance", DEFAULT_AMBULANCE),
        ("police", DEFAULT_POLICE),
    ):
        for axis_idx, axis in ((0, "w_c"), (1, "w_d")):
            for w, v in _sweep_axis(base, axis_idx, [0.15, 0.4, 0.65]):
                if abs(v - base[axis_idx]) < 1e-6:
                    continue
                spec = RunSpec(
                    name=f"{role}_{axis}_{v:.2f}",
                    description=f"{role}: {axis}={v:.2f}",
                    weights_fire=w if role == "fire" else None,
                    weights_ambulance=w if role == "ambulance" else None,
                    weights_police=w if role == "police" else None,
                    ablation_role=role,
                    ablation_axis=axis,
                    ablation_value=v,
                )
                runs.append(spec)
    return runs


def preset_heatmap_wc_wd() -> list[RunSpec]:
    # Двумерная сетка (w_c, w_d) при фиксированных w_e=w_n=0.15
    # (для генерации heatmap).
    runs: list[RunSpec] = []
    w_c_values = [0.2, 0.3, 0.4, 0.5, 0.6]
    w_d_values = [0.1, 0.2, 0.3, 0.4]
    for wc in w_c_values:
        for wd in w_d_values:
            rest = 1.0 - wc - wd
            if rest <= 0:
                continue
            we = wn = round(rest / 2.0, 6)
            w = (round(wc, 6), round(wd, 6), we, wn)
            runs.append(
                RunSpec(
                    name=f"hm_wc{wc:.2f}_wd{wd:.2f}",
                    description=f"heatmap w_c={wc}, w_d={wd}",
                    weights_fire=w,
                    weights_ambulance=w,
                    weights_police=w,
                    ablation_role="heatmap",
                    ablation_axis="wc_wd",
                    ablation_value=wc * 10 + wd,
                )
            )
    return runs


def preset_radio_ablation() -> list[RunSpec]:
    # Проверка влияния радио-координации при одинаковых весах.
    return [
        RunSpec(
            name="radio_on_default",
            description="Baseline с радио (AKSay claim-ы включены)",
            radio_enabled=True,
            ablation_role="radio",
            ablation_axis="radio",
            ablation_value=1.0,
        ),
        RunSpec(
            name="radio_off_default",
            description="Baseline без радио (чистая неявная координация)",
            radio_enabled=False,
            ablation_role="radio",
            ablation_axis="radio",
            ablation_value=0.0,
        ),
    ]


def preset_full() -> list[RunSpec]:
    # Полный ablation: sweep + heatmap + radio ablation. Много прогонов.
    seen = set()
    runs: list[RunSpec] = []
    for block in (
        preset_weight_sweep_all(),
        preset_heatmap_wc_wd(),
        preset_radio_ablation(),
    ):
        for r in block:
            if r.name in seen:
                continue
            seen.add(r.name)
            runs.append(r)
    return runs


# ---------------------------------------------------------------------------
# Публичный реестр пресетов
# ---------------------------------------------------------------------------

PRESETS: dict[str, Callable[[], list[RunSpec]]] = {
    "smoke": preset_smoke,
    "weight_sweep_fire": preset_weight_sweep_fire,
    "weight_sweep_all": preset_weight_sweep_all,
    "heatmap": preset_heatmap_wc_wd,
    "radio": preset_radio_ablation,
    "full": preset_full,
}
