from __future__ import annotations

# Я централизую все константы и настройку логирования здесь.
# Каждый модуль импортирует нужные значения отсюда, чтобы избежать магических
# чисел и дублирования констант по всей кодовой базе.

import configparser
import logging
import logging.handlers
from pathlib import Path

_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "agent.cfg"
_cfg = configparser.ConfigParser()
_cfg.read(_CONFIG_PATH)

LOG_ENABLED: bool = _cfg.getboolean("logging", "enabled", fallback=True)
LOG_LEVEL: str    = _cfg.get("logging", "level", fallback="INFO").upper()

# Я храню все логи/метрики в одном каталоге (по умолчанию <project_root>/log).
# Путь нормализую в абсолютный и создаю каталог при setup_logging, чтобы
# последующие writer-ы не заботились об отсутствии директории.
_PROJECT_ROOT: Path = _CONFIG_PATH.resolve().parent.parent
_raw_log_dir: str = _cfg.get("logging", "log_dir", fallback=str(_PROJECT_ROOT / "log"))
LOG_DIR: Path = Path(_raw_log_dir).expanduser()
if not LOG_DIR.is_absolute():
    LOG_DIR = (_CONFIG_PATH.resolve().parent / LOG_DIR).resolve()

# ---------------------------------------------------------------------------
# Параметры сбора метрик
# ---------------------------------------------------------------------------

METRICS_ENABLED: bool = _cfg.getboolean("metrics", "enabled", fallback=True)
METRICS_BUDGET_MS: float = _cfg.getfloat("metrics", "budget_ms", fallback=100.0)
METRICS_REPORT_PERIOD: int = _cfg.getint("metrics", "report_period", fallback=25)
METRICS_FILE: str = _cfg.get("metrics", "file", fallback="time.md")

# Детализированные замеры (per-tick CSV, per-event CSV, session summary JSON).
# Каждый тумблер независим, все файлы кладутся в LOG_DIR.
TICK_CSV_ENABLED: bool = _cfg.getboolean("csv_metrics", "tick_csv_enabled", fallback=False)
EVENTS_CSV_ENABLED: bool = _cfg.getboolean("csv_metrics", "events_csv_enabled", fallback=False)
SUMMARY_JSON_ENABLED: bool = _cfg.getboolean("csv_metrics", "summary_json_enabled", fallback=False)


def setup_logging() -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    if not LOG_ENABLED:
        logging.disable(logging.CRITICAL)
        return

    import os
    level: int = getattr(logging, LOG_LEVEL, logging.INFO)
    debug_log_path = LOG_DIR / f"agent_debug_{os.getpid()}.log"
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(debug_log_path), mode="a"),
        ],
    )


# ---------------------------------------------------------------------------
# Параметры подключения к ядру
# ---------------------------------------------------------------------------

KERNEL_HOST: str        = "127.0.0.1"
KERNEL_PORT: int        = 7000
KERNEL_TIMEOUT: float   = 30.0
MAX_CONNECT_RETRIES: int = 20

# Максимальный размер одного TCP-фрейма от ядра (байт).
MAX_FRAME_SIZE: int = 10_000_000

# ---------------------------------------------------------------------------
# Параметры физической модели
# ---------------------------------------------------------------------------

# Средняя скорость агента (мм/такт). Используется для оценки t_travel.
AVERAGE_SPEED: float = 70_000.0

# Радиус социального взаимодействия (мм): агенты одного типа в этом радиусе
# от цели снижают её приоритет (f_social).
SOCIAL_RADIUS: float = 30_000.0

# Максимум воды за один такт AKExtinguish (resq-fire.max_extinguish_power_sum).
MAX_WATER_DISCHARGE: int = 15_000

# Дальность тушения пожарным (мм, resq-fire.water_distance = 150000).
FIRE_EXTINGUISH_MAX_DISTANCE: float = 150_000.0

# Дальность расчистки полицейским (мм).
POLICE_CLEAR_MAX_DISTANCE: float = 10_000.0

# ---------------------------------------------------------------------------
# Параметры обхода завалов (планировщик маршрутов)
# ---------------------------------------------------------------------------

# Базовый мультипликатор веса ребра, инцидентного узлу с блокадой.
BLOCKADE_EDGE_PENALTY: float = 8.0

# Делитель repair_cost для плавного роста штрафа (чем больше ремонт — тем хуже путь).
BLOCKADE_REPAIR_COST_DIVISOR: float = 2_000.0

# Максимальный множитель штрафа (верхняя граница, чтобы путь не становился ∞).
BLOCKADE_PENALTY_MAX: float = 50.0

# ---------------------------------------------------------------------------
# Параметры исследования карты
# ---------------------------------------------------------------------------

# Fallback: количество шагов случайного блуждания (резерв если нет цели).
RANDOM_WALK_LENGTH: int = 50

# Длина маршрута исследования, передаваемого ядру за один такт.
EXPLORATION_PATH_LENGTH: int = 40

# Простое число для разброса RNG-сидов разных агентов.
EXPLORATION_SEED_PRIME: int = 1_000_003

# Количество тактов до смены цели исследования при отсутствии прогресса.
EXPLORATION_MAX_TICKS: int = 80

# ---------------------------------------------------------------------------
# Параметры координации и гистерезиса
# ---------------------------------------------------------------------------

# Штраф полезности для цели, заявленной другим агентом через AKSay.
# Применяется аддитивно: U' = U - CLAIMED_TARGET_PENALTY.
CLAIMED_TARGET_PENALTY: float = 0.3

# Дополнительный штраф для same-role claim. Нужен для простого ownership:
# если цель уже занята агентом той же роли, я почти всегда уступаю её и не
# дублирую работу. Legacy-claims без роли получают только базовый штраф выше.
SAME_ROLE_CLAIM_PENALTY: float = 0.7

# Шаг 10: сколько тактов помнить услышанную цель союзника (FIRE_BRIGADE),
# чтобы разводить пожарных по разным очагам. Заменяет K-means
# кластеризацию из ADF на простую «память» о занятых целях.
RECENT_ALLY_TARGET_TICKS: int = 15

# Шаг 10: дополнительный штраф (поверх CLAIMED_TARGET_PENALTY),
# применяемый к целям из recent_allies_targets для FIRE_BRIGADE.
ALLY_TARGET_LONGTERM_PENALTY: float = 0.2

# Порог гистерезиса при переключении цели.
C_SWITCH: float = 0.1

# ---------------------------------------------------------------------------
# Параметры f_urgency для полиции
# ---------------------------------------------------------------------------

# Масштаб нормировки расстояния «завал → ближайшая важная цель/убежище»
# перед подстановкой в формулу 1/(d + ε). Без нормировки d в миллиметрах
# (10^6) давал f_urgency ≈ 1e-6, после clamp — всегда 0, и полиция ничего
# не выбирала. Делю на 1000 мм (1 м) → d' ∈ [1..100] для типичных расстояний
# [1м..100м], что даёт осмысленное f_urgency ∈ [0.01..1.0].
POLICE_URGENCY_DISTANCE_SCALE: float = 1_000.0

# ---------------------------------------------------------------------------
# Параметры застревания / чёрного списка
# ---------------------------------------------------------------------------

# Агент считается застрявшим, если N тактов не меняет узел графа рядом с целью.
STUCK_TICKS: int = 8

# Цель, у которой застрял агент, блокируется на это количество тактов.
STUCK_BLACKLIST_TICKS: int = 60

# Цель без графового пути блокируется на меньший срок — путь может появиться.
UNREACHABLE_BLACKLIST_TICKS: int = 30

# Через сколько тактов после попадания завала в skip-список полицейский
# снова попробует его расчистить (иначе завалы, попавшие туда ошибочно,
# остались бы заблокированы навсегда).
CLEAR_SKIP_EXPIRE_TICKS: int = 30

# ---------------------------------------------------------------------------
# Служебные параметры главного цикла
# ---------------------------------------------------------------------------

# Бюджет времени одного такта (с). При превышении — WARNING в лог.
TICK_BUDGET_SECONDS: float = 0.1

# Период вывода диагностических сообщений (такты).
LOG_DIAG_PERIOD: int = 10

# Максимальное количество неудачных попыток найти убежище подряд.
NO_REFUGE_MAX_RETRIES: int = 5


__all__ = [
    # Логирование
    "LOG_ENABLED",
    "LOG_LEVEL",
    "LOG_DIR",
    "setup_logging",
    # Метрики
    "METRICS_ENABLED",
    "METRICS_BUDGET_MS",
    "METRICS_REPORT_PERIOD",
    "METRICS_FILE",
    "TICK_CSV_ENABLED",
    "EVENTS_CSV_ENABLED",
    "SUMMARY_JSON_ENABLED",
    # Подключение
    "KERNEL_HOST",
    "KERNEL_PORT",
    "KERNEL_TIMEOUT",
    "MAX_CONNECT_RETRIES",
    "MAX_FRAME_SIZE",
    # Физика
    "AVERAGE_SPEED",
    "SOCIAL_RADIUS",
    "MAX_WATER_DISCHARGE",
    "FIRE_EXTINGUISH_MAX_DISTANCE",
    "POLICE_CLEAR_MAX_DISTANCE",
    "BLOCKADE_EDGE_PENALTY",
    "BLOCKADE_REPAIR_COST_DIVISOR",
    "BLOCKADE_PENALTY_MAX",
    # Исследование
    "RANDOM_WALK_LENGTH",
    "EXPLORATION_PATH_LENGTH",
    "EXPLORATION_SEED_PRIME",
    "EXPLORATION_MAX_TICKS",
    # Координация
    "CLAIMED_TARGET_PENALTY",
    "SAME_ROLE_CLAIM_PENALTY",
    "RECENT_ALLY_TARGET_TICKS",
    "ALLY_TARGET_LONGTERM_PENALTY",
    "C_SWITCH",
    # Застревание
    "STUCK_TICKS",
    "STUCK_BLACKLIST_TICKS",
    "UNREACHABLE_BLACKLIST_TICKS",
    "CLEAR_SKIP_EXPIRE_TICKS",
    # Цикл
    "TICK_BUDGET_SECONDS",
    "LOG_DIAG_PERIOD",
    "NO_REFUGE_MAX_RETRIES",
]
