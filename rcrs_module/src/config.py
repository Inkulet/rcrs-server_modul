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


def setup_logging() -> None:
    if not LOG_ENABLED:
        logging.disable(logging.CRITICAL)
        return

    level: int = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("agent_debug.log", mode="a"),
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

# Объём воды, выбрасываемой за один такт AKExtinguish (мл).
MAX_WATER_DISCHARGE: int = 3_000

# Дальность тушения пожарным (мм).
FIRE_EXTINGUISH_MAX_DISTANCE: float = 30_000.0

# Дальность расчистки полицейским (мм).
POLICE_CLEAR_MAX_DISTANCE: float = 10_000.0

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
CLAIMED_TARGET_PENALTY: float = 0.3

# Порог гистерезиса при переключении цели.
C_SWITCH: float = 0.1

# ---------------------------------------------------------------------------
# Параметры застревания / чёрного списка
# ---------------------------------------------------------------------------

# Агент считается застрявшим, если N тактов не меняет узел графа рядом с целью.
STUCK_TICKS: int = 8

# Цель, у которой застрял агент, блокируется на это количество тактов.
STUCK_BLACKLIST_TICKS: int = 60

# Цель без графового пути блокируется на меньший срок — путь может появиться.
UNREACHABLE_BLACKLIST_TICKS: int = 30

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
    "setup_logging",
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
    # Исследование
    "RANDOM_WALK_LENGTH",
    "EXPLORATION_PATH_LENGTH",
    "EXPLORATION_SEED_PRIME",
    "EXPLORATION_MAX_TICKS",
    # Координация
    "CLAIMED_TARGET_PENALTY",
    "C_SWITCH",
    # Застревание
    "STUCK_TICKS",
    "STUCK_BLACKLIST_TICKS",
    "UNREACHABLE_BLACKLIST_TICKS",
    # Цикл
    "TICK_BUDGET_SECONDS",
    "LOG_DIAG_PERIOD",
    "NO_REFUGE_MAX_RETRIES",
]
