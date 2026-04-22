<div align="center">

# Многоагентная система принятия решений RCRS

_Модуль распределения задач между автономными агентами-спасателями
в среде RoboCup Rescue Simulation на основе многокритериальной
функции полезности_

</div>

---

## Оглавление

1. [Постановка задачи](#1-постановка-задачи)
2. [Архитектура](#2-архитектура)
3. [Математическая модель](#3-математическая-модель)
4. [Структура репозитория](#4-структура-репозитория)
5. [Быстрый старт](#5-быстрый-старт)
6. [Конфигурация](#6-конфигурация-agentcfg)
7. [Тестирование](#7-тестирование)
8. [Автоматизированные эксперименты](#8-автоматизированные-эксперименты)

---

## 1. Постановка задачи

Многоагентная система формализована как динамическая система
`Σ(t) = ⟨G, A(t), T(t)⟩`, функционирующая в дискретном времени
`t ∈ {0, 1, …, T_max}`:

- `G = (V, E)` — граф дорожной сети и объектов города;
- `A(t)` — вектор состояний агентов `S_j(t) = ⟨type_j, pos_j, state_j, resources_j⟩`;
- `T(t)` — вектор состояний активных задач `P_i(t) = ⟨loc_i, obs_i, ĥ est_i⟩`.

**Глобальная цель** — максимизация показателя эффективности
`Score = N_alive + α·√Σ Area_intact(b)` за фиксированное число тактов.

**Ограничения:**

| Критерий             | Значение                                                     |
| -------------------- | ------------------------------------------------------------ |
| Время отклика агента | **≤ 1000 мс/тик**, целевой бенчмарк **50–100 мс**            |
| Сложность вычислений | `O(M)` по множеству задач, без вложенных циклов `O(N·M)`     |
| Коммуникации         | гибридные: локальный расчёт + опциональные claim-сообщения по AKSay |

Подробная постановка: [`documents/md_files/2.2.md`](documents/md_files/2.2.md).

---

## 2. Архитектура

Агент реализован как строгий вычислительный конвейер из пяти слоёв,
каждый с единственной ответственностью. Кросс-слойное протекание
бизнес-логики исключено:

```
┌─────────────────────────────────────────────────────────┐
│                  Ядро RCRS (TCP)                        │
└─────────────────┬────────────────────────────▲──────────┘
                  │ KASense                    │ AKMove / AKRescue
                  │ (PerceptionPacket)         │ AKExtinguish / AKClear
                  ▼                            │ AKLoad / AKSay
        ┌────────────────────┐                 │
        │  src/network/      │  Восприятие    │
        │  client, codec     │  (Facade)      │
        └─────────┬──────────┘                 │
                  ▼                            │
        ┌────────────────────┐                 │
        │  src/world/        │  Состояние     │
        │  cache, entities   │  мира           │
        └─────────┬──────────┘  (Repository)   │
                  ▼                            │
        ┌────────────────────┐                 │
        │  src/decision/     │  Pre-filter     │
        │      filters/      │  (Template)     │
        └─────────┬──────────┘                 │
                  ▼                            │
        ┌────────────────────┐                 │
        │  src/decision/     │  Utility        │
        │      utility/      │  (Strategy)     │
        └─────────┬──────────┘                 │
                  ▼                            │
        ┌────────────────────┐                 │
        │  src/action/       │  Selection +    │
        │  selection,        │  Navigation +   │
        │  navigation,       │  Execution ─────┘
        │  executor          │  (A* на графе)
        └────────────────────┘
```

**Применённые паттерны проектирования:**

| Паттерн         | Класс / функция                                              |
| --------------- | ------------------------------------------------------------ |
| Facade          | `RCRSClient`                                                 |
| Strategy        | `compute_urgency` (`AgentType` → роль-специфичная формула)   |
| Template Method | `PreFilterDispatcher.filter_tasks` + `_is_relevant`          |
| Repository      | `WorldModel`                                                 |
| Value Object    | Pydantic-модели `AgentState`, `VisibleEntity`, `PerceptionPacket` |

Детали архитектуры: [`documents/md_files/2.4.md`](documents/md_files/2.4.md),
[`documents/md_files/2.6.md`](documents/md_files/2.6.md).
Диаграммы: [`documents/diagrams/`](documents/diagrams/).

---

## 3. Математическая модель

Для каждой задачи `t_i ∈ T(t)` агент `a_j` вычисляет полезность:

```
U_ij(t) = w_c · f_urgency(t_i) − ( w_d · f_dist(d_ij) + w_e · f_effort(t_i) + w_n · f_social(N_i) )
```

Все компоненты нормированы к `[0, 1]`. Агент выбирает

```
i* = argmax_{i ∈ T(t)} U_ij(t)
```

с защитой **гистерезисом** `C_switch = 0.1` — смена цели происходит
только при выигрыше полезности больше порога, что исключает
«дрожание» между близкими по приоритету задачами.

### Факторы

| Фактор       | Обозначение | Смысл                                                      | Вес по умолчанию |
| ------------ | ----------- | ---------------------------------------------------------- | ---------------- |
| Срочность    | `f_urgency` | TTL жертвы / стадия горения / близость к ключевым объектам | **w_c = 0.4**    |
| Дистанция    | `f_dist`    | Реальное расстояние по графу `/ MaxMapDistance`            | **w_d = 0.2**    |
| Трудоёмкость | `f_effort`  | Зависит от роли: buriedness / площадь / repair_cost        | **w_e = 0.2**    |
| Социальный   | `f_social`  | Плотность союзников-одноролевцев в радиусе `r`             | **w_n = 0.2**    |

### Ролевые формулы срочности

| Роль              | Формула `f_urgency`                                          |
| ----------------- | ------------------------------------------------------------ |
| **AmbulanceTeam** | по TTL жертвы: `TTL = HP / Damage`; ноль, если `TTL ≤ t_travel + t_work` |
| **FireBrigade**   | `Temperature / T_max · I(fieryness ∈ {1, 2, 3})`             |
| **PoliceForce**   | `1 / (min_dist(blockade, важные_объекты) + ε)`               |

Подробная выкладка и обоснования:
[`documents/md_files/2.2.md`](documents/md_files/2.2.md),
[`documents/md_files/factors_table.md`](documents/md_files/factors_table.md).

---

## 4. Структура репозитория

```
rcrs_diplom/
├── rcrs_module/                      Python-модуль агента
│   ├── main.py                       точка входа, CLI-парсер
│   ├── launch_agents.sh              скрипт запуска полной команды (93 процесса)
│   ├── agent.cfg                     runtime-конфигурация (INI)
│   ├── requirements.txt              pydantic ≥ 2, networkx ≥ 3, protobuf ≥ 4, pytest
│   │
│   ├── src/
│   │   ├── network/                  слой восприятия (TCP + Protobuf)
│   │   ├── world/                    состояние мира (Pydantic + NetworkX)
│   │   ├── decision/
│   │   │   ├── filters/              предварительная фильтрация задач
│   │   │   └── utility/              urgency / effort / distance / social
│   │   ├── action/                   выбор цели, навигация, исполнение
│   │   ├── agent/                    ролевые циклы (fire, ambulance, police, center)
│   │   ├── config.py                 константы и парсинг agent.cfg
│   │   └── metrics.py                сбор per-tick метрик
│   │
│   ├── tests/                        18 модулей unit-тестов (pytest)
│   │
│   └── tools/
│       ├── finalize_run.py           пост-прогонный агрегатор (runs.md)
│       └── experiments/              стенд автоматизированных ablation-прогонов
│
└── documents/                        пояснительная записка
    ├── md_files/                     главы 2.1, 2.2, 2.3, 2.4, 2.6
    ├── diagrams/                     UML: компоненты, классы, use-case, activity
    ├── math/                         математические выкладки, таблицы факторов
    └── Презентация/                  слайды к защите
```

---

## 5. Быстрый старт

### Требования

- Python **3.11+**
- Java **21** (для Java-ядра симулятора RCRS)
- Gradle (собирается из репозитория `rcrs-server`, в этом репо отсутствует)

### Установка Python-модуля

```bash
cd rcrs_module
python3 -m venv venv
source venv/bin/activate          # или venv\Scripts\activate на Windows
pip install -r requirements.txt
```

### Запуск одного агента

Ядро RCRS должно быть уже запущено и слушать на `host:port`
(для Kobe по умолчанию **:27931**, задаётся в `maps/kobe/config/common.cfg`):

```bash
python3 main.py --agent-type FIRE_BRIGADE --host 127.0.0.1 --port 27931
python3 main.py --agent-type AMBULANCE_TEAM
python3 main.py --agent-type POLICE_FORCE
python3 main.py --agent-type FIRE_STATION        # командный центр
python3 main.py --agent-type AMBULANCE_CENTRE
python3 main.py --agent-type POLICE_OFFICE
```

| Параметр       | По умолчанию    | Описание                      |
| -------------- | --------------- | ----------------------------- |
| `--agent-type` | `FIRE_BRIGADE`  | Один из шести типов агента    |
| `--host`       | `127.0.0.1`     | Адрес ядра симулятора         |
| `--port`       | `7000`          | TCP-порт ядра (Kobe: `27931`) |
| `--name`       | `diploma-agent` | Имя агента для регистрации    |

### Запуск всей команды

Скрипт спавнит 30 FIRE + 30 AMB + 30 POL + 5 центров (в сумме ~93 процесса)
и ждёт их завершения:

```bash
./launch_agents.sh 127.0.0.1 27931
```

Первые 90 агентов подключаются к слотам карты, лишние корректно отвергаются
ядром.

---

## 6. Конфигурация (`agent.cfg`)

Все экспериментальные параметры вынесены в INI-файл, изменение которого
**не требует перекомпиляции или переимпорта кода**:

```ini
[logging]
enabled   = true
level     = WARNING              # DEBUG | INFO | WARNING | ERROR
log_dir   = /абсолютный/путь     # куда писать логи и метрики

[metrics]
enabled           = true
budget_ms         = 100.0        # бюджет utility-фазы

[csv_metrics]
tick_csv_enabled      = true     # per-tick, per-agent замеры (CSV)
events_csv_enabled    = true     # события: target_change, stuck, freeze
summary_json_enabled  = true     # агрегированная сводка за сессию

[communication]
radio_enabled = true             # AKSay-координация (claim-сообщения)

[weights]                        # оверрайд весов (для ablation)
fire      =                      # w_c,w_d,w_e,w_n (пусто = дефолт)
ambulance =
police    =

[run_table]
enabled = true                   # автоматически дописывать runs.md
```

---

## 7. Тестирование

```bash
cd rcrs_module
python3 -m pytest tests/ -v

# запуск отдельного модуля
python3 -m pytest tests/test_aggregator.py -v

# запуск одного теста
python3 -m pytest tests/test_selection.py::TestHysteresis::test_switch_threshold -v
```

**Покрытие:** 18 модулей, ~270 тестов. Включают проверки формулы
полезности, ролевых функций срочности, гистерезиса, blacklist-а,
stuck-детектора, алгоритмов навигации и сериализации Pydantic-моделей.

---

## 8. Автоматизированные эксперименты

Для сравнительных исследований разработан стенд ablation-прогонов,
который:

1. Запускает Java-ядро и 90 Python-агентов.
2. По заданному пресету меняет веса `(w_c, w_d, w_e, w_n)` и флаг
   `radio_enabled` в `agent.cfg`.
3. Ждёт завершения симуляции, парсит `score` из `kernel.log`.
4. Агрегирует per-agent метрики из 90 `summary_*.json`.
5. Строит CSV-таблицу, Markdown-отчёт и графики matplotlib
   (sweep, heatmap, top-N bars, perf vs score).

```bash
cd rcrs_module
pip install -r tools/experiments/requirements.txt

# быстрая проверка (2 прогона, ~20 мин):
python3 -m tools.experiments --map kobe --preset smoke --kernel-port 27931

# одномерные вариации w_c/w_d/w_e/w_n у пожарных (20 прогонов):
python3 -m tools.experiments --map kobe --preset weight_sweep_fire

# полная ablation (37 прогонов, ~5 часов):
python3 -m tools.experiments --map kobe --preset full --timeout-minutes 15
```

| Пресет              | Описание                                             | Прогонов |
| ------------------- | ---------------------------------------------------- | -------- |
| `smoke`             | Baseline + no-radio — проверка стенда                | 2        |
| `weight_sweep_fire` | Вариация четырёх весов только у FIRE_BRIGADE         | 20       |
| `weight_sweep_all`  | Вариация `w_c, w_d` у всех трёх ролей                | 16       |
| `heatmap`           | Двумерная сетка `(w_c, w_d)` для heatmap             | 19       |
| `radio`             | Baseline с включённым / выключенным радио            | 2        |
| `full`              | Объединение `weight_sweep_all` + `heatmap` + `radio` | 37       |

Подробно: [`rcrs_module/tools/experiments/README.md`](rcrs_module/tools/experiments/README.md).

</div>
