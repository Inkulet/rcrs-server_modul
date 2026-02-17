#  Программный модуль

## 1. Назначение

## 2. Научная основа и инварианты

## 3. Архитектура

- `module/calculator.py` - реализация формализации полезности 2.2;
- `module/strategy.py` - Strategy-паттерн выбора цели;
- `module/agents.py` c `BaseAgent` и специализированные агенты;
- `module/network/` - интеграция с kernel:
  - `rcrs_client.py` (TCP клиент),
  - `world_model.py` (локальная модель мира),
  - `adapter.py` (Adapter Pattern: server -> model schema),
  - `protocol.py` (protobuf framing + AK/KA пакеты),
  - `agent_runtime.py` (Receive->Parse->Adapt->Think->Act),
  - `snapshot.py` (live snapshot в JSON);
- `module/ui/dashboard.py` - Streamlit UI (`Sample Mode` и `Live Mode`);
- `module/data/live_state.json` - shared state между агентами и UI.

## 4. Структура проекта

```text
rcrs-server_modul/
├── app.py
├── README.md
├── .gitignore
├── documents/
│   ├── 2.1.md
│   └── 2.2.md
├── module/
│   ├── calculator.py
│   ├── strategy.py
│   ├── agents.py
│   ├── data_models.py
│   ├── config.py
│   ├── config_profile.py
│   ├── sample_data.py
│   ├── simulation.py
│   ├── main.py
│   ├── main_agent.py
│   ├── run_agents.py
│   ├── requirements.txt
│   ├── data/
│   │   └── live_state.json
│   ├── network/
│   │   ├── protocol.py
│   │   ├── rcrs_client.py
│   │   ├── world_model.py
│   │   ├── adapter.py
│   │   ├── runtime_config.py
│   │   ├── agent_runtime.py
│   │   └── snapshot.py
│   └── ui/
│       └── dashboard.py
├── tests/
│   ├── test_math_model.py
│   ├── test_profile_io.py
│   ├── test_network_adapter.py
│   └── test_network_protocol.py
└── rcrs-server/   # локально, исключено из git
```

## 5. How To Run

### 5.0. Быстрый запуск (одна команда)

```bash
source venv/bin/activate
python launcher.py
```

`launcher.py` автоматически:

- запускает `rcrs-server`;
- читает `kernel.host/kernel.port` из карты и ждёт готовности фактического endpoint;
- запускает Python-агентов;
- запускает Streamlit UI;
- открывает вкладку браузера с UI.

Остановка всей системы: `Ctrl+C` в терминале launcher.

Если нужно отключить автооткрытие вкладки:

```bash
python launcher.py --no-browser
```

Логи RCRS для каждого запуска launcher сохраняются в отдельной папке:

- `.../rcrs-server_modul/rcrs-server/logs/launcher-YYYYMMDD-HHMMSS`

### 5.1. Предусловия
- Тестированно на MacBook Pro: Apple M3 Pro, macOS 26.2 (25C56)
- Python 3.11+;
- Java/JDK (для `rcrs-server`);
- локальная копия `rcrs-server`, содержащая:
  - `/scripts/platforms/python/URN.py`,
  - `/scripts/platforms/python/RCRSProto_pb2.py`.

Если `rcrs-server` расположен не в `rcrs-server_modul/rcrs-server`, укажите путь явно:

```bash
export RCRS_PLATFORM_PYTHON_DIR="/absolute/path/to/rcrs-server/scripts/platforms/python"
```

### 5.2. Установка Python-окружения

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r module/requirements.txt
```

### 5.3. Запуск RoboCup Rescue Server (kernel + simulators)

Терминал 1:

```bash
cd .../rcrs-server_modul/rcrs-server
./gradlew completeBuild
cd scripts
./start.sh -m ../maps/test/map -c ../maps/test/config
```

Порт kernel берется из конфигов карты (`maps/.../config/kernel.cfg` + include-цепочка).
В типовом тестовом наборе это часто `localhost:27931`.

### 5.4. Запуск Python-агентов

Терминал 2 (все три агента сразу):

```bash
cd .../rcrs-server_modul
source venv/bin/activate
python -m module.run_agents --host 127.0.0.1 --port <kernel_port_from_config>
```

Альтернативно, по одному:

```bash
python -m module.main_agent --host 127.0.0.1 --port <kernel_port_from_config> --agent-type FIRE_BRIGADE --request-id 1
python -m module.main_agent --host 127.0.0.1 --port <kernel_port_from_config> --agent-type AMBULANCE_TEAM --request-id 2
python -m module.main_agent --host 127.0.0.1 --port <kernel_port_from_config> --agent-type POLICE_FORCE --request-id 3
```

Каждый тик агенты сохраняют состояние в:

- `module/data/live_state.json`

Важно: порт должен совпадать с `kernel.port` из конфига карты.
Если запускаете через `launcher.py`, порт подставляется автоматически.

### 5.5. Запуск UI

Терминал 3:

```bash
cd .../rcrs-server_modul
source venv/bin/activate
streamlit run app.py
```

В UI:

1. Выбрать `Live Mode` в боковой панели.
2. Убедиться, что путь к snapshot: `module/data/live_state.json`.
3. Выбрать агента и анализировать `utility_matrix`, `decision`, `warnings`.

### 5.6. Troubleshooting

- `ModuleNotFoundError: No module named 'module'`:
  - запускайте как модуль: `python -m module.main_agent ...` или `python -m module.run_agents`.
- `Не удалось загрузить URN/RCRSProto_pb2`:
  - проверьте `rcrs-server/scripts/platforms/python`;
  - либо задайте `RCRS_PLATFORM_PYTHON_DIR`.
- Пустой `Live Mode` в UI:
  - убедитесь, что агенты запущены и обновляют `module/data/live_state.json`;
  - проверьте путь к snapshot в боковой панели UI.

## 6. Режимы UI

- Источник данных:
  - `Sample Mode` — локальный демонстрационный сценарий и профили параметров.
  - `Live Mode` — чтение snapshot от сетевых агентов, подключенных к kernel.
- Профиль интерфейса:
  - `Debug (для разработчика)` — полный trace: матрица, raw/computed сенсоры, decision log, warnings.
  - `Комиссия (презентация)` — компактный слой: KPI, топ-цели, объяснение выбора и статус соответствия 2.2.
- Вкладка `Аудит 2.2`:
  - проверяет диапазоны факторов `[0,1]`;
  - сверяет `U_ij` с формулой weighted-sum (в Sample Mode, где профиль известен);
  - проверяет pre-filter правила раздела 4.

## 7. Кастомизация параметров модели

Формулы остаются неизменными. Допускается изменение численных параметров:

- через профиль в `Sample Mode` (import/export JSON),
- через `--profile-path` при запуске `module.main_agent`.

Это позволяет проводить вычислительные эксперименты без модификации базовой математической постановки.

## 8. Тестирование

```bash
cd .../rcrs-server_modul
source venv/bin/activate
python -m unittest discover -s tests -p "test_*.py" -v
```

Покрыты:

- корректность `U_ij`, `TTL`, pre-filter;
- self-reference защита для police;
- protobuf framing/parsing;
- адаптация server данных в модель 2.2;
- сериализация профилей.

## 9. Git-правила

`.gitignore` исключает:

- `venv/`,
- `rcrs-server/`,
- кэши, логи, артефакты IDE/OS.
