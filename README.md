
## Состав репозитория

```
rcrs_module/       — Python-модуль агента (основной объект разработки)
    main.py        — точка входа, главный цикл симуляции
    src/
        network/   — TCP-клиент, разбор пакетов ядра
        world/     — модель мира (Pydantic), дорожный граф (NetworkX)
        decision/  — предварительная фильтрация задач, расчёт полезности
        action/    — навигация, выбор цели с гистерезисом
    tests/         — модульные тесты (pytest)
documents/         — диаграммы и разделы пояснительной записки
```

## Архитектура

Агент реализует строгий послойный конвейер принятия решений:

```
Ядро RCRS (TCP:7000)
        ↓
  network/client    — приём пакетов, разбор протобаф
        ↓
  world/cache       — обновление модели мира, дорожный граф
        ↓
  decision/filters  — предварительная фильтрация нерелевантных задач
        ↓
  decision/utility  — расчёт U = w_c·f_urgency − w_d·f_distance + w_e·f_effort − w_n·f_social
        ↓
  action/selection  — выбор цели с гистерезисом (порог C_switch)
        ↓
  network/client    — отправка команды (AKMove / AKRescue / AKLoad / AKExtinguish / AKClear)
```

## Требования

- Python 3.11 или новее
- pip (стандартный менеджер пакетов Python)
- Запущенный сервер RCRS (ядро на `localhost:7000`)

Проверить версию Python:

```bash
python3 --version
```

## Установка

```bash
# 1. Клонировать репозиторий
git clone <url-репозитория>
cd rcrs_diplom/rcrs_module

# 2. Создать виртуальное окружение (рекомендуется)
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3. Установить зависимости
pip install pydantic networkx pytest
```

## Запуск агента

Сначала должен быть запущен сервер RCRS. Затем, из каталога `rcrs_module/`:

```bash
# Пожарный агент (по умолчанию)
python3 main.py --agent-type FIRE_BRIGADE

# Бригада скорой помощи
python3 main.py --agent-type AMBULANCE_TEAM

# Полицейский агент
python3 main.py --agent-type POLICE_FORCE

# Подключение к нестандартному адресу ядра
python3 main.py --agent-type FIRE_BRIGADE --host 192.168.1.10 --port 7000
```

Все доступные параметры:

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--agent-type` | `FIRE_BRIGADE` | Тип агента: `FIRE_BRIGADE`, `AMBULANCE_TEAM`, `POLICE_FORCE` |
| `--host` | `127.0.0.1` | Адрес ядра RCRS |
| `--port` | `7000` | Порт ядра RCRS |
| `--name` | `diploma-agent` | Имя агента для регистрации в ядре |

## Запуск тестов

```bash
cd rcrs_module
python3 -m pytest tests/ -v
```
