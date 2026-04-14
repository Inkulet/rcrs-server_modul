#!/bin/bash
# Я запускаю все 93 агента для карты Kobe: 30 пожарных, 30 скорых, 30 полицейских + 3 центральных.
# Каждый агент — отдельный Python-процесс, подключающийся к ядру RCRS на порту 7000.
#
# Использование:
#   ./launch_agents.sh [HOST] [PORT]
#   По умолчанию: ./launch_agents.sh 127.0.0.1 7000
#
# -------------------------------------------------------------------
# УПРАВЛЕНИЕ ЛОГИРОВАНИЕМ
# -------------------------------------------------------------------
# Чтобы включить/отключить логи — отредактируй файл agent.cfg:
#
#   [logging]
#   enabled = true   ← поменяй на false, чтобы полностью отключить логи
#   level = INFO     ← DEBUG / INFO / WARNING / ERROR / CRITICAL
#
# -------------------------------------------------------------------
# Для остановки всех агентов: kill $(jobs -p) или Ctrl+C

set -e

HOST="${1:-127.0.0.1}"
PORT="${2:-7000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Запуск агентов RCRS на ${HOST}:${PORT} ==="

# Я запускаю 30 пожарных агентов
for i in $(seq 1 30); do
    python3 "${SCRIPT_DIR}/main.py" --agent-type FIRE_BRIGADE --host "$HOST" --port "$PORT" &
done
echo "Запущено 30 FIRE_BRIGADE"

# Я запускаю 30 агентов скорой помощи
for i in $(seq 1 30); do
    python3 "${SCRIPT_DIR}/main.py" --agent-type AMBULANCE_TEAM --host "$HOST" --port "$PORT" &
done
echo "Запущено 30 AMBULANCE_TEAM"

# Я запускаю 30 полицейских агентов
for i in $(seq 1 30); do
    python3 "${SCRIPT_DIR}/main.py" --agent-type POLICE_FORCE --host "$HOST" --port "$PORT" &
done
echo "Запущено 30 POLICE_FORCE"

# Я запускаю 3 центральных агента
python3 "${SCRIPT_DIR}/main.py" --agent-type FIRE_STATION --host "$HOST" --port "$PORT" &
echo "Запущен FIRE_STATION"

python3 "${SCRIPT_DIR}/main.py" --agent-type AMBULANCE_CENTRE --host "$HOST" --port "$PORT" &
echo "Запущен AMBULANCE_CENTRE"

python3 "${SCRIPT_DIR}/main.py" --agent-type POLICE_OFFICE --host "$HOST" --port "$PORT" &
echo "Запущен POLICE_OFFICE"

echo "=== Все 93 агента запущены. PID: $(jobs -p | wc -l) процессов ==="
echo "Для остановки: kill \$(jobs -p) или Ctrl+C"

# Я жду завершения всех фоновых процессов
wait
