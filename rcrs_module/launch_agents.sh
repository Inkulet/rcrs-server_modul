#!/bin/bash

set -e

HOST="${1:-127.0.0.1}"
PORT="${2:-7000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Запуск агентов RCRS на ${HOST}:${PORT} ==="

for i in $(seq 1 50); do
    python3 "${SCRIPT_DIR}/main.py" --agent-type FIRE_BRIGADE --host "$HOST" --port "$PORT" &
done
echo "Запущено 30 FIRE_BRIGADE"

for i in $(seq 1 50); do
    python3 "${SCRIPT_DIR}/main.py" --agent-type AMBULANCE_TEAM --host "$HOST" --port "$PORT" &
done
echo "Запущено 30 AMBULANCE_TEAM"

for i in $(seq 1 50); do
    python3 "${SCRIPT_DIR}/main.py" --agent-type POLICE_FORCE --host "$HOST" --port "$PORT" &
done
echo "Запущено 30 POLICE_FORCE"

python3 "${SCRIPT_DIR}/main.py" --agent-type FIRE_STATION --host "$HOST" --port "$PORT" &
echo "Запущен FIRE_STATION"

python3 "${SCRIPT_DIR}/main.py" --agent-type FIRE_STATION --host "$HOST" --port "$PORT" &
echo "Запущен FIRE_STATION"

python3 "${SCRIPT_DIR}/main.py" --agent-type AMBULANCE_CENTRE --host "$HOST" --port "$PORT" &
echo "Запущен AMBULANCE_CENTRE"

python3 "${SCRIPT_DIR}/main.py" --agent-type AMBULANCE_CENTRE --host "$HOST" --port "$PORT" &
echo "Запущен AMBULANCE_CENTRE"
python3 "${SCRIPT_DIR}/main.py" --agent-type AMBULANCE_CENTRE --host "$HOST" --port "$PORT" &
echo "Запущен AMBULANCE_CENTRE"
python3 "${SCRIPT_DIR}/main.py" --agent-type AMBULANCE_CENTRE --host "$HOST" --port "$PORT" &
echo "Запущен AMBULANCE_CENTRE"
python3 "${SCRIPT_DIR}/main.py" --agent-type AMBULANCE_CENTRE --host "$HOST" --port "$PORT" &
echo "Запущен AMBULANCE_CENTRE"

python3 "${SCRIPT_DIR}/main.py" --agent-type POLICE_OFFICE --host "$HOST" --port "$PORT" &
echo "Запущен POLICE_OFFICE"

echo "=== Все 93 агента запущены. PID: $(jobs -p | wc -l) процессов ==="
echo "Для остановки: kill \$(jobs -p) или Ctrl+C"

wait

# Пост-прогонная сводка. Скрипт сам читает agent.cfg; если
# [run_table] enabled=false — тихо выходит.
python3 "${SCRIPT_DIR}/tools/finalize_run.py" || true
