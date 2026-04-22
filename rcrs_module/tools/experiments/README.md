# Стенд ablation-исследований

Автоматизированный запуск ядра RCRS, агентов и сбор результатов по
наборам конфигураций (вариация весов `w_c/w_d/w_e/w_n`, вкл/выкл радио).

## Состав пакета

| Файл | Назначение |
|---|---|
| `configs.py` | Описание пресетов (smoke, weight_sweep_fire, weight_sweep_all, heatmap, radio, full) |
| `runner.py` | Оркестратор одного прогона: правит `agent.cfg`, запускает `start.sh` + `launch_agents.sh`, ждёт `"Kernel has shut down"`, собирает логи |
| `metrics.py` | Парсер серверных логов (score, final_tick, civilians, buildings) и сводок `summary_*.json` (freeze_pct, tick_p95, utility_ms, switches) |
| `visualize.py` | `results.csv`, `results.md` и графики matplotlib: sweep, heatmap, top-N bars, perf vs score |
| `__main__.py` | CLI-обёртка |

## Зависимости

Обязательно: `pydantic>=2.0`, `networkx>=3.0`, `protobuf>=4.0` (уже в
`requirements.txt`).

Для графиков: `matplotlib`, `numpy`. Если не установлены, CSV/Markdown
всё равно формируются, диаграммы пропускаются.

```bash
pip install matplotlib numpy
```

## Быстрая проверка

Перед многочасовым прогоном рекомендуется прогнать `smoke` (2 запуска):

```bash
cd rcrs_module
python3 -m tools.experiments --map kobe --preset smoke --timeout-minutes 15
```

Результаты попадут в `log/experiments/smoke_kobe_<timestamp>/`:

```
log/experiments/smoke_kobe_20260421_180301/
├── smoke_baseline/
│   ├── agent.cfg              # копия конфига прогона
│   ├── agents_stdout.log
│   ├── kernel_stdout.log
│   ├── kernel_logs/           # /rcrs-server/logs/log/* — снапшот
│   ├── agents_logs/           # /log/* — снапшот summary_*.json и tick CSV
│   └── run_metrics.json
├── smoke_no_radio/
│   └── …
├── checkpoint.json            # промежуточный снимок (пишется после каждого прогона)
├── results.csv
├── results.md
├── sweep_*.png
├── heatmap_wc_wd.png          # если в пресете есть heatmap-прогоны
├── top_configs.png
└── perf_vs_score.png
```

## Пресеты

| Пресет | Что делает | Прогонов |
|---|---|---|
| `smoke` | Baseline + no-radio. Для отладки стенда. | 2 |
| `weight_sweep_fire` | Вариация каждого из четырёх весов только у FIRE_BRIGADE. | 20 |
| `weight_sweep_all` | Вариация `w_c` и `w_d` у всех трёх ролей. | 16 |
| `heatmap` | Двумерная сетка `(w_c, w_d)` для общего heatmap. | 19 |
| `radio` | Включено/выключено радио при дефолтных весах. | 2 |
| `full` | sweep_all + heatmap + radio. Несколько часов. | 37 |

## Полная ablation на карте Kobe

```bash
cd rcrs_module
python3 -m tools.experiments \
    --map kobe \
    --preset full \
    --runs-per-config 1 \
    --timeout-minutes 20 \
    --output-dir ../log/experiments/kobe_full
```

Один прогон Kobe на 270 тактов в headless-режиме обычно занимает
6–15 минут (зависит от железа и `think-time=1000`). Пресет `full`
из 37 конфигураций → 4–10 часов.

Если нужно повторить каждую конфигурацию для усреднения шума —
`--runs-per-config 3` (каждому RunSpec добавится суффикс `_rep1/rep2/rep3`).

## Что пишется в результирующий CSV

Поля `results.csv`:

```
run_name, role, axis, axis_value,
w_c_fire, w_d_fire, w_e_fire, w_n_fire,
w_c_amb, w_d_amb, w_e_amb, w_n_amb,
w_c_pol, w_d_pol, w_e_pol, w_n_pol,
radio_enabled,
score, final_tick,
freeze_pct, tick_p95_ms, tick_mean_ms,
utility_ms_mean, target_switches, stuck_detected,
ak_say_total, agents_count, wall_clock_s, error
```

## Замечания и типичные проблемы

- **Порт 7000 занят.** Если что-то осталось от прошлого прогона,
  `runner._kill_stale_agents()` перед каждым запуском выполняет
  `pkill -9 -f 'main.py --agent-type'`; ядро убивается через
  `killpg` своей группы. Если порт всё равно занят посторонним
  процессом — задайте `--kernel-port 7001` и поправьте kernel-конфиг.
- **`agent.cfg`.** Перед стартом стенд делает `agent.cfg.bak.experiments`
  и восстанавливает его по завершении (в т.ч. при Ctrl+C).
- **Score не парсится.** Формат вывода RCRS отличается между версиями;
  `metrics.SCORE_PATTERNS` содержит 4 regex-а. Если ядро пишет счёт
  в другом формате — достаточно добавить regex в этот список.
- **Таймаут.** По умолчанию `--timeout-minutes 20` — если ядро не
  завершилось за это время, прогон останавливается с `error="timeout"`,
  результаты всё равно парсятся. Для длинных карт увеличьте.
- **macOS: pgrep не находит процессы.** `_kill_stale_agents` ищет по
  паттерну `main.py --agent-type`; если вы переименовали точку входа,
  поправьте строку в `runner.py`.
