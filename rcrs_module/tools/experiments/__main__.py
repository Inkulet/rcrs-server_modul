"""CLI-оболочка для ablation-стенда.

Пример запуска:

    # Быстрая проверка (2 прогона):
    python3 -m tools.experiments --map kobe --preset smoke

    # Полная ablation (много прогонов, несколько часов):
    python3 -m tools.experiments --map kobe --preset full --timeout-minutes 25

    # Сетка (w_c, w_d) для построения heatmap:
    python3 -m tools.experiments --map kobe --preset heatmap

Запускать из каталога rcrs_module/.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .configs import PRESETS, RunSpec
from .metrics import RunMetrics
from .runner import EnvConfig, ExperimentRunner, detect_env
from .visualize import render_all


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RCRS ablation experiment runner")
    ap.add_argument("--map", default="kobe", help="Имя папки под maps/: kobe, berlin, test, …")
    ap.add_argument("--preset", default="smoke", choices=sorted(PRESETS.keys()))
    ap.add_argument("--runs-per-config", type=int, default=1, help="Повторить каждую конфигурацию N раз")
    ap.add_argument("--timeout-minutes", type=float, default=20.0)
    ap.add_argument("--kernel-host", default="127.0.0.1")
    ap.add_argument("--kernel-port", type=int, default=7000)
    ap.add_argument(
        "--output-dir",
        default="",
        help="Куда складывать результаты. По умолчанию <repo>/log/experiments/<timestamp>",
    )
    ap.add_argument(
        "--repo-root",
        default="",
        help="Корень проекта (по умолчанию определяется автоматически)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Запустить только первые N RunSpec из пресета")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def _resolve_repo_root(arg: str) -> Path:
    if arg:
        return Path(arg).resolve()
    # tools/experiments/__main__.py → ../../../
    here = Path(__file__).resolve()
    return here.parents[3]


def _restore_agent_cfg_backup(agent_cfg: Path, backup: Path) -> None:
    if backup.exists():
        try:
            shutil.copy(backup, agent_cfg)
        except OSError:
            pass


def _checkpoint(results: list[tuple[RunSpec, RunMetrics]], path: Path) -> None:
    data = [
        {"run": spec.name, "metrics": metrics.to_dict()}
        for spec, metrics in results
    ]
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def main(argv: list[str]) -> int:
    ns = _parse_args(argv)
    _setup_logging(ns.verbose)
    logger = logging.getLogger("experiments.cli")

    repo_root = _resolve_repo_root(ns.repo_root)
    logger.info("repo_root: %s", repo_root)

    env: EnvConfig = detect_env(repo_root, map_name=ns.map)
    env = env.__class__(
        repo_root=env.repo_root,
        server_dir=env.server_dir,
        module_dir=env.module_dir,
        agent_cfg_path=env.agent_cfg_path,
        launch_agents_sh=env.launch_agents_sh,
        kernel_scripts_dir=env.kernel_scripts_dir,
        kernel_log_dir=env.kernel_log_dir,
        map_dir=env.map_dir,
        config_dir=env.config_dir,
        agents_log_dir=env.agents_log_dir,
        kernel_host=ns.kernel_host,
        kernel_port=ns.kernel_port,
    )

    # Предусловия.
    missing = []
    for p in (env.server_dir, env.module_dir, env.map_dir, env.config_dir,
              env.launch_agents_sh, env.kernel_scripts_dir):
        if not p.exists():
            missing.append(str(p))
    if missing:
        logger.error("отсутствуют пути: %s", missing)
        return 2

    specs = PRESETS[ns.preset]()
    if ns.runs_per_config > 1:
        repeated: list[RunSpec] = []
        for spec in specs:
            for k in range(ns.runs_per_config):
                # Pydantic-like frozen dataclass — пересоздаём.
                rep_name = f"{spec.name}_rep{k+1}"
                repeated.append(
                    RunSpec(
                        name=rep_name,
                        description=spec.description,
                        weights_fire=spec.weights_fire,
                        weights_ambulance=spec.weights_ambulance,
                        weights_police=spec.weights_police,
                        radio_enabled=spec.radio_enabled,
                        ablation_role=spec.ablation_role,
                        ablation_axis=spec.ablation_axis,
                        ablation_value=spec.ablation_value,
                    )
                )
        specs = repeated

    if ns.limit > 0:
        specs = specs[: ns.limit]

    # Резервная копия текущего agent.cfg.
    cfg_backup = env.module_dir / "agent.cfg.bak.experiments"
    try:
        shutil.copy(env.agent_cfg_path, cfg_backup)
    except OSError:
        logger.warning("не смог сделать бекап agent.cfg")

    # Каталог результатов.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if ns.output_dir:
        out_root = Path(ns.output_dir).resolve()
    else:
        out_root = repo_root / "log" / "experiments" / f"{ns.preset}_{ns.map}_{ts}"

    logger.info("output_dir: %s", out_root)
    logger.info("запланировано прогонов: %d", len(specs))

    runner = ExperimentRunner(env, out_root, timeout_minutes=ns.timeout_minutes)
    results: list[tuple[RunSpec, RunMetrics]] = []

    try:
        for i, spec in enumerate(specs, 1):
            logger.info("[%d/%d] %s", i, len(specs), spec.name)
            metrics = runner.run(spec)
            results.append((spec, metrics))
            _checkpoint(results, out_root / "checkpoint.json")
    except KeyboardInterrupt:
        logger.warning("прервано пользователем, сохраняю уже полученные результаты")
    finally:
        _restore_agent_cfg_backup(env.agent_cfg_path, cfg_backup)

    # Таблица + отчёт + графики.
    render_all(results, out_root)
    logger.info("готово. Итог: %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
