from __future__ import annotations

from module.ui.dashboard import run_dashboard


if __name__ == "__main__":
    # Entry-point поддерживает два режима в UI: Sample Mode и Live Mode (чтение module/data/live_state.json).
    run_dashboard()
