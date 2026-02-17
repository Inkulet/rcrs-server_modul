from __future__ import annotations

from module.sample_data import build_demo_engine


if __name__ == "__main__":
    # Консольный режим оставлен для быстрой проверки логики без запуска Streamlit UI.
    engine = build_demo_engine()
    for _ in range(5):
        engine.step()

    for agent_id in engine.get_agent_ids():
        state = engine.get_agent_state(agent_id)
        print(f"agent={agent_id}, type={state.type.value}, state={state.state.value}, target={state.current_target_id}")
        rows = engine.get_agent_utility_matrix(agent_id)
        print(f"matrix_rows={len(rows)}")
        for row in rows:
            print(row)
