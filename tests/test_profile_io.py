from __future__ import annotations

import unittest

from module.config_profile import model_config_from_dict, model_config_to_dict
from module.sample_data import build_baseline_config, build_demo_scenario


class ProfileRoundtripTests(unittest.TestCase):
    """Проверяем, что профиль можно сериализовать/десериализовать без потери параметров."""

    def test_roundtrip(self) -> None:
        visible_entities, refuges = build_demo_scenario()
        baseline_config = build_baseline_config(visible_entities, refuges)

        payload = model_config_to_dict(baseline_config)
        restored_config = model_config_from_dict(payload)
        restored_payload = model_config_to_dict(restored_config)

        self.assertEqual(payload, restored_payload)


if __name__ == "__main__":
    unittest.main()
