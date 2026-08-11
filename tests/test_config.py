import unittest
from pathlib import Path

from cosmit.config import load_config


class ConfigTests(unittest.TestCase):
    def test_example_config_is_valid(self) -> None:
        config = load_config(Path("configs/cosmit.example.yaml"))
        self.assertEqual(set(config.frameworks), {"tensorflow", "pytorch"})
        self.assertEqual(
            [direction.slug for direction in config.directions],
            ["tensorflow-to-pytorch", "pytorch-to-tensorflow"],
        )


if __name__ == "__main__":
    unittest.main()
