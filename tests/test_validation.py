import tempfile
import unittest
from pathlib import Path

from cosmit.domain.models import ComponentId, MigrationCandidate, MigrationStatus
from cosmit.engine.validation import validate_candidate


class ValidationTests(unittest.TestCase):
    def test_static_feedback_repairs_source_and_missing_target_imports(self) -> None:
        candidate = MigrationCandidate(
            candidate_id="candidate-1",
            knowledge_id="knowledge-1",
            source_framework="tensorflow",
            target_component=ComponentId("pytorch", "torch.relu"),
            target_api="torch.relu",
            code=(
                "import tensorflow as tf\n\n"
                "def test_relu():\n"
                "    actual = torch.relu(torch.tensor([-1.0]))\n"
                "    torch.testing.assert_close(actual, torch.tensor([0.0]))\n"
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            repaired, record = validate_candidate(
                candidate,
                Path(directory) / "candidate.py",
                dynamic=False,
                timeout_seconds=5,
                max_repair_rounds=2,
            )
        self.assertEqual(repaired.status, MigrationStatus.STATIC_VALIDATED)
        self.assertIn("import torch", repaired.code)
        self.assertNotIn("tensorflow", repaired.code)
        self.assertTrue(record["repair_history"])


if __name__ == "__main__":
    unittest.main()
