import tempfile
import unittest
from pathlib import Path

from cosmit.config import load_config
from cosmit.io import read_jsonl
from cosmit.runner import run_pipeline


class BidirectionalPipelineTests(unittest.TestCase):
    def test_example_runs_all_six_stages_in_both_directions(self) -> None:
        config = load_config(Path("configs/cosmit.example.yaml"))
        with tempfile.TemporaryDirectory() as directory:
            result = run_pipeline(config, "integration", artifacts_dir=Path(directory))
            run_dir = Path(result["run_dir"])
            for slug in ("tensorflow-to-pytorch", "pytorch-to-tensorflow"):
                summary = result["directions"][slug]
                self.assertEqual(summary["critical_components"], 2)
                self.assertEqual(summary["component_matches"], 2)
                self.assertEqual(summary["test_knowledge_units"], 2)
                self.assertEqual(summary["migration_candidates"], 2)
                self.assertEqual(summary["validation_status"], {"static_validated": 2})
                direction_dir = run_dir / slug
                for stage in (
                    "01-critical-components",
                    "02-component-matches",
                    "03-test-knowledge",
                    "04-migration-candidates",
                    "05-validation",
                    "06-ranked-tests",
                ):
                    self.assertTrue((direction_dir / stage).is_dir())

            tf_to_pt = read_jsonl(
                run_dir
                / "tensorflow-to-pytorch"
                / "04-migration-candidates"
                / "migration_candidates.jsonl"
            )
            pt_code = "\n".join(str(item["code"]) for item in tf_to_pt)
            self.assertIn("torch.relu", pt_code)
            self.assertIn("torch.sum(values, dim=1)", pt_code)

            pt_to_tf = read_jsonl(
                run_dir
                / "pytorch-to-tensorflow"
                / "04-migration-candidates"
                / "migration_candidates.jsonl"
            )
            tf_code = "\n".join(str(item["code"]) for item in pt_to_tf)
            self.assertIn("tensorflow.nn.relu", tf_code)
            self.assertIn("tensorflow.math.reduce_sum(values, axis=1)", tf_code)


if __name__ == "__main__":
    unittest.main()
