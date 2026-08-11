import unittest

from cosmit.domain.models import (
    ComponentId,
    CriticalitySignals,
    SimilaritySignals,
    TestIntent,
    TestKnowledge,
)


class DomainModelTests(unittest.TestCase):
    def test_component_has_stable_canonical_id(self) -> None:
        component = ComponentId("tensorflow", "tensorflow.nn.relu")
        self.assertEqual(component.canonical, "tensorflow:tensorflow.nn.relu")

    def test_criticality_score_uses_all_five_signals(self) -> None:
        signals = CriticalitySignals(1.0, 0.8, 0.6, 0.4, 0.2)
        self.assertAlmostEqual(signals.score(), 0.6)

    def test_similarity_rejects_out_of_range_signal(self) -> None:
        with self.assertRaises(ValueError):
            SimilaritySignals(1.1, 0.5, 0.5, 0.5, 0.5, 0.5)

    def test_test_knowledge_requires_execution_and_oracle(self) -> None:
        component = ComponentId("pytorch", "torch.nn.CrossEntropyLoss")
        with self.assertRaises(ValueError):
            TestKnowledge(
                knowledge_id="k-1",
                source_component=component,
                source_test="test_cross_entropy",
                intents=(TestIntent.NUMERICAL,),
            )


if __name__ == "__main__":
    unittest.main()
