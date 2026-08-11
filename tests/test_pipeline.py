import unittest

from cosmit.pipeline.stages import PIPELINE_STAGES, StageName, validate_pipeline_contract


class PipelineContractTests(unittest.TestCase):
    def test_six_stages_follow_patent_order(self) -> None:
        self.assertEqual(
            [stage.name for stage in PIPELINE_STAGES],
            [
                StageName.IDENTIFY,
                StageName.RETRIEVE,
                StageName.EXTRACT,
                StageName.MIGRATE,
                StageName.VALIDATE,
                StageName.RANK,
            ],
        )

    def test_all_stage_inputs_are_available(self) -> None:
        validate_pipeline_contract()


if __name__ == "__main__":
    unittest.main()
