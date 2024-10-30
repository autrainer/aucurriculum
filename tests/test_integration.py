import os
from unittest.mock import patch

import aucurriculum.cli
import aucurriculum.postprocessing

from .utils import BaseSharedTempDir


class TestCLIIntegration(BaseSharedTempDir):
    def _train_without_curriculum(self) -> None:
        with patch("sys.argv", [""]):
            aucurriculum.cli.train({"iterations": 2})

    def _train_with_curriculum(self) -> None:
        with patch("sys.argv", [""]):
            aucurriculum.cli.train(
                {
                    "iterations": 2,
                    "curriculum": "Curriculum",
                    "curriculum/sampling": "Balanced",
                    "curriculum/scoring": "Random",
                    "curriculum.scoring.dataset": "ToyTabular-C",
                    "curriculum/pacing": "Linear",
                    "curriculum.pacing.initial_size": [0.1, 0.2],
                    "curriculum.pacing.final_iteration": 0.8,
                }
            )

    def _assert_created_runs(self, path: str, num_runs: int) -> None:
        runs = [
            d
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d != "plots"
        ]
        assert len(runs) == num_runs, f"Should create {num_runs} runs."
        for run in runs:
            assert os.path.isfile(
                os.path.join(path, run, "metrics.csv")
            ), "Should create metrics file."

    def test_curriculum_score_train(self) -> None:
        # train without curriculum
        self._train_without_curriculum()
        self._train_without_curriculum()  # test filtering out existing runs

        _train_base_dir = os.path.join("results", "default", "training")
        self._assert_created_runs(_train_base_dir, 1)

        # curriculum
        with patch("sys.argv", [""]):
            aucurriculum.cli.curriculum(
                {
                    "curriculum/scoring": "Random",
                    "curriculum.scoring.dataset": "ToyTabular-C",
                }
            )
        _curr_base_dir = os.path.join("results", "default", "curriculum")
        assert os.path.isfile(
            os.path.join(_curr_base_dir, "Random", "Random.csv")
        ), "Should create scores file."

        # train with curriculum
        self._train_with_curriculum()
        training_runs = [
            d
            for d in os.listdir(_train_base_dir)
            if os.path.isdir(os.path.join(_train_base_dir, d))
        ]
        assert len(training_runs) == 3, "Should create training runs."
        for r in training_runs:
            assert os.path.isfile(
                os.path.join(_train_base_dir, r, "metrics.csv")
            ), "Should create metrics file."

        # postprocess and aggregate
        aucurriculum.cli.postprocess(
            "results",
            "default",
            aggregate=[["curriculum.pacing.initial_size"]],
        )
        _agg_base_dir = os.path.join(
            "results", "default", "agg_curriculum.pacing.initial_size"
        )
        self._assert_created_runs(_agg_base_dir, 2)

        # aggregate over all curriculum attributes
        _agg = [
            "curriculum",
            "curriculum.sampling",
            "curriculum.scoring",
            "curriculum.pacing",
            "curriculum.pacing.initial_size",
            "curriculum.pacing.final_iteration",
        ]
        aucurriculum.cli.postprocess("results", "default", aggregate=[_agg])
        _agg_base_dir = os.path.join(
            "results", "default", "_".join(["agg"] + _agg)
        )
        self._assert_created_runs(_agg_base_dir, 1)
