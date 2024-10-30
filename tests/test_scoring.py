import os
import shutil
import tempfile
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
from omegaconf import DictConfig
import pandas as pd
import pytest

import aucurriculum.cli
from aucurriculum.curricula.scoring import (
    AbstractScore,
    CELoss,
    CumulativeAccuracy,
    CVLoss,
    FirstIteration,
    Predefined,
    PredictionDepth,
    Random,
    TransferTeacher,
)

from .utils import BaseIndividualTempDir


class TestScoringFunctions(BaseIndividualTempDir):
    @classmethod
    def setup_class(cls) -> None:
        cls.temp_run_dir = tempfile.TemporaryDirectory()
        with patch("sys.argv", [""]):
            aucurriculum.cli.train({"results_dir": cls.temp_run_dir.name})
        _train_path = os.path.join(
            cls.temp_run_dir.name, "default", "training"
        )
        cls.temp_run_name = next(
            f
            for f in os.listdir(_train_path)
            if os.path.isdir(os.path.join(_train_path, f))
        )

    @classmethod
    def teardown_class(cls) -> None:
        cls.temp_run_dir.cleanup()

    @pytest.mark.parametrize(
        "cls, kwargs",
        [
            (CELoss, {"criterion": "autrainer.criterions.CrossEntropyLoss"}),
            (
                CVLoss,
                {
                    "criterion": "autrainer.criterions.CrossEntropyLoss",
                    "splits": 3,
                    "setup": {
                        "filters": [{"type": "expr", "expr": "1 != 1"}],
                        "dataset": "ToyTabular-C",
                        "model": "ToyFFNN",
                        "optimizer": "Adam",
                        "learning_rate": 0.001,
                        "scheduler": "None",
                        "augmentation": "None",
                        "seed": 1,
                        "batch_size": 32,
                        "inference_batch_size": 32,
                        "plotting": "Default",
                        "training_type": "epoch",
                        "iterations": 2,
                        "eval_frequency": 1,
                        "save_frequency": 2,
                        "save_train_outputs": True,
                        "save_dev_outputs": True,
                        "save_test_outputs": True,
                    },
                },
            ),
            (CumulativeAccuracy, {}),
            (FirstIteration, {}),
            (
                PredictionDepth,
                {"probe_placements": [r"linear\d+"], "max_embedding_size": 32},
            ),
            (TransferTeacher, {"model": "ToyFFNN", "dataset": "ToyTabular-C"}),
        ],
    )
    def test_scoring_functions(
        self,
        cls: AbstractScore,
        kwargs: Dict[str, Any],
    ) -> None:
        shutil.copytree(self.temp_run_dir.name, "results")
        if "run_name" in cls.__init__.__code__.co_varnames:
            kwargs["run_name"] = self.temp_run_name
        sf = cls(
            output_directory=f"results/default/curriculum{cls.__name__}",
            results_dir="results",
            experiment_id="default",
            **kwargs,
        )
        self._calculate_scores(sf)

    def test_predefined_scoring_function(self) -> None:
        pd.DataFrame(
            {
                "id": range(600),
                "score": np.random.rand(600),
            }
        ).to_csv("scores.csv", index=False)

        sf = Predefined(
            output_directory="results/default/curriculum/Predefined",
            results_dir="results",
            experiment_id="default",
            file="scores.csv",
            scores_column="score",
            reverse=False,
            dataset="ToyTabular-C",
        )
        self._calculate_scores(sf)

    def test_random_scoring_function(self) -> None:
        sf = Random(
            output_directory="results/default/curriculum/Random",
            results_dir="results",
            experiment_id="default",
            dataset="ToyTabular-C",
            seed=1,
        )
        self._calculate_scores(sf)

    def test_invalid_subset(self) -> None:
        kwargs = self._celoss_base_kwargs()
        kwargs["subset"] = "invalid"
        with pytest.raises(ValueError):
            CELoss(**kwargs)

    def test_invalid_stop(self) -> None:
        kwargs = self._celoss_base_kwargs()
        kwargs["stop"] = "invalid"
        with pytest.raises(ValueError):
            CELoss(**kwargs)

    def test_multiple_scores(self) -> None:
        shutil.copytree(self.temp_run_dir.name, "results")
        kwargs = self._celoss_base_kwargs()
        kwargs["run_name"] = [self.temp_run_name, self.temp_run_name]
        sf = CELoss(**kwargs)
        configs, _ = sf.preprocess()
        assert len(configs) == 1, "Should have 1 configuration."

    def test_aggregated_score(self) -> None:
        shutil.copytree(self.temp_run_dir.name, "results")
        aucurriculum.cli.postprocess(
            "results",
            "default",
            aggregate=[["seed"]],
        )
        kwargs = self._celoss_base_kwargs()
        kwargs["run_name"] = self.temp_run_name[:-1] + "#"
        sf = CELoss(**kwargs)
        configs, _ = sf.preprocess()
        assert len(configs) == 1, "Should have 1 configurations."

    def _celoss_base_kwargs(self) -> Dict[str, Any]:
        return {
            "output_directory": "results/default/curriculum/CELoss",
            "results_dir": "results",
            "experiment_id": "default",
            "run_name": self.temp_run_name,
            "criterion": "autrainer.criterions.CrossEntropyLoss",
        }

    @staticmethod
    def _calculate_scores(sf: AbstractScore) -> None:
        cfg = DictConfig(
            {
                "batch_size": 32,
                "progress_bar": True,
                "training_type": "epoch",
                "device": "cuda:0",
            }
        )
        configs, runs = sf.preprocess()
        for config, run in zip(configs, runs):
            sf.run(cfg, DictConfig(config).copy(), run)
            output_dir = os.path.join(sf.output_directory, run, "scores.csv")
            assert os.path.isfile(output_dir), "Scores should be saved."
        sf.postprocess(sf.__class__.__name__, runs)
