import os
import shutil
from typing import Tuple

import autrainer
from autrainer.core.utils import save_hardware_info, set_device
from autrainer.postprocessing.postprocessing_utils import load_yaml, save_yaml
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from aucurriculum.core.plotting import CurriculumPlots

from .scoring import AbstractScore


class CurriculumScoreManager:
    def __init__(
        self,
        cfg: DictConfig,
        output_directory: str,
    ) -> None:
        """Curriculum score manager to control pre-processing, running, and
        post-processing of scoring functions.

        Args:
            cfg: Curriculum configuration.
            output_directory: The output directory to save the results.
        """
        self.output_directory = output_directory
        self.results_dir = cfg.results_dir
        self.experiment_id = cfg.experiment_id
        self.device = set_device(cfg.device)
        self.cfg = cfg
        _score_cfg = cfg.curriculum.scoring.copy()
        _score_cfg.pop("type")
        self.scoring_function = autrainer.instantiate(
            config=_score_cfg,
            instance_of=AbstractScore,
            output_directory=output_directory,
            results_dir=self.results_dir,
            experiment_id=self.experiment_id,
        )

    def preprocess(self) -> Tuple[list, list]:
        """Preprocess the scoring function, possibly creating one or more
        configurations and run names.

        Returns:
            List of configurations and list of run names.
        """
        configs, runs = self.scoring_function.preprocess()
        self._create_configs(runs, configs)
        self._create_mappings(runs)
        return configs, runs

    def run(self, run_config: DictConfig, run_name: str) -> None:
        """Run a single scoring function.

        Args:
            run_config: The run configuration.
            run_name: The name of the run.
        """
        scores = os.path.join(self.output_directory, run_name, "scores.csv")
        if os.path.exists(scores):
            return
        self.scoring_function.run(self.cfg.copy(), run_config, run_name)
        save_hardware_info(
            os.path.join(self.output_directory, run_name),
            device=self.device,
        )

    def postprocess(
        self,
        score_id: str,
        correlation: DictConfig = None,
    ) -> None:
        """Postprocess the scoring function and optionally create a correlation
        matrix.

        Args:
            score_id: The score ID to postprocess.
            correlation: The correlation matrix configuration. Dictionary of
                lists of score IDs to include in a single correlation matrix.
                Defaults to None.
        """
        mappings = load_yaml(
            os.path.join(self.output_directory, "mappings.yaml")
        )
        runs = mappings[score_id]
        self.scoring_function.postprocess(score_id, runs)
        self._visualize_score(score_id)
        if correlation is not None:
            self._correlation_matrix(correlation)

    def _create_configs(self, runs: list, configs: list) -> None:
        for run_name, config in zip(runs, configs):
            s = os.path.join(self.output_directory, run_name, "score.yaml")
            if os.path.exists(s):
                continue

            os.makedirs(
                os.path.join(self.output_directory, run_name), exist_ok=True
            )
            save_yaml(
                os.path.join(self.output_directory, run_name, "config.yaml"),
                OmegaConf.to_container(config, resolve=True),
            )
            save_yaml(
                os.path.join(self.output_directory, run_name, "score.yaml"),
                OmegaConf.to_container(self.cfg, resolve=True),
            )
        shutil.rmtree(os.path.join(self.output_directory, ".hydra"))

    def _create_mappings(self, runs: list) -> None:
        if not os.path.exists(
            os.path.join(self.output_directory, "mappings.yaml")
        ):
            mappings = DictConfig({})
        else:
            mappings = DictConfig(
                load_yaml(os.path.join(self.output_directory, "mappings.yaml"))
            )
        mappings[self.cfg.curriculum.scoring.id] = runs
        save_yaml(
            os.path.join(self.output_directory, "mappings.yaml"),
            OmegaConf.to_container(mappings, resolve=True),
        )

    def _visualize_score(self, score_id: str) -> None:
        path = os.path.join(self.output_directory, score_id + ".csv")
        df = pd.read_csv(path)
        cp = CurriculumPlots(
            output_directory=self.output_directory,
            training_type="",
            **self.cfg.plotting,
        )
        cp.plot_score(df, score_id)

    def _correlation_matrix(self, correlation: DictConfig) -> None:
        for matrix_name, scores in correlation.items():
            df = pd.DataFrame()
            base_path = os.path.dirname(self.output_directory)
            dirs = [
                d
                for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))
            ]
            for score_dir in dirs:
                csv_names = [
                    f
                    for f in os.listdir(os.path.join(base_path, score_dir))
                    if f.endswith(".csv")
                ]
                for csv_name in csv_names:
                    score_df = pd.read_csv(
                        os.path.join(base_path, score_dir, csv_name)
                    )
                    name = csv_name.replace(".csv", "")
                    if scores == "all" or name in scores:
                        df[name] = score_df["ranks"]
            if df.empty:
                continue
            cp = CurriculumPlots(
                output_directory=base_path,
                training_type="",
                **self.cfg.plotting,
            )
            cp.plot_correlation_matrix(df, matrix_name)
