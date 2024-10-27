import os
from typing import Tuple

import autrainer
from autrainer.datasets import AbstractDataset
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from .abstract_score import AbstractScore
from .utils import load_hydra_configuration


class Predefined(AbstractScore):
    def __init__(
        self,
        output_directory: str,
        results_dir: str,
        experiment_id: str,
        file: str,
        scores_column: str,
        reverse: bool,
        dataset: str,
        subset: str = "train",
    ) -> None:
        """Predefined scoring function using predefined scores from a file.

        Args:
            output_directory: Directory where the scores will be stored.
            results_dir: The directory where the results are stored.
            experiment_id: The ID of the grid search experiment.
            file: Path to the file containing the scores.
            scores_column: Column name of the scores in the file.
            reverse: Whether to reverse the order of the scores.
            dataset: Dataset ID to use for scoring.
            subset: Dataset subset to use for scoring in ["train", "dev",
                "test"]. Defaults to "train".

        Raises:
            ValueError: If the file does not exist.
        """
        super().__init__(
            output_directory=output_directory,
            results_dir=results_dir,
            experiment_id=experiment_id,
            run_name=None,
            subset=subset,
            reverse_score=reverse,
        )
        self.dataset_id = dataset
        if not os.path.exists(file):
            raise ValueError(f"File {file} does not exist")
        self.file = file
        self.scores_column = scores_column

    def preprocess(self) -> Tuple[list, list]:
        config = OmegaConf.create({})
        config.dataset = load_hydra_configuration("dataset", self.dataset_id)
        run_name = (
            config.dataset.id + "_" + os.path.basename(self.file).split(".")[0]
        )
        return [config], [run_name]

    def run(
        self, config: DictConfig, run_config: DictConfig, run_name: str
    ) -> None:
        run_config.dataset.pop("criterion")
        run_config.dataset.pop("transform")
        data = autrainer.instantiate(
            config=run_config.dataset,
            instance_of=AbstractDataset,
            batch_size=1,
            seed=0,
        )
        df = self.get_dataframe(data, self.subset)
        scores_df = pd.read_csv(self.file)
        scores_df["scores"] = scores_df[self.scores_column]
        scores_df["decoded"] = df[data.target_column]
        scores_df["encoded"] = scores_df["decoded"].apply(
            data.target_transform.encode
        )
        df = pd.concat([df, scores_df], axis=1)
        self.save_scores(df, os.path.join(self.output_directory, run_name))
