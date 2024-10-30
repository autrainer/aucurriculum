import os
from typing import Tuple

import autrainer
from autrainer.core.utils import set_seed
from autrainer.datasets import AbstractDataset
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .abstract_score import AbstractScore
from .utils import load_hydra_configuration


class Random(AbstractScore):
    def __init__(
        self,
        output_directory: str,
        results_dir: str,
        experiment_id: str,
        dataset: str,
        seed: int,
        subset: str = "train",
    ) -> None:
        """Random scoring function that assigns random scores to each sample in
        the dataset.

        Args:
            output_directory: Directory where the scores will be stored.
            results_dir: The directory where the results are stored.
            experiment_id: The ID of the grid search experiment.
            dataset: Dataset ID to use for scoring.
            seed: Seed to use for random scoring.
            subset: Dataset subset to use for scoring in ["train", "dev",
                "test"]. Defaults to "train".
        """
        super().__init__(
            output_directory=output_directory,
            results_dir=results_dir,
            experiment_id=experiment_id,
            run_name=None,
            subset=subset,
        )
        self.dataset_id = dataset
        self.seed = seed

    def preprocess(self) -> Tuple[list, list]:
        config = config = OmegaConf.create({})
        config.dataset = load_hydra_configuration("dataset", self.dataset_id)
        config.seed = self.seed
        run_name = f"{config.dataset.id}_{config.seed}"
        return [config], [run_name]

    def run(
        self,
        config: DictConfig,
        run_config: DictConfig,
        run_name: str,
    ) -> None:
        set_seed(run_config.seed)
        run_config.dataset.pop("criterion")
        run_config.dataset.pop("transform")
        data = autrainer.instantiate(
            config=run_config.dataset,
            instance_of=AbstractDataset,
            batch_size=1,
            seed=run_config.seed,
        )
        df = self.get_dataframe(data, self.subset)
        df["scores"] = np.random.rand(len(df))
        df["decoded"] = df[data.target_column]
        df["encoded"] = df["decoded"].apply(data.target_transform.encode)
        self.save_scores(df, os.path.join(self.output_directory, run_name))
