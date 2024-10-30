import os
from typing import Tuple

from autrainer.core.utils import Timer, set_device
from autrainer.postprocessing.postprocessing_utils import load_yaml
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .abstract_score import AbstractScore


class FirstIteration(AbstractScore):
    def __init__(
        self,
        output_directory: str,
        results_dir: str,
        experiment_id: str,
        run_name: str,
        stop: str = "best",
        subset: str = "train",
    ) -> None:
        """First Iteration scoring function computing the first iteration
        in which the model correctly predicts the target including all
        subsequent iterations for each sample in the dataset individually as
        described in:
        https://arxiv.org/abs/2012.03107

        Args:
            output_directory: Directory where the scores will be stored.
            results_dir: The directory where the results are stored.
            experiment_id: The ID of the grid search experiment.
            run_name: Name or list of names of the runs to score. Runs can be
                single runs or aggregated runs.
            stop: Model state dict to load or to stop at in ["best", "last"].
                Defaults to "best".
            subset: Dataset subset to use for scoring in ["train", "dev",
                "test"]. Defaults to "train".
        """
        super().__init__(
            output_directory=output_directory,
            results_dir=results_dir,
            experiment_id=experiment_id,
            run_name=run_name,
            stop=stop,
            subset=subset,
        )

    def run(
        self, config: DictConfig, run_config: DictConfig, run_name: str
    ) -> None:
        run_name, full_run_name = self.split_run_name(run_name)
        training_dir = os.path.join(
            self.results_dir, self.experiment_id, "training", run_name
        )
        training_type = run_config.training_type
        dirs = os.listdir(training_dir)
        if self.stop == "best":
            best_iteration = load_yaml(
                os.path.join(training_dir, "_best", "dev.yaml")
            )["iteration"]
        else:
            best_iteration = np.inf

        iteration_folders = []
        for d in dirs:
            if not d.startswith(training_type.lower()):
                continue
            if not os.path.exists(os.path.join(training_dir, d, "model.pt")):
                continue
            if int(d.split("_")[-1]) > best_iteration:
                continue
            iteration_folders.append(d)

        if not iteration_folders:
            raise ValueError(
                (
                    f"No model states found for '{run_name}'"
                    f"\n\tin '{training_dir}'"
                    f"\n\twith stop='{self.stop}'"
                )
            )

        iteration_folders = sorted(
            iteration_folders, key=lambda x: int(x.split("_")[-1])
        )
        run_path = os.path.join(self.output_directory, full_run_name)
        self.forward_timer = Timer(run_path, "model_forward")
        self.batch_size = config.get(
            "batch_size", run_config.get("batch_size", 32)
        )

        data, model = self.prepare_data_and_model(run_config)
        dataset = self.get_dataset_subset(data, self.subset)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        self.device = set_device(config.device)

        dfs = []
        labels = None
        for iteration_folder in tqdm(
            iteration_folders,
            desc=full_run_name,
            disable=not config.get("progress_bar", False),
        ):
            df = pd.DataFrame()
            o, l = self._run_iteration(
                model, loader, run_name, iteration_folder
            )
            df[iteration_folder] = o
            if labels is None:
                labels = l
            dfs.append(df)

        self.forward_timer.save()
        df = pd.concat(dfs, axis=1)
        df["encoded"] = labels
        df["decoded"] = df["encoded"].apply(data.target_transform.decode)
        df = self._calculate_first_iteration(df)
        df = pd.concat(
            [
                self.get_dataframe(data, self.subset),
                df[["scores", "encoded", "decoded"]],
            ],
            axis=1,
        )
        self.save_scores(df, run_path)

    def _run_iteration(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        run_name: str,
        iteration_folder: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        model_checkpoint = os.path.join(
            self.results_dir,
            self.experiment_id,
            "training",
            run_name,
            iteration_folder,
            "model.pt",
        )
        model.load_state_dict(
            torch.load(
                model_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        outputs, labels = self.forward_pass(
            model=model,
            loader=loader,
            batch_size=self.batch_size,
            output_map_fn=lambda outs, y: torch.argmax(outs, dim=1),
            disable_progress_bar=True,
            device=self.device,
            timer=self.forward_timer,
        )
        return outputs, labels

    def _calculate_first_iteration(self, df: pd.DataFrame) -> pd.DataFrame:
        iteration_cols = list(df.columns)
        iteration_cols.remove("encoded")
        iteration_cols.remove("decoded")
        predictions = df[iteration_cols].eq(df["encoded"], axis=0)
        scores = len(iteration_cols) - (
            predictions.iloc[:, ::-1].cumprod(axis=1).sum(axis=1)
        )

        df["scores"] = scores
        df["scores"] = df["scores"] / len(iteration_cols)
        return df
