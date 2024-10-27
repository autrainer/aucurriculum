from abc import ABC, abstractmethod
import os
from typing import Callable, List, Optional, Tuple, Union

import autrainer
from autrainer.augmentations import AugmentationManager
from autrainer.core.utils import Timer, set_device, set_seed
from autrainer.datasets import AbstractDataset
from autrainer.models import AbstractModel
from autrainer.postprocessing.postprocessing_utils import load_yaml
from autrainer.transforms import TransformManager
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class AbstractScore(ABC):
    def __init__(
        self,
        output_directory: str,
        results_dir: str,
        experiment_id: str,
        run_name: Union[str, List[str]],
        stop: str = None,
        subset: str = "train",
        reverse_score: bool = False,
        criterion: str = None,
    ) -> None:
        """Abstract class for scoring functions.

        Args:
            output_directory: Directory where the scores will be stored.
            results_dir: The directory where the results are stored.
            experiment_id: The ID of the grid search experiment.
            run_name: Name or list of names of the runs to score. Runs can be
                single runs or aggregated runs.
            stop: Model state dict to load or to stop at in ["best", "last"].
                Defaults to None.
            subset: Dataset subset to use for scoring in ["train", "dev",
                "test"]. Defaults to "train".
            reverse_score: Whether to reverse the score ranking. Defaults to
                False.
            criterion: The criterion to use for scoring. If None, no criterion
                will be used. Defaults to None.

        Raises:
            ValueError: If subset is not in ["train", "dev", "test"] or if stop
                is not in ["best", "last", None].
        """
        self.output_directory = output_directory
        self.results_dir = results_dir
        self.experiment_id = experiment_id
        self.run_name = run_name
        if subset not in ["train", "dev", "test"]:
            raise ValueError(f"Subset '{subset}' not supported")
        self.subset = subset
        if stop not in ["best", "last", None]:
            raise ValueError(
                f"Stop must be in ['best', 'last'], but got stop='{stop}'"
            )
        self.stop = stop
        self.reverse_score = reverse_score
        self.criterion_cfg = criterion

    def preprocess(self) -> Tuple[list, list]:
        """Preprocess one or multiple runs, creating a list of configurations
        and a list of run names to score.

        Returns:
            List of configurations and list of run names to score.
        """
        if isinstance(self.run_name, str):
            return self._preprocess_single(self.run_name)
        configs = []
        runs = []
        for run in self.run_name:
            _configs, _runs = self._preprocess_single(run)
            for c, r in zip(_configs, _runs):
                if r in runs:
                    continue
                configs.append(c)
                runs.append(r)
        return configs, runs

    def _preprocess_single(self, run_name: str) -> Tuple[list, list]:
        base_path = os.path.join(self.results_dir, self.experiment_id)
        dirs = [d for d in os.listdir(base_path) if d.startswith("agg_")]
        dirs.append("training")
        for d in dirs:
            if os.path.exists(os.path.join(base_path, d, run_name)):
                config = DictConfig(
                    load_yaml(
                        os.path.join(
                            base_path, d, run_name, ".hydra", "config.yaml"
                        )
                    )
                )
                if d == "training":
                    if self.criterion_cfg is not None:
                        run_name = (
                            f"{run_name}_{self.criterion_cfg.split('.')[-1]}"
                        )
                    if self.stop is not None:
                        run_name = f"{run_name}_{self.stop[0]}"
                    return [config], [run_name]
                runs = load_yaml(
                    os.path.join(base_path, d, run_name, "runs.yaml")
                )
                configs = [
                    DictConfig(
                        load_yaml(
                            os.path.join(
                                base_path,
                                "training",
                                r,
                                ".hydra",
                                "config.yaml",
                            )
                        )
                    )
                    for r in runs
                ]
                if self.criterion_cfg is not None:
                    runs = [
                        f"{r}_{self.criterion_cfg.split('.')[-1]}"
                        for r in runs
                    ]
                if self.stop is not None:
                    runs = [f"{r}_{self.stop[0]}" for r in runs]
                return configs, runs
        raise ValueError(f"Run {run_name} does not exist")

    @abstractmethod
    def run(
        self,
        config: DictConfig,
        run_config: DictConfig,
        run_name: str,
    ) -> None:
        """Run the scoring function for a single run and generate the scores.

        Args:
            config: The configuration of the curriculum scoring function.
            run_config: The configuration of the run to score.
            run_name: The name of the run to score.
        """

    def postprocess(self, score_id: str, runs: list) -> None:
        """Postprocess the scores and create the final scoring function
        ordering by averaging the scores of multiple runs and ranking the
        samples based on the mean score.

        Args:
            score_id: ID of the score to save.
            runs: List of run names to postprocess and include in the score.
        """
        df = pd.read_csv(
            os.path.join(self.output_directory, runs[0], "scores.csv")
        )
        df.drop(columns=["scores"], inplace=True)
        scores = pd.DataFrame()
        for run in runs:
            scores[run] = pd.read_csv(
                os.path.join(self.output_directory, run, "scores.csv")
            )["scores"]
        df["mean"] = scores.mean(axis=1)
        df["ranks"] = self.rank_and_normalize(df)
        df.to_csv(
            os.path.join(self.output_directory, score_id + ".csv"),
            index=False,
        )

    def split_run_name(self, run_name: str) -> Tuple[str, str]:
        """Split the full run name run name into the underlying training run
        name and the full run name containing the `stop` iteration (and
        optional criterion).

        Args:
            run_name: The run name to split.

        Returns:
            The name of the underlying training run and the full run name.
        """
        if self.criterion_cfg is not None:
            return "_".join(run_name.split("_")[:-2]), run_name
        return "_".join(run_name.split("_")[:-1]), run_name

    def create_criterion(
        self,
        data: AbstractDataset,
        reduction: str = "none",
    ) -> torch.nn.Module:
        """Create the criterion for the scoring function based on the
        criterion configuration.

        Args:
            data: Dataset to use for criterion setup.
            reduction: Reduction to use for the criterion. Defaults to "none".

        Returns:
            Criterion for the scoring function.
        """
        criterion = autrainer.instantiate_shorthand(
            config=self.criterion_cfg,
            instance_of=torch.nn.modules.loss._Loss,
        )
        if hasattr(criterion, "setup"):
            criterion.setup(data)
        criterion.reduction = reduction
        return criterion

    @staticmethod
    def prepare_data_and_model(
        cfg: DictConfig,
    ) -> Tuple[AbstractDataset, AbstractModel]:
        """Prepare the dataset and model for the scoring function based on the
        underlying training run configuration.

        Args:
            cfg: The configuration of the underlying training run.

        Returns:
            The instantiated dataset and model.
        """
        if isinstance(cfg.seed, str):
            training_seed, dataset_seed = map(int, cfg.seed.split("-"))
        else:
            training_seed = dataset_seed = cfg.seed
        set_seed(training_seed)

        am = AugmentationManager(cfg.augmentation)
        train_aug, *_ = am.get_augmentations()
        transform_manager = TransformManager(
            model_transform=cfg.model.pop("transform", None),
            dataset_transform=cfg.dataset.pop("transform", None),
            train_augmentation=train_aug,
        )

        transforms = transform_manager.get_transforms()
        train_transform, dev_transform, test_transform = transforms

        cfg.criterion = cfg.dataset.pop("criterion", None)
        data = autrainer.instantiate(
            config=cfg.dataset,
            instance_of=AbstractDataset,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            seed=dataset_seed,
            batch_size=cfg.batch_size,
            inference_batch_size=cfg.get("inference_batch_size", None),
        )

        cfg.model.pop("pretrained", None)
        cfg.model.output_dim = data.output_dim
        model = autrainer.instantiate(
            config=cfg.model,
            instance_of=AbstractModel,
        )

        return data, model

    def load_model_checkpoint(
        self,
        model: torch.nn.Module,
        run_name: str,
    ) -> None:
        """Load the trained model checkpoint based on the run name and run
        configuration.
        The model will be loaded from the best checkpoint if stop is set to
        "best" and from the last checkpoint if stop is set to "last".

        Args:
            model: Model to load the checkpoint into.
            run_name: Name of the run to load the checkpoint from.
        """
        training_dir = os.path.join(
            self.results_dir, self.experiment_id, "training", run_name
        )
        if self.stop == "best":
            model_checkpoint = os.path.join(training_dir, "_best", "model.pt")
        else:
            dirs = os.listdir(training_dir)
            dirs = [
                d
                for d in dirs
                if d.startswith("epoch_") or d.startswith("step_")
            ]
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[-1]))
            model_checkpoint = os.path.join(training_dir, dirs[-1], "model.pt")
        model.load_state_dict(
            torch.load(
                model_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()

    @staticmethod
    def forward_pass(
        model: AbstractModel,
        loader: DataLoader,
        batch_size: int,
        output_map_fn: Callable[[torch.Tensor], torch.Tensor],
        output_size: Optional[int] = None,
        tqdm_desc: str = "Scoring Forward Pass",
        disable_progress_bar: bool = True,
        device: Optional[torch.device] = None,
        timer: Optional[Timer] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a forward pass through the model and return the outputs and
        labels.

        Args:
            model: Model to perform the forward pass with.
            loader: DataLoader to use for the forward pass.
            batch_size: Batch size to use for the forward pass.
            output_map_fn: Function to map the model outputs to the desired
                output format.
            output_size: Size of the output tensor. If None, the model output
                should be a single scalar. Defaults to None.
            tqdm_desc: Description for the tqdm progress bar. Defaults to
                "Scoring Forward Pass".
            disable_progress_bar: Whether to disable the progress bar. Defaults
                to True.
            device: Device to use for the forward pass. If None, the device
                will be set to "cpu". Defaults to None.
            timer: Timer to time the forward pass. If provided, the timer is
                started before the forward pass and stopped after the forward
                pass. Defaults to None.

        Returns:
            Mapped model outputs and labels.
        """
        if device is None:
            device = set_device("cpu")
        dataset_size = len(loader.dataset)
        if output_size is None:
            outputs = torch.zeros(dataset_size, dtype=torch.float32)
        else:
            outputs = torch.zeros(
                dataset_size, output_size, dtype=torch.float32
            )
        labels = torch.zeros(dataset_size, dtype=torch.long)
        model.to(device)
        if timer:
            timer.start()
        with torch.no_grad():
            for idx, (x, y, _) in enumerate(
                tqdm(loader, desc=tqdm_desc, disable=disable_progress_bar)
            ):
                _lower = idx * batch_size
                _upper = min(dataset_size, (idx + 1) * batch_size)
                x, y = x.to(device), y.to(device)
                _map_args = output_map_fn.__annotations__.keys()
                if "_lower" in _map_args and "_upper" in _map_args:
                    outputs[_lower:_upper] = output_map_fn(
                        model(x), y, _lower=_lower, _upper=_upper
                    ).cpu()
                else:
                    outputs[_lower:_upper] = output_map_fn(model(x), y).cpu()
                labels[_lower:_upper] = y.cpu()
        if timer:
            timer.stop()
        return outputs.numpy(), labels.numpy()

    def create_dataframe(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        data: AbstractDataset,
    ) -> pd.DataFrame:
        """Create a dataframe from the scores, labels, and dataset.

        Args:
            scores: The score for each sample.
            labels: The encoded labels for each sample.
            data: The dataset.

        Returns:
            The dataframe with the scores, encoded labels, and decoded labels.
        """
        df = self.get_dataframe(data, self.subset)
        df["scores"] = scores
        df["encoded"] = labels
        df["decoded"] = df["encoded"].apply(data.target_transform.decode)
        return df

    def rank_and_normalize(self, df: pd.DataFrame) -> pd.Series:
        """Rank and normalize the scores in the dataframe by ranking the scores
        using method="first" and normalizing the ranks to the range [0, 1].
        If `reverse_score` is set to True, the ranks will be reversed.

        In the resulting difficulty orderint, lower ranks always indicate
        easier samples.

        Args:
            df: The output dataframe with "mean" column containing the scores.

        Returns:
            The normalized ranks.
        """
        ascending = not self.reverse_score
        ranks = df["mean"].rank(ascending=ascending, method="first")
        ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
        return ranks

    @staticmethod
    def save_scores(df: pd.DataFrame, path: str) -> None:
        """Save the scores dataframe to the specified path.

        Args:
            df: The scores dataframe.
            path: The path to save the scores dataframe.
        """
        os.makedirs(path, exist_ok=True)
        df.to_csv(os.path.join(path, "scores.csv"), index=False)

    @staticmethod
    def get_dataset_subset(data: AbstractDataset, subset: str) -> Dataset:
        """Get the dataset subset.

        Args:
            data: The dataset.
            subset: The dataset subset to get in ["train", "dev", "test"].

        Returns:
            The dataset subset.
        """
        return {
            "train": data.train_dataset,
            "dev": data.dev_dataset,
            "test": data.test_dataset,
        }[subset]

    @staticmethod
    def get_dataframe(data: AbstractDataset, subset: str) -> pd.DataFrame:
        """Get the dataframe of the dataset subset.

        Args:
            data: The dataset.
            subset: The dataset subset to get in ["train", "dev", "test"].

        Returns:
            The dataframe of the dataset subset.
        """
        return {
            "train": data.df_train,
            "dev": data.df_dev,
            "test": data.df_test,
        }[subset].reset_index()
