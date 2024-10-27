import itertools
import os
from typing import Tuple

from autrainer.core.utils import Timer, set_device
from autrainer.postprocessing.postprocessing_utils import save_yaml
from hydra_filter_sweeper import FILTERMAP
from omegaconf import DictConfig, ListConfig, OmegaConf
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .abstract_score import AbstractScore
from .utils import load_hydra_configuration


class CVLoss(AbstractScore):
    def __init__(
        self,
        output_directory: str,
        results_dir: str,
        experiment_id: str,
        splits: int,
        setup: DictConfig,
        criterion: str,
        stop: str = "best",
        subset: str = "train",
    ) -> None:
        """Cross-Validation Loss scoring function computing the cross-entropy
        loss for each sample in the dataset individually. The dataset is split
        into `splits` parts and the loss is computed for each part individually
        by training on the remaining parts as described in:
        TODO: add reference once paper is published.

        Args:
            output_directory: Directory where the scores will be stored.
            results_dir: The directory where the results are stored.
            experiment_id: The ID of the grid search experiment.
            splits: Number of splits for the cross-validation.
            setup: Configuration for the grid search to perform for each split.
                Each configuration parameter can be a string or list of strings
                for multiple configurations. The following parameters are
                required:

                - filters: Optional list of filters to apply to the runs.
                - dataset: Dataset ID.
                - model: Model ID.
                - optimizer: Optimizer ID.
                - learning_rate: Learning rate.
                - scheduler: Scheduler ID.
                - augmentation: Augmentation ID.
                - seed: Seed.
                - batch_size: Batch size.
                - inference_batch_size: Batch size for inference.
                - plotting: Plotting ID.
                - training_type: Training type.
                - iterations: Number of iterations.
                - eval_frequency: Evaluation frequency.
                - save_frequency: Save frequency.
                - save_train_outputs: Whether to save the training outputs.
                - save_dev_outputs: Whether to save the dev outputs.
                - save_test_outputs: Whether to save the test outputs.

            criterion: The criterion to use for obtaining the per-example loss.
                The reduction of the criterion is automatically set to "none".
            stop: Model state dict to load or to stop at in ["best", "last"].
                Defaults to "best".
            subset: Dataset subset to use for scoring in ["train", "dev",
                "test"]. Defaults to "train".

        Raises:
            ValueError: If the number of splits is less than 2.
        """
        super().__init__(
            output_directory=output_directory,
            results_dir=results_dir,
            experiment_id=experiment_id,
            run_name=None,
            stop=stop,
            subset=subset,
            criterion=criterion,
        )
        if splits < 2:
            raise ValueError(
                f"Number of splits must be at least 2, got '{splits}'"
            )
        self.num_splits = splits
        self.setup = {"split": list(range(splits))}
        self.filters = setup.pop("filters", None)
        for k, v in setup.items():
            if isinstance(v, ListConfig):
                self.setup[k] = OmegaConf.to_container(v)
            if isinstance(v, list):
                self.setup[k] = v
            else:
                self.setup[k] = [v]

    def preprocess(self) -> Tuple[list, list]:
        configs = []
        runs = []
        combinations = list(itertools.product(*self.setup.values()))
        for c in combinations:
            config = dict(zip(self.setup.keys(), c))
            for k, v in config.items():
                if not isinstance(v, str) or k in ["seed", "training_type"]:
                    continue
                config[k] = load_hydra_configuration(k, v)
            config = OmegaConf.create(config)
            OmegaConf.resolve(config)
            run_name = self._get_run_name(config)
            configs.append(config)
            runs.append(run_name)
        if self.criterion_cfg is not None:
            runs = [f"{r}_{self.criterion_cfg.split('.')[-1]}" for r in runs]
        if self.stop is not None:
            runs = [f"{r}_{self.stop[0]}" for r in runs]
        if self.filters is not None:
            f = [self._filter_out_run(c, r) for r, c in zip(runs, configs)]
            runs = [r for r, f in zip(runs, f) if not f]
            configs = [c for c, f in zip(configs, f) if not f]

        return configs, runs

    def _filter_out_run(self, config: DictConfig, run_name: str) -> bool:
        for f in self.filters:
            f = f.copy()
            fail = f.pop("fail", True)
            filter_cls = FILTERMAP[f.pop("type")]()

            try:
                should_filter = filter_cls.filter(
                    config,
                    os.path.join(self.output_directory, run_name),
                    **f,
                )
            except Exception as e:
                if fail:
                    raise ValueError(f"Filter {f} failed: {e}") from e

            if should_filter:
                return True

        return False

    def run(
        self, config: DictConfig, run_config: DictConfig, run_name: str
    ) -> None:
        if isinstance(run_config.seed, str):
            _, dataset_seed = map(int, run_config.seed.split("-"))
        else:
            dataset_seed = run_config.seed
        run_config.progress_bar = config.get("progress_bar", False)
        run_config.curriculum = DictConfig({"id": "None", "_target_": "None"})
        run_config.device = config.device
        trainer_config = run_config.copy()
        run_path = os.path.join(self.output_directory, run_name)
        criterion = run_config.dataset.pop("criterion")
        data, _ = self.prepare_data_and_model(run_config)
        run_config.dataset.criterion = criterion

        train_dataset = self.get_dataset_subset(data, "train")
        if self.subset != "train":
            score_dataset = self.get_dataset_subset(data, self.subset)
            score_on = score_dataset
        else:
            score_on = train_dataset

        dataset_splits = self._split_train_dataset(
            run_config.split, score_on, dataset_seed
        )
        train_subset, score_subset, score_indices = dataset_splits
        if self.subset != "train":
            train_subset += train_dataset

        g = torch.Generator().manual_seed(dataset_seed)
        run_config.progress_bar = config.get("progress_bar", False)
        hydra_path = os.path.join(run_path, ".hydra")
        os.makedirs(hydra_path, exist_ok=True)
        save_yaml(
            os.path.join(hydra_path, "config.yaml"),
            OmegaConf.to_container(run_config, resolve=True),
        )
        from autrainer.training import ModularTaskTrainer

        trainer = ModularTaskTrainer(
            cfg=trainer_config,
            output_directory=run_path,
            experiment_id=f"{self.experiment_id}.curriculum.CVLoss",
            run_name=run_name,
        )
        trainer.train_loader = DataLoader(
            train_subset,
            batch_size=run_config.batch_size,
            shuffle=True,
            generator=g,
        )
        score_loader = DataLoader(
            score_subset, batch_size=run_config.batch_size, shuffle=False
        )
        trainer.train()
        device = set_device(config.device)
        criterion = self.create_criterion(data).to(device)

        if self.stop == "best":
            trainer.bookkeeping.load_state(trainer.model, "model.pt", "_best")
        else:
            dirs = os.listdir(run_path)
            dirs = [d for d in dirs if d.startswith(run_config.training_type)]
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[-1]))
            trainer.bookkeeping.load_state(trainer.model, "model.pt", dirs[-1])

        trainer.model.eval()

        forward_timer = Timer(run_path, "model_forward")
        losses, labels = self.forward_pass(
            model=trainer.model,
            loader=score_loader,
            batch_size=run_config.batch_size,
            output_map_fn=lambda outs, y: criterion(outs, y),
            tqdm_desc="CV Loss",
            disable_progress_bar=not config.get("progress_bar", False),
            device=device,
            timer=forward_timer,
        )
        forward_timer.save()
        df = self.get_dataframe(data, self.subset).iloc[score_indices.numpy()]
        df["score_indices"] = score_indices.numpy()
        df["scores"] = losses
        df["encoded"] = labels
        df["decoded"] = df["encoded"].apply(data.target_transform.decode)
        self.save_scores(df, run_path)

    def _get_run_name(self, c: DictConfig) -> str:
        template = [
            c.split,
            c.dataset.id,
            c.model.id,
            c.optimizer.id,
            c.learning_rate,
            c.batch_size,
            c.training_type,
            c.iterations,
            c.scheduler.id,
            c.augmentation.id,
            c.seed,
        ]
        return "_".join([str(t) for t in template])

    def postprocess(self, score_id: str, runs: list) -> None:
        base_runs = {r.split("_", 1)[1] for r in runs}
        df = None
        scores = pd.DataFrame()
        for run_name in base_runs:
            run_dfs = []
            matching_runs = [r for r in runs if r.endswith(run_name)]
            for r in matching_runs:
                run_df = pd.read_csv(
                    os.path.join(self.output_directory, r, "scores.csv"),
                    index_col="score_indices",
                )
                run_dfs.append(run_df)
            run_df = pd.concat(run_dfs)
            run_df = run_df.sort_index()
            scores[run_name] = run_df["scores"]
            if df is None:
                df = run_df.drop(columns=["scores"])
        df["mean"] = scores.mean(axis=1)
        df["ranks"] = self.rank_and_normalize(df)
        df.to_csv(
            os.path.join(self.output_directory, score_id + ".csv"), index=False
        )

    def _split_train_dataset(
        self,
        split: int,
        dataset: Dataset,
        dataset_seed: int,
    ) -> Tuple[Subset, Subset, torch.Tensor]:
        dataset_size = len(dataset)
        g = torch.Generator().manual_seed(dataset_seed)
        permuted_indices = torch.randperm(dataset_size, generator=g)
        shuffled_dataset = Subset(dataset, permuted_indices.tolist())
        split_size = dataset_size // self.num_splits
        split_sizes = [split_size] * (self.num_splits - 1) + [
            dataset_size - split_size * (self.num_splits - 1)
        ]
        splits = list(torch.split(permuted_indices, split_sizes))
        score_indices = splits.pop(split)
        train_indices = torch.cat(splits)
        train_subset = Subset(shuffled_dataset, train_indices.tolist())
        score_subset = Subset(shuffled_dataset, score_indices.tolist())
        original_score_indices = permuted_indices[score_indices]
        return train_subset, score_subset, original_score_indices
