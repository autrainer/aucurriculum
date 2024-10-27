from functools import cached_property
import os
from typing import TYPE_CHECKING, Iterable, List

import autrainer
from autrainer.criterions import BalancedCrossEntropyLoss
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from aucurriculum.core.plotting import CurriculumPlots

from .pacing import AbstractPace


if TYPE_CHECKING:  # pragma: no cover
    from autrainer.training import ModularTaskTrainer


class CurriculumPaceManager:
    def setup(self, trainer: "ModularTaskTrainer") -> None:
        """Setup the curriculum pace manager to control the training dataset
        size and order based on the scoring function and pacing function.

        Args:
            trainer: The trainer instance.
        """
        scoring_path = os.path.join(
            trainer.cfg.curriculum._results_dir,
            trainer.cfg.curriculum._experiment_id,
            "curriculum",
            trainer.cfg.curriculum.scoring.type,
            trainer.cfg.curriculum.scoring.id + ".csv",
        )
        self.data = trainer.data
        trainer.train_loader = self.train_loader
        self.train_dataset_size = len(self.data.train_dataset)
        self.curriculum_type = trainer.cfg.curriculum.type
        self.sampling = trainer.cfg.curriculum.sampling.id
        self.generator = torch.Generator()
        self.generator.manual_seed(self.data.seed)

        self.pacing_function = autrainer.instantiate(
            config=trainer.cfg.curriculum.pacing,
            instance_of=AbstractPace,
            total_iterations=trainer.cfg.iterations,
            dataset_size=self.train_dataset_size,
        )

        trainer.callback_manager.register(self.pacing_function)
        if hasattr(self.pacing_function, "cb_on_train_begin"):
            self.pacing_function.cb_on_train_begin(trainer)

        self.scores = pd.read_csv(scoring_path)
        self.scores.sort_values(
            by="ranks",
            ascending=self.curriculum_type == "Curriculum",
            inplace=True,
        )
        self.iteration = 0
        self.size = self.pacing_function.get_dataset_size(self.iteration)
        self.pace = pd.DataFrame(
            columns=["iteration", "dataset_size"]
        ).set_index("iteration")
        self.pace.loc[self.iteration] = self.size
        self.plot_config = trainer.cfg.plotting

    def shuffle_indices(self) -> List[int]:
        """Shuffle the indices of the dataset based on the current size.

        Returns:
            Shuffled indices of the dataset.
        """
        indices = self._get_indices(self.size)
        perm = torch.randperm(
            len(indices),
            generator=self.generator,
        )
        return indices[perm].tolist()

    def calculate_weight(self) -> torch.Tensor:
        """Calculate the weight for each class based on the current dataset
        size.

        Returns:
            Weight for each class.
        """
        frequency = (
            self.scores.loc[self._get_indices(self.size)]["encoded"]
            .value_counts()
            .sort_index()
            .values
        )
        weight = torch.tensor(1 / frequency, dtype=torch.float32)
        weight /= weight.sum()
        return weight

    @cached_property
    def train_loader(self) -> DataLoader:
        """Create a DataLoader for the training dataset which samples based on
        the current dataset size and difficulty ordering.

        Returns:
            The DataLoader for the training dataset.
        """
        return DataLoader(
            self.data.train_dataset,
            batch_size=self.data.batch_size,
            generator=self.data.train_loader.generator,
            sampler=CurriculumSampler(self),
        )

    def log_metrics(
        self, trainer: "ModularTaskTrainer", iteration: int
    ) -> None:
        """Log the current dataset size to the loggers.

        Args:
            trainer: The trainer instance.
            iteration: Current iteration.
        """
        self.pace.loc[iteration] = self.size
        for logger in trainer.loggers:
            logger.log_and_update_metrics(
                self.pace.loc[iteration].to_dict(), iteration
            )

    def cb_on_loader_exhausted(
        self, trainer: "ModularTaskTrainer", iteration: int
    ) -> None:
        """Callback to update the dataset size and weight after the loader is
        exhausted. The dataset size update is postponed to this point to avoid
        changing the dataset size during an iteration and ensure that each
        available sample is used for training at least once.

        Args:
            trainer: The trainer instance.
            iteration: Current iteration.
        """
        self.iteration = iteration
        self.size = self.pacing_function.get_dataset_size(self.iteration)
        if self.iteration >= trainer.cfg.iterations:
            return
        self.pace.loc[self.iteration] = self.size
        if isinstance(trainer.criterion, BalancedCrossEntropyLoss):
            trainer.criterion.weight = self.calculate_weight().to(
                trainer.DEVICE
            )
        self.log_metrics(trainer, self.iteration)

    def cb_on_train_begin(self, trainer: "ModularTaskTrainer") -> None:
        """Callback to setup the curriculum pace manager at the beginning of
        the training and overwrite the criterion weight if it is a
        BalancedCrossEntropyLoss.

        Args:
            trainer: The trainer instance.
        """
        self.setup(trainer)

        if isinstance(trainer.criterion, BalancedCrossEntropyLoss):
            trainer.criterion.weight = self.calculate_weight().to(
                trainer.DEVICE
            )
        self.log_metrics(trainer, self.iteration)

    def cb_on_train_end(self, trainer: "ModularTaskTrainer") -> None:
        """Callback to save the final dataset size and plot the training
        progress.

        Args:
            trainer: The trainer instance.
        """
        i = trainer.cfg.iterations
        if self.pace["dataset_size"].get(i, None) is None:
            self.pace.loc[i] = self.size
            self.log_metrics(trainer, i)
        trainer.bookkeeping.save_results_df(
            self.pace.reset_index(), "pace.csv"
        )
        output_path = os.path.join(trainer.output_directory, "_plots")
        cp = CurriculumPlots(
            output_directory=output_path,
            training_type=trainer.cfg.training_type,
            **self.plot_config,
        )
        cp.plot_run(self.pace)

    def _get_indices(self, size: int) -> torch.Tensor:
        if size >= self.train_dataset_size:
            return self.scores.index.values
        if self.sampling == "Unbalanced":
            indices = self.scores.head(size).index.values
        elif self.sampling == "Balanced":
            indices = (
                self.scores.groupby("encoded")
                .head(size // self.data.output_dim)
                .index.values
            )
        elif self.sampling == "Original":
            distribution = self.scores["encoded"].value_counts(normalize=True)
            size_by_class = (distribution * size).round().astype(int)
            indices = pd.concat(
                [
                    self.scores[self.scores["encoded"] == l].head(c)
                    for l, c in size_by_class.items()
                ]
            ).index.values
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling}")

        if len(indices) == 0:
            raise ValueError(
                (
                    f"Unable to sample a training subset of size {size} "
                    f"with {self.sampling} sampling and initial dataset "
                    f"fraction of {self.pacing_function.initial_size}"
                )
            )
        return torch.from_numpy(indices)


class CurriculumSampler(Sampler):
    def __init__(self, manager: CurriculumPaceManager) -> None:
        super().__init__()
        self.manager = manager

    def __iter__(self) -> Iterable[int]:
        return iter(self.manager.shuffle_indices())

    def __len__(self) -> int:
        return self.manager.size
