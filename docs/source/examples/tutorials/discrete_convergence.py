from typing import TYPE_CHECKING

from aucurriculum.curricula.pacing import AbstractPace


if TYPE_CHECKING:
    from autrainer.training import ModularTaskTrainer


class DiscreteConvergence(AbstractPace):
    def __init__(
        self,
        initial_size: float,
        final_iteration: float,
        total_iterations: int,
        dataset_size: int,
        patience: int = 1,
        min_improvement: float = 0.0,
        buckets: int = 10,
    ) -> None:
        super().__init__(
            initial_size, final_iteration, total_iterations, dataset_size
        )
        """Discrete convergence pacing function adding a new bucket of training
        data every time the validation performance of the tracking metric does
        not improve by at least `min_improvement` for `patience` iterations.

        Args:
            initial_size: The initial fraction of the dataset to start training
                with.
            final_iteration: The fraction of training iterations at which the
                dataset size will be the full dataset size. If not all buckets
                are introduced by this iteration, the remaining buckets will be
                added immediately.
            total_iterations: The total number of training iterations.
            dataset_size: The size of the dataset.
            patience: The number of iterations to wait before adding a new
                bucket of training data. Defaults to 1.
            min_improvement: The minimum improvement in the tracking metric to
                consider as an improvement. Defaults to 0.0.
            buckets: The number of buckets to divide the remaining dataset size
                into. Defaults to 10.
        """
        if patience < 1:
            raise ValueError(f"patience {patience} must be a positive integer")
        self.patience = patience
        if min_improvement < 0:
            raise ValueError(
                f"min_improvement {min_improvement} must be a positive float"
            )
        self.min_improvement = min_improvement
        if buckets < 1:
            raise ValueError(f"buckets {buckets} must be a positive integer")
        self.bucket_size = int((1 - initial_size) * dataset_size / buckets)
        self.current_size = int(initial_size * dataset_size)
        self.current_wait = 0

    def get_dataset_size(self, iteration: int) -> int:
        if self.total_iterations * self.final_iteration <= iteration:
            return self.dataset_size
        return self.current_size

    def convergence_criterion(self, metric: float) -> None:
        if self.metric_fn.compare(
            metric, self.current_best + self.min_improvement
        ):
            self.current_best = metric
            self.current_wait = 0
            return

        self.current_wait += 1
        if self.current_wait >= self.patience:
            size = min(self.current_size + self.bucket_size, self.dataset_size)
            self.current_size = size
            self.current_wait = 0

    def cb_on_train_begin(self, trainer: "ModularTaskTrainer") -> None:
        self.metric_fn = trainer.data.tracking_metric
        self.current_best = self.metric_fn.starting_metric
        if self.metric_fn.suffix == "min":
            self.min_improvement = -self.min_improvement

    def cb_on_val_end(
        self, trainer: "ModularTaskTrainer", iteration: int, val_results: dict
    ) -> None:
        self.convergence_criterion(val_results[self.metric_fn.name])
