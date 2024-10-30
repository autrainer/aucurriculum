import numpy as np

from .abstract_pace import AbstractPace


class Logarithmic(AbstractPace):
    def __init__(
        self,
        initial_size: float,
        final_iteration: float,
        total_iterations: int,
        dataset_size: int,
    ) -> None:
        """Logarithmic pacing function adapted from:
        https://arxiv.org/abs/2012.03107

        Args:
            initial_size: The initial fraction of the dataset to start training
                with.
            final_iteration: The fraction of training iterations at which the
                dataset size will be the full dataset size.
            total_iterations: The total number of training iterations.
            dataset_size: The size of the dataset.

        Raises:
            ValueError: If the initial size is not in (0, 1] or if the final
                iteration is not in [0, 1].
        """
        super().__init__(
            initial_size,
            final_iteration,
            total_iterations,
            dataset_size,
        )

    def get_dataset_size(self, iteration: int) -> int:
        if self.initial_size == 1 or self.final_iteration == 0:
            return self.dataset_size
        loga = int(
            self.dataset_size * self.initial_size
            + (self.dataset_size * (1 - self.initial_size))
            * (
                1
                + 0.1
                * np.log(
                    (
                        iteration
                        / (self.final_iteration * self.total_iterations)
                    )
                    + np.exp(-10)
                )
            )
        )
        return min(loga, self.dataset_size)
