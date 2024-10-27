from abc import ABC, abstractmethod


class AbstractPace(ABC):
    def __init__(
        self,
        initial_size: float,
        final_iteration: float,
        total_iterations: int,
        dataset_size: int,
    ) -> None:
        """Abstract class for pacing functions.

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
        if not 0 < initial_size <= 1:
            raise ValueError(
                f"Initial size must be in (0, 1], got {initial_size}"
            )
        if not 0 <= final_iteration <= 1:
            raise ValueError(
                f"Final iteration must be in [0, 1], got {final_iteration}"
            )
        self.initial_size = initial_size
        self.final_iteration = final_iteration
        self.total_iterations = total_iterations
        self.dataset_size = dataset_size

    @abstractmethod
    def get_dataset_size(self, iteration: int) -> int:
        """Get the dataset size at a given iteration according to the pacing
        function.

        Args:
            iteration: The iteration number.

        Returns:
            The dataset size at the given iteration.
        """
