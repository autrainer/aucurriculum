import os

import autrainer
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

import aucurriculum


def load_hydra_configuration(directory: str, config_name: str) -> DictConfig:
    """Load a configuration file from the aucurriculum or autrainer
    configuration directories. If the configuration file is not found in the
    aucurriculum or autrainer directories, it will be loaded from the local
    configuration directory if it exists.

    Args:
        directory: Directory where the configuration file is located.
        config_name: Name of the configuration file.

    Raises:
        FileNotFoundError: If the configuration file is not found in the
            aucurriculum, autrainer, or local directories.

    Returns:
        Configuration file.
    """
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    local_path = os.path.join("conf", directory, config_name)
    aucurriculum_path = os.path.join(
        os.path.dirname(aucurriculum.__path__[0]),
        "aucurriculum-configurations",
        directory,
        config_name,
    )
    autrainer_path = os.path.join(
        os.path.dirname(autrainer.__path__[0]),
        "autrainer-configurations",
        directory,
        config_name,
    )
    if os.path.isfile(local_path):
        return OmegaConf.load(local_path)
    elif os.path.isfile(aucurriculum_path):
        return OmegaConf.load(aucurriculum_path)
    elif os.path.isfile(autrainer_path):
        return OmegaConf.load(autrainer_path)
    else:
        raise FileNotFoundError(
            f"Configuration {config_name} not found in {directory}"
        )


class ParallelKNN:
    def __init__(
        self,
        n_neighbors: int,
        batch_size: int,
        device: torch.device,
        progress_bar: bool = True,
    ) -> None:
        """Parallelized torch implementation of sklearn's KNeighborsClassifier
        for GPU usage. The implementation aligns with a KNeighborsClassifier
        using "uniform" as weights, "brute" as algorithm, and "euclidean" as
        distance metric.

        Args:
            n_neighbors: Number of neighbors to use.
            batch_size: Number of samples to process at once during prediction.
            device: CUDA-enabled device to use.
            progress_bar: Whether to display a progress bar during prediction.
                Defaults to True.
        """
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.device = device
        self.progress_bar = progress_bar
        self.x = None
        self.y = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the model to the training data. The model stores the training
        data in memory for prediction.

        Args:
            x: Training samples.
            y: Training labels.
        """
        self.x = x
        self.y = y
        self.num_classes = y.max().item() + 1
        self._x_device = x.device
        self._y_device = y.device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Lazily predict the labels for the input samples with the given batch
        size.

        Args:
            x: Data samples.

        Raises:
            ValueError: If the model has not been fitted yet.

        Returns:
            Prediction labels.
        """
        if self.x is None or self.y is None:
            raise ValueError("Model has not been fitted yet.")

        num_samples = x.shape[0]
        predictions = []

        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)

        with torch.no_grad():
            for i in tqdm(
                range(0, num_samples, self.batch_size),
                disable=not self.progress_bar,
                desc="KNN",
            ):
                x_batch = x[i : i + self.batch_size].to(self.device)
                batch_distances = self.compute_distances(x_batch, self.x)
                _, batch_indices = torch.topk(
                    batch_distances,
                    self.n_neighbors,
                    largest=False,
                    sorted=False,
                )
                batch_labels = self.y[batch_indices]
                preds = self.majority_vote(batch_labels, self.num_classes)
                predictions.append(preds.cpu())

        del x_batch, batch_distances, batch_indices, batch_labels, preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.x = self.x.to(self._x_device)
        self.y = self.y.to(self._y_device)
        return torch.cat(predictions)

    @staticmethod
    def compute_distances(
        x_batch: torch.Tensor,
        x_full: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the pairwise Euclidean distances between two sets of data.

        Args:
            x_batch: Batch of data samples.
            x_full: Full set of training samples.

        Returns:
            Pairwise Euclidean distances.
        """
        dot_prod = torch.mm(x_batch, x_full.T)
        norms_batch = torch.sum(x_batch**2, dim=1, keepdim=True)
        norms_full = torch.sum(x_full**2, dim=1).unsqueeze(0)
        distances = norms_batch + norms_full - 2 * dot_prod
        return torch.sqrt(torch.clamp(distances, min=0))

    @staticmethod
    def majority_vote(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Compute the majority vote for the given labels.

        Args:
            labels: Labels of the nearest neighbors.
            num_classes: Number of classes.

        Returns:
            Predicted labels.
        """
        counts = torch.nn.functional.one_hot(
            labels,
            num_classes=num_classes,
        ).sum(dim=1)
        return torch.argmax(counts, dim=1)
