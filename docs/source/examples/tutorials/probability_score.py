import os

from autrainer.core.utils import Timer, set_device
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from aucurriculum.curricula.scoring import AbstractScore


class ProbabilityScore(AbstractScore):
    def __init__(
        self,
        output_directory: str,
        results_dir: str,
        experiment_id: str,
        run_name: str,
        stop: str = "best",
        subset: str = "train",
    ) -> None:
        """Probability scoring function determining the difficulty of a sample
        based on the model's highest output probability (regardless of the true
        class).

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
            reverse_score=True,  # assume higher probabilities are easier
        )

    def run(
        self, config: DictConfig, run_config: DictConfig, run_name: str
    ) -> None:
        run_name, full_run_name = self.split_run_name(run_name)
        run_path = os.path.join(self.output_directory, full_run_name)
        data, model = self.prepare_data_and_model(run_config)
        dataset = self.get_dataset_subset(data, self.subset)
        batch_size = config.get("batch_size", run_config.get("batch_size", 32))
        loader = DataLoader(dataset, batch_size=batch_size)
        self.load_model_checkpoint(model, run_name)
        device = set_device(config.device)
        forward_timer = Timer(run_path, "model_forward")
        probabilities, labels = self.forward_pass(
            model=model,
            loader=loader,
            batch_size=batch_size,
            output_map_fn=self.score,
            tqdm_desc=run_name,
            disable_progress_bar=not config.get("progress_bar", False),
            device=device,
            timer=forward_timer,
        )
        forward_timer.save()
        df = self.create_dataframe(probabilities, labels, data)
        self.save_scores(df, run_path)

    def score(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the highest probability per sample regardless of the true
        class.

        Args:
            outputs: Batch of model outputs.
            y: Batch of labels.

        Returns:
            Batch of highest probability per sample.
        """
        return torch.softmax(outputs, dim=1).max(dim=1).values
