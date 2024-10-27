import os
from typing import List, Tuple, Union

from autrainer.core.utils import Timer, set_device
import numpy as np
from omegaconf import DictConfig, ListConfig
from sklearn.svm import SVC
import torch
from torch.utils.data import DataLoader

from .abstract_score import AbstractScore
from .utils import load_hydra_configuration


class TransferTeacher(AbstractScore):
    def __init__(
        self,
        output_directory: str,
        results_dir: str,
        experiment_id: str,
        model: Union[str, List[str]],
        dataset: str,
        subset: str = "train",
    ) -> None:
        """Transfer Teacher scoring function that computes margin to the
        decision boundary of a support vector machine (SVM) trained on the
        embeddings of a pre-trained model for each sample in the dataset as
        described in:
        https://arxiv.org/abs/1904.03626

        Args:
            output_directory: Directory where the scores will be stored.
            results_dir: The directory where the results are stored.
            experiment_id: The ID of the grid search experiment.
            model: Model ID or list of model IDs to use for scoring.
            dataset: Dataset ID to use for scoring.
            subset: Dataset subset to use for scoring in ["train", "dev",
                "test"]. Defaults to "train".
        """
        if isinstance(model, (list, ListConfig)):
            self.model_ids = model
        else:
            self.model_ids = [model]
        self.dataset_id = dataset

        super().__init__(
            output_directory=output_directory,
            results_dir=results_dir,
            experiment_id=experiment_id,
            run_name=None,
            subset=subset,
            reverse_score=True,
        )

    def preprocess(self) -> Tuple[list, list]:
        configs = []
        runs = []
        dataset_config = load_hydra_configuration("dataset", self.dataset_id)
        model_config = [
            load_hydra_configuration("model", m) for m in self.model_ids
        ]

        for m in model_config:
            config = DictConfig({})
            config["dataset"] = dataset_config
            config["model"] = m
            configs.append(config)
            runs.append(m.id + "_" + dataset_config.id)

        return configs, runs

    def run(
        self, config: DictConfig, run_config: DictConfig, run_name: str
    ) -> None:
        run_path = os.path.join(self.output_directory, run_name)
        forward_timer = Timer(run_path, "model_forward")
        svm_timer = Timer(run_path, "svm")
        batch_size = config.get("batch_size", run_config.get("batch_size", 32))

        run_config.augmentation = None
        run_config.seed = 1
        run_config.batch_size = batch_size
        data, model = self.prepare_data_and_model(run_config)
        model.eval()
        dataset = self.get_dataset_subset(data, self.subset)

        self._register_forward_hook(model)
        self.embedding_size = self._get_embedding_size(model, dataset[0][0])
        loader = DataLoader(dataset, batch_size=batch_size)

        outputs, labels = self.forward_pass(
            model=model,
            loader=loader,
            batch_size=batch_size,
            output_map_fn=lambda outs, y: self.embeddings.flatten(1),
            output_size=self.embedding_size,
            tqdm_desc=run_name,
            disable_progress_bar=not config.get("progress_bar", False),
            device=set_device(config.device),
            timer=forward_timer,
        )
        forward_timer.save()
        svm_timer.start()
        scores = self._generate_svm_scores(outputs, labels)
        svm_timer.stop()
        svm_timer.save()
        df = self.create_dataframe(
            scores=scores,
            labels=labels,
            data=data,
        )
        self.save_scores(df, run_path)

    def _register_forward_hook(self, model: torch.nn.Module) -> None:
        self.embeddings = None

        def flatten_layers(layer, layers):
            if len(list(layer.children())) == 0:
                layers.append(layer)
            else:
                for child in layer.children():
                    flatten_layers(child, layers)

        all_layers = []
        flatten_layers(model, all_layers)
        second_to_last_idx = (
            max(
                idx
                for idx, layer in enumerate(all_layers)
                if isinstance(layer, torch.nn.Linear)
            )
            - 1
        )
        second_to_last_layer = all_layers[second_to_last_idx]

        def hook(module, input, output):
            self.embeddings = output

        second_to_last_layer.register_forward_hook(hook)

    def _get_embedding_size(
        self, model: torch.nn.Module, x: torch.Tensor
    ) -> int:
        with torch.no_grad():
            model(x.unsqueeze(0))
        return self.embeddings.flatten().shape[0]

    def _generate_svm_scores(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        svm = SVC(probability=True, kernel="rbf")
        svm.fit(x, y)
        probabilities = svm.predict_proba(x)
        scores = probabilities[np.arange(len(y)), y]
        return scores
