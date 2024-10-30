from copy import deepcopy
import hashlib
import os
import re
from typing import Dict, List, Tuple, Union
import warnings

from autrainer.core.utils import Timer, set_device
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
import pandas as pd
import torch
import torch.fx
from torch.utils.data import DataLoader
from tqdm import tqdm

from .abstract_score import AbstractScore
from .utils import ParallelKNN


class PredictionDepth(AbstractScore):
    def __init__(
        self,
        output_directory: str,
        results_dir: str,
        experiment_id: str,
        run_name: str,
        probe_placements: Union[List[str], Dict[str, List[str]]],
        max_embedding_size: int = None,
        match_dimensions: bool = False,
        knn_n_neighbors: int = 30,
        knn_batch_size: int = 1024,
        save_embeddings: bool = False,
        stop: str = "best",
        subset: str = "train",
    ) -> None:
        """Prediction Depth scoring function computing the depth at which the
        first and all subsequent KNN probes align with the model's prediction
        for each sample in the dataset individually as described in:
        https://arxiv.org/abs/2106.09647

        Args:
            output_directory: Directory where the scores will be stored.
            results_dir: The directory where the results are stored.
            experiment_id: The ID of the grid search experiment.
            run_name: Name or list of names of the runs to score. Runs can be
                single runs or aggregated runs.
            probe_placements: Names of the nodes in the traced model graph
                where the probes should be placed, specified using regex
                patterns. The input and output of the model are automatically
                added. If a list is provided, the same placements will be used
                for all runs. If a dictionary is provided, the placements will
                be used for the corresponding run names.
            max_embedding_size: Maximum dimensionality of the flattened
                embeddings. If embeddings exceed this size, they will be
                pooled. Defaults to None.
            match_dimensions: Whether to match the spatial dimensions of the
                embeddings and create square embeddings. Defaults to False.
            knn_n_neighbors: Number of neighbors to use for the parallel
                k-nearest neighbors algorithm. Defaults to 30.
            knn_batch_size: Batch size for the parallel k-nearest neighbors
                algorithm. Defaults to 1024.
            save_embeddings: Whether to save the embeddings for each probe.
                Defaults to False.
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
        if isinstance(probe_placements, (DictConfig, ListConfig)):
            probe_placements = OmegaConf.to_container(probe_placements)
        self.probe_placements = probe_placements
        self.max_embedding_size = max_embedding_size
        self.match_dimensions = match_dimensions
        self.knn_n_neighbors = knn_n_neighbors
        self.knn_batch_size = knn_batch_size
        self.save_embeddings = save_embeddings

    def _preprocess_single(self, run_name: str) -> Tuple[list, list]:
        configs, runs = super()._preprocess_single(run_name)
        if isinstance(self.probe_placements, dict):
            p = self.probe_placements.get(run_name)
            if p is None:
                raise ValueError(
                    f"Could not find probe placements for run_name={run_name}"
                )
        else:
            p = self.probe_placements
        _n = "".join(
            sorted(p)
            + [
                str(self.max_embedding_size),
                str(self.match_dimensions),
                str(self.knn_n_neighbors),
            ]
        )
        _hash = hashlib.sha256(_n.encode()).hexdigest()[:8]  # for brevity
        runs = [f"{r}_{_hash}" for r in runs]
        return configs, runs

    def run(
        self, config: DictConfig, run_config: DictConfig, run_name: str
    ) -> None:
        run_name, full_run_name = "_".join(run_name.split("_")[:-2]), run_name
        run_path = os.path.join(self.output_directory, full_run_name)
        self.forward_timer = Timer(run_path, "model_forward")
        self.knn_timer = Timer(run_path, "knn_fit")
        self.disable_progress_bar = not config.get("progress_bar", False)
        batch_size = config.get("batch_size", run_config.get("batch_size", 32))

        data, model = self.prepare_data_and_model(run_config)
        self.load_model_checkpoint(model=model, run_name=run_name)
        dataset = self.get_dataset_subset(data, self.subset)
        loader = DataLoader(dataset, batch_size=batch_size)
        self.device = set_device(config.device)

        model = self._trace_model(model)
        exact_probe_placements = self._create_probe_placements(model, run_name)
        self.num_probes = len(exact_probe_placements)

        predictions: List[pd.DataFrame] = []

        for idx, placement in enumerate(exact_probe_placements):
            self.x, self.y = self._extract_embeddings(
                model, loader, idx, placement
            )
            if idx == len(exact_probe_placements) - 1:
                self.x = torch.softmax(self.x, dim=1)  # softmax output layer

            if self.save_embeddings:
                torch.save(
                    self.x, os.path.join(run_path, f"embeddings_{idx}.pt")
                )

            predictions.append(
                pd.DataFrame(
                    data=self._predict_knn_probe().numpy(),
                    columns=[f"probe_{idx}"],
                )
            )

            if idx < len(exact_probe_placements) - 1:
                del self.x, self.y

        model_preds = torch.argmax(self.x, dim=1).numpy()
        df = self._calculate_prediction_depth(predictions, model_preds)
        df["encoded"] = self.y.numpy()
        df["decoded"] = df["encoded"].apply(data.target_transform.decode)

        self.forward_timer.save()
        self.knn_timer.save()
        df = pd.concat([self.get_dataframe(data, self.subset), df], axis=1)
        self.save_scores(df, run_path)

    def _trace_model(self, model: torch.nn.Module) -> torch.fx.GraphModule:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return torch.fx.symbolic_trace(model)

    def _create_probe_placements(
        self,
        model: torch.fx.GraphModule,
        run_name: str,
    ) -> List[str]:
        if isinstance(self.probe_placements, dict):
            placements = self.probe_placements.get(run_name)
            if placements is None:
                raise ValueError(
                    f"Could not find probe placements for run_name={run_name}"
                )
        else:
            placements = self.probe_placements

        self.first_node = None
        self.last_node = None
        for node in model.graph.nodes:
            node: torch.fx.Node
            if node.op == "placeholder":
                self.first_node = node.name
            if node.op == "output":
                self.last_node = node.name

        if self.first_node is None or self.last_node is None:
            raise ValueError(
                "Could not find input and output nodes in the model graph."
            )

        placements += [
            self.first_node,
            self.last_node,
        ]  # input and output layers
        patterns = [re.compile(p) for p in placements]

        probe_placements = []
        for node in model.graph.nodes:
            node: torch.fx.Node
            if any(p.match(node.name) for p in patterns):
                probe_placements.append(node.name)

        if len(probe_placements) < 2:
            raise ValueError(
                f"Could not find any nodes matching the specified "
                f"placements: {placements}."
            )
        return probe_placements

    def _extract_embeddings(
        self,
        model: torch.fx.GraphModule,
        loader: DataLoader,
        index: int,
        placement: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        model = self._prune_graph_until_node(
            model,
            placement,
            self.first_node,
            self.last_node,
        )
        model.to(self.device)

        _sample_input = next(iter(loader))[0].to(self.device)
        _out = model(_sample_input)
        _batch_size = _sample_input.shape[0]
        _ds_size = len(loader.dataset)
        pool = self._get_pooling_layer(_out.shape)
        _expected_size = pool(_out).flatten(1).shape[-1]

        outputs = torch.zeros((_ds_size, _expected_size), dtype=torch.float32)
        targets = torch.zeros(_ds_size, dtype=torch.long)
        self.forward_timer.start()
        with torch.no_grad():
            for idx, (x, y, _) in enumerate(
                tqdm(
                    loader,
                    disable=self.disable_progress_bar,
                    desc=f"Forward {index+1}/{self.num_probes}",
                )
            ):
                _lower = idx * _batch_size
                _upper = min(_ds_size, (idx + 1) * _batch_size)
                out = model(x.to(self.device))
                outputs[_lower:_upper] = pool(out).flatten(1).cpu()
                targets[_lower:_upper] = y
        self.forward_timer.stop()
        return outputs, targets

    @staticmethod
    def _prune_graph_until_node(
        model: torch.fx.GraphModule,
        placement: str,
        input_layer: str,
        output_layer: str,
    ) -> None:
        if placement == input_layer:
            return torch.nn.Identity()
        if placement == output_layer:
            return model

        gm = deepcopy(model)
        found_node = None
        nodes_to_remove = []

        for node in gm.graph.nodes:
            node: torch.fx.Node
            if node.name == placement:
                found_node = node
            elif found_node:
                nodes_to_remove.append(node)

        if found_node is None:
            raise ValueError(f"Could not find node {placement} in the graph.")

        for node in nodes_to_remove[::-1]:
            gm.graph.erase_node(node)

        gm.graph.output(found_node)
        gm.recompile()
        return gm

    def _get_pooling_layer(self, size: torch.Size) -> torch.nn.Module:
        if not hasattr(self, "_pooling_cache"):
            self._pooling_cache = {}
        if tuple(size) not in self._pooling_cache:
            self._pooling_cache[tuple(size)] = self._create_pooling_layer(size)
        return self._pooling_cache[tuple(size)]

    def _create_pooling_layer(self, size: torch.Size) -> torch.nn.Module:
        if len(size) == 2:  # vector
            if self.max_embedding_size is None:
                return torch.nn.Identity()
            return torch.nn.AdaptiveAvgPool1d(
                min(size[-1], self.max_embedding_size)
            )

        if len(size) == 3:  # sequence
            _, x, y = size
            c = 1
        elif len(size) == 4:  # multi-channel
            _, c, x, y = size
        else:
            raise ValueError(f"Unsupported input size: {tuple(size)}.")

        if self.max_embedding_size is None:
            if self.match_dimensions:
                return torch.nn.AdaptiveAvgPool2d(min(x, y))
            return torch.nn.Identity()

        if self.match_dimensions:
            x = y = min(x, y)

        while c * x * y > self.max_embedding_size:
            if x * y == 1:
                raise ValueError(
                    f"Could not find a valid embedding size for {tuple(size)}."
                )
            x = max(x - 1, 1)
            y = max(y - 1, 1)
        return torch.nn.AdaptiveAvgPool2d((x, y))

    def _predict_knn_probe(self) -> torch.Tensor:
        knn = ParallelKNN(
            n_neighbors=self.knn_n_neighbors,
            batch_size=self.knn_batch_size,
            device=self.device,
            progress_bar=not self.disable_progress_bar,
        )

        self.knn_timer.start()
        knn.fit(self.x, self.y)
        predictions = knn.predict(self.x)
        self.knn_timer.stop()
        return predictions

    def _calculate_prediction_depth(
        self,
        probe_predictions: List[pd.DataFrame],
        model_predictions: np.ndarray,
    ) -> pd.DataFrame:
        df = pd.concat(probe_predictions, axis=1)

        # assign max score if the final probe is incorrect
        scores = len(df.columns) - (
            df.iloc[:, ::-1]
            .eq(model_predictions, axis=0)
            .cumprod(axis=1)
            .sum(axis=1)
        )
        df["predictions"] = model_predictions
        df["scores"] = scores / len(df.columns)  # normalize
        return df
