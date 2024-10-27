import os
from typing import List, Tuple

from autrainer.postprocessing import AggregateGrid
from autrainer.postprocessing.postprocessing_utils import save_yaml
from omegaconf import DictConfig, OmegaConf


class AggregateCurriculum(AggregateGrid):
    def _aggregate_config(self, agg_name: str, run_list: list) -> None:
        path = os.path.join(self.output_directory, agg_name, ".hydra")
        os.makedirs(path, exist_ok=True)
        runs = [
            OmegaConf.load(
                os.path.join(
                    self.training_directory, r, ".hydra", "config.yaml"
                )
            )
            for r in run_list
        ]
        config = runs.pop(0)
        if self.aggregated_dict is None:
            for key in self.aggregate_list:
                c, r, k = self._traverse_nested_keys(key, config, runs)
                c[k] = self._replace_differing_values(c, r, k)
        save_yaml(
            os.path.join(path, "config.yaml"), OmegaConf.to_container(config)
        )

    @staticmethod
    def _traverse_nested_keys(
        key: str,
        config: DictConfig,
        runs: List[DictConfig],
    ) -> Tuple[DictConfig, List[DictConfig], str]:
        if "." not in key:
            return config, runs, key
        for subkey in key.split(".")[:-1]:
            config = config[subkey]
            runs = [r[subkey] for r in runs]
        return config, runs, key.split(".")[-1]

    def _replace_differing_values(
        self,
        config: DictConfig,
        runs: List[DictConfig],
        key: str,
    ) -> DictConfig:
        if not runs:
            return config[key]

        if not isinstance(config[key], DictConfig):
            return "#"

        # can no longer assume that all keys are present in all runs
        for k in config[key].keys():
            if not isinstance(config[key][k], DictConfig):
                values = [r[key].get(k) for r in runs + [config]]
                if len(set(values)) > 1:
                    config[key][k] = "#"
            else:
                if any(r.get(key) is None for r in runs):
                    return "#"
                config[key][k] = self._replace_differing_values(
                    config[key], [r[key] for r in runs], k
                )

        return config[key]
