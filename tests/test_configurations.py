import importlib
import inspect
import os
from typing import Any, Dict, List, Tuple, Type

from omegaconf import OmegaConf
import pytest


def load_configurations(subdir: str) -> List[Tuple[str, dict]]:
    config_names = [
        f
        for f in os.listdir(f"aucurriculum-configurations/{subdir}")
        if f.endswith(".yaml")
    ]
    assert config_names, f"No {subdir} configurations found."
    configurations = []
    for name in config_names:
        cfg = OmegaConf.to_container(
            OmegaConf.load(f"aucurriculum-configurations/{subdir}/{name}")
        )
        configurations.append((name, cfg))
    return configurations


def get_class_from_import_path(import_path: str) -> Type:
    m, c = import_path.rsplit(".", 1)
    m = importlib.import_module(m)
    assert hasattr(m, c), f"Class '{c}' not found in module '{m}'."
    return getattr(m, c)


def get_required_parameters(
    test_class: Type,
    ignore_params: List[str] = None,
) -> list[str]:
    ignore_params = ignore_params or []
    return [
        k
        for k, v in inspect.signature(test_class.__init__).parameters.items()
        if v.default == inspect.Parameter.empty
        and k not in ["self", "args", "kwargs"] + ignore_params
    ]


class TestConfigurations:
    @pytest.mark.parametrize("name, config", load_configurations("curriculum"))
    def test_curriculum_type(self, name: str, config: Dict[str, Any]) -> None:
        for c in ["id", "type", "short", "_results_dir", "_experiment_id"]:
            assert config.get(c), f"{name} missing {c} in configuration."

    @pytest.mark.parametrize(
        "name, config",
        load_configurations("curriculum/sampling"),
    )
    def test_sampling(self, name: str, config: Dict[str, Any]) -> None:
        assert config.get("id"), f"{name} missing ID in configuration."
        assert config.get("short"), f"{name} missing short in configuration."

    @pytest.mark.parametrize(
        "name, config",
        load_configurations("curriculum/scoring"),
    )
    def test_scoring(self, name: str, config: Dict[str, Any]) -> None:
        self._test_configuration(
            name,
            config,
            ["output_directory", "results_dir", "experiment_id"],
        )

    @pytest.mark.parametrize(
        "name, config",
        load_configurations("curriculum/pacing"),
    )
    def test_pacing(self, name: str, config: Dict[str, Any]) -> None:
        self._test_configuration(
            name,
            config,
            ["total_iterations", "dataset_size"],
        )

    @staticmethod
    def _test_configuration(
        name: str,
        config: Dict[str, Any],
        ignore_params: List[str],
    ) -> None:
        assert config.get("id"), f"{name} missing ID in configuration."
        assert config.get(
            "_target_"
        ), f"{name} missing target in configuration."
        if config["id"] == "None":
            return
        cls = get_class_from_import_path(config["_target_"])
        required_params = get_required_parameters(cls, ignore_params)
        for arg in required_params:
            assert arg in config, f"{name} missing {arg} in configuration."
