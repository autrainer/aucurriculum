import os
import subprocess
from typing import List, Union
from unittest.mock import patch

from autrainer.core.constants import NamingConstants
from autrainer.core.scripts.utils import CommandLineError
from omegaconf import OmegaConf
import pytest

import aucurriculum.cli
from aucurriculum.core.constants import CurriculumConstants

from .utils import BaseIndividualTempDir


class TestMainEntryPoint(BaseIndividualTempDir):
    @pytest.mark.parametrize(
        "args",
        [
            ["-h"],
            ["-v"],
            ["create", "-e"],
        ],
    )
    def test_main(self, args: List[str]) -> None:
        result = subprocess.run(
            ["aucurriculum"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert (
            result.returncode == 0 and result.stderr == ""
        ), "Should return 0 and no error message."

    def test_no_command(self) -> None:
        result = subprocess.run(
            ["aucurriculum"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert result.returncode == 2, "Should return 2."

    def test_non_existent_command(self) -> None:
        result = subprocess.run(
            ["aucurriculum", "create", "invalid"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert result.returncode == 2, "Should return 2."


class TestCLICreate(BaseIndividualTempDir):
    def test_empty_directory(self) -> None:
        with pytest.raises(
            CommandLineError, match="No configuration directories specified."
        ):
            aucurriculum.cli.create([])

    def test_invalid_directory(self) -> None:
        with pytest.raises(
            CommandLineError,
            match="Invalid configuration directory 'invalid'.",
        ):
            aucurriculum.cli.create(["invalid"])

    @pytest.mark.parametrize(
        "dirs, empty, all",
        [
            (None, True, True),
            (["model"], False, True),
            (["model"], True, False),
        ],
    )
    def test_mutually_exclusive(
        self,
        dirs: Union[List[str], None],
        empty: bool,
        all: bool,
    ) -> None:
        with pytest.raises(
            CommandLineError,
            match="The flags -e/--empty and -a/--all are mutually exclusive",
        ):
            aucurriculum.cli.create(dirs, empty, all)

    def test_force_overwrite(self) -> None:
        os.mkdir("conf")
        with pytest.raises(
            CommandLineError,
            match="Directory 'conf' already exists.",
        ):
            aucurriculum.cli.create(empty=True)
        aucurriculum.cli.create(empty=True, force=True)

    @pytest.mark.parametrize(
        "dirs",
        [
            ["model"],
            ["model", "curriculum/scoring"],
            ["model", "dataset", "optimizer", "scheduler"],
            CurriculumConstants().CONFIG_DIRS,
        ],
    )
    def test_create_directories(self, dirs: List[str]) -> None:
        aucurriculum.cli.create(dirs)
        assert all(
            os.path.exists(f"conf/{directory}") for directory in dirs
        ), "Should create directories."
        assert os.path.exists("conf/config.yaml"), "Should create config.yaml."
        assert os.path.exists(
            "conf/curriculum.yaml"
        ), "Should create curriculum.yaml."

    def test_create_empty(self) -> None:
        aucurriculum.cli.create(empty=True)
        assert os.path.exists("conf/config.yaml"), "Should create config.yaml."
        assert os.listdir("conf") == [
            "config.yaml",
            "curriculum.yaml",
        ], "Should only contain config.yaml and curriculum.yaml."

    def test_create_all(self) -> None:
        aucurriculum.cli.create(all=True)
        assert all(
            os.path.exists(f"conf/{directory}")
            for directory in NamingConstants().CONFIG_DIRS
        ), "Should create all directories."
        assert os.path.exists("conf/config.yaml"), "Should create config.yaml."
        assert os.path.exists(
            "conf/curriculum.yaml"
        ), "Should create curriculum.yaml."


class TestCLIList(BaseIndividualTempDir):
    @pytest.mark.parametrize(
        "local_only, global_only",
        [(True, False), (False, True), (True, True)],
    )
    def test_local_global_configs(
        self,
        capfd: pytest.CaptureFixture,
        local_only: bool,
        global_only: bool,
    ) -> None:
        os.makedirs("conf/curriculum/scoring", exist_ok=True)
        OmegaConf.save({}, "conf/curriculum/scoring/Random.yaml")
        aucurriculum.cli.list(
            "curriculum/scoring",
            local_only=local_only,
            global_only=global_only,
        )
        out, _ = capfd.readouterr()
        if local_only:
            assert (
                "Local 'curriculum/scoring' configurations:" in out
            ), "Should print local configurations."
        if global_only:
            assert (
                "Global 'curriculum/scoring' configurations:" in out
            ), "Should print global configurations."

    def test_local_global_missing_configs(
        self, capfd: pytest.CaptureFixture
    ) -> None:
        os.makedirs("conf/curriculum/scoring", exist_ok=True)
        aucurriculum.cli.list("curriculum/scoring", pattern="Invalid*")
        out, _ = capfd.readouterr()
        assert (
            "No local 'curriculum/scoring' configurations found." in out
        ), "Should not print local configurations."
        assert (
            "No global 'curriculum/scoring' configurations found." in out
        ), "Should not print global configurations."


class TestCLIShow(BaseIndividualTempDir):
    @pytest.mark.parametrize("config", ["Random.yaml", "CELoss.yaml"])
    def test_valid_directory(
        self, capfd: pytest.CaptureFixture, config: str
    ) -> None:
        aucurriculum.cli.show("curriculum/scoring", config)
        out, _ = capfd.readouterr()
        config = config.replace(".yaml", "")
        assert f"id: {config}" in out, "Should print configuration."

    def test_save_configuration(self, capfd: pytest.CaptureFixture) -> None:
        aucurriculum.cli.show("curriculum/scoring", "Random", save=True)
        out, _ = capfd.readouterr()
        assert "id: Random" in out, "Should print configuration."
        assert os.path.exists(
            "conf/curriculum/scoring/Random.yaml"
        ), "Should save configuration."


class TestCLIFetch(BaseIndividualTempDir):
    @pytest.mark.parametrize("cfg_launcher", [False, True])
    def test_launcher_override(
        self, capfd: pytest.CaptureFixture, cfg_launcher: bool
    ) -> None:
        with patch("sys.argv", [""]):
            aucurriculum.cli.fetch(
                cfg_launcher=cfg_launcher,
            )
        out, _ = capfd.readouterr()
        assert "Fetching datasets..." in out, "Should print fetching message."
        assert "Fetching models..." in out, "Should print fetching message."


class TestCLIPreprocess(BaseIndividualTempDir):
    def test_preprocess(self, capfd: pytest.CaptureFixture) -> None:
        with patch("sys.argv", [""]):
            aucurriculum.cli.preprocess(cfg_launcher=True, update_frequency=0)
        out, _ = capfd.readouterr()
        assert (
            "Preprocessing datasets..." in out
        ), "Should print preprocessing message."
