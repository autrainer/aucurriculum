import os
import shutil
from typing import List, Optional

from autrainer.core.constants import NamingConstants
from autrainer.core.scripts import CreateScript
from autrainer.core.scripts.abstract_script import MockParser
from autrainer.core.scripts.create_script import CreateArgs
from autrainer.core.scripts.utils import catch_cli_errors

import aucurriculum


class CreateCurriculumScript(CreateScript):
    def __init__(self) -> None:
        super().__init__()
        self.description = (
            "Create a new curriculum project with default configurations."
        )

    def _create_directory(self, args: CreateArgs) -> None:
        directories = []
        if args.empty:
            directories.append("conf")
        elif args.all:
            directories.extend(
                f"conf/{directory}"
                for directory in NamingConstants().CONFIG_DIRS
            )
        else:
            directories.extend(
                f"conf/{directory}" for directory in args.directories
            )

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _create_default_config(self) -> None:
        lib_path = os.path.join(
            os.path.dirname(aucurriculum.__path__[0]),
            "aucurriculum-configurations",
        )

        cfg_src = os.path.join(lib_path, "config.yaml")
        cfg_dst = os.path.join("conf", "config.yaml")
        curr_src = os.path.join(lib_path, "curriculum.yaml")
        curr_dst = os.path.join("conf", "curriculum.yaml")

        shutil.copy(cfg_src, cfg_dst)
        shutil.copy(curr_src, curr_dst)


@catch_cli_errors
def create(
    directories: Optional[List[str]] = None,
    empty: bool = False,
    all: bool = False,
    force: bool = False,
) -> None:
    """Create a new curriculum project with default configurations.

    Args:
        directories: Configuration directories to create. One or more of:
            :const:`~autrainer.core.constants.NamingConstants.CONFIG_DIRS`
            and :const:`~aucurriculum.core.constants.CurriculumConstants.CONFIG_DIRS`.
            Defaults to None.
        empty: Create an empty project without any configuration directory.
            Defaults to False.
        all: Create a project with all configuration directories.
            Defaults to False.
        force: Force overwrite if the configuration directory already exists.
            Defaults to False.
    """

    script = CreateCurriculumScript()
    script.parser = MockParser()
    script.main(CreateArgs(directories, empty, all, force))
