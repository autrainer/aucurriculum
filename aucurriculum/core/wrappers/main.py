import os
from typing import Optional

import autrainer
from autrainer.core.wrappers.main import add_current_directory_to_path
import hydra
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

import aucurriculum


class AucurriculumPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        lib_path = os.path.join(
            os.path.dirname(aucurriculum.__path__[0]),
            "aucurriculum-configurations",
        )
        autrainer_path = os.path.join(
            os.path.dirname(autrainer.__path__[0]),
            "autrainer-configurations",
        )
        search_path.append(provider="aucurriculum-current", path="file://conf")
        search_path.append(
            provider="aucurriculum-configs",
            path=f"file://{lib_path}",
        )
        search_path.append(
            provider="autrainer-configs",
            path=f"file://{autrainer_path}",
        )


def main(
    config_name: str,
    config_path: Optional[str] = None,
    version_base: Optional[str] = None,
):
    """Hydra main decorator with additional `aucurriculum` and `autrainer`
    configs.

    The `conf` directory in the current working directory is always added to
    the search path if it exists.
    The current working directory is also added to the Python path.

    Args:
        config_name: The name of the config (usually the file name without the
            .yaml extension).
        config_path: The config path, a directory where Hydra will search for
            config files. If config_path is None no directory is added to the
            search path. Defaults to None.
        version_base: Hydra version base. Defaults to None.
    """

    Plugins.instance().register(AucurriculumPathPlugin)
    add_current_directory_to_path()
    return hydra.main(
        version_base=version_base,
        config_path=config_path,
        config_name=config_name,
    )


autrainer.main = main
