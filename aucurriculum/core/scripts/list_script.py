import os

import autrainer
from autrainer.core.scripts import ListScript
from autrainer.core.scripts.abstract_script import MockParser
from autrainer.core.scripts.list_script import ListArgs
from autrainer.core.scripts.utils import catch_cli_errors

import aucurriculum
from aucurriculum.core.constants import CurriculumConstants


class ListCurriculumScript(ListScript):
    def _list_global_configs(self, args: ListArgs) -> None:
        if args.directory in CurriculumConstants().CONFIG_DIRS:
            lib_path = os.path.join(
                os.path.dirname(aucurriculum.__path__[0]),
                "aucurriculum-configurations",
            )
        else:
            lib_path = os.path.join(
                os.path.dirname(autrainer.__path__[0]),
                "autrainer-configurations",
            )

        path = os.path.join(lib_path, args.directory)
        self._print_configs(args.directory, path, "global", args.pattern)


@catch_cli_errors
def list_configs(
    directory: str,
    local_only: bool = False,
    global_only: bool = False,
    pattern: str = "*",
) -> None:
    """List local and global configurations.

    Args:
        directory: The directory to list configurations from. Choose from:
            :const:`~autrainer.core.constants.NamingConstants.CONFIG_DIRS`
            and :const:`~aucurriculum.core.constants.CurriculumConstants.CONFIG_DIRS`.
        local_only: List local configurations only. Defaults to False.
        global_only: List global configurations only. Defaults to False.
        pattern: Glob pattern to filter configurations. Defaults to "*".
    """

    script = ListCurriculumScript()
    script.parser = MockParser()
    script.main(ListArgs(directory, local_only, global_only, pattern))
