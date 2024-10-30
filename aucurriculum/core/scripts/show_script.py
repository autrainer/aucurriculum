import os

import autrainer
from autrainer.core.scripts import ShowScript
from autrainer.core.scripts.abstract_script import MockParser
from autrainer.core.scripts.show_script import ShowArgs
from autrainer.core.scripts.utils import catch_cli_errors

import aucurriculum
from aucurriculum.core.constants import CurriculumConstants


class ShowCurriculumScript(ShowScript):
    def _get_path(self, args: ShowArgs) -> str:
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

        return os.path.join(
            lib_path,
            args.directory,
            ShowScript._get_config_name(args.config),
        )


@catch_cli_errors
def show(
    directory: str,
    config: str,
    save: bool = False,
    force: bool = False,
) -> None:
    """Show and save a global configuration.

    Args:
        directory: The directory to list configurations from. Choose from:
            :const:`~autrainer.core.constants.NamingConstants.CONFIG_FOLDERS`
            and :const:`~aucurriculum.core.constants.CurriculumConstants.CONFIG_FOLDERS`.
        config: The global configuration to show. Configurations can be
            discovered using the 'autrainer list' command.
        save: Save the global configuration to the local conf directory.
            Defaults to False.
        force: Force overwrite local configuration if it exists in combination
            with save=True. Defaults to False.
    """
    script = ShowCurriculumScript()
    script.parser = MockParser()
    script.main(ShowArgs(directory, config, save, force))
