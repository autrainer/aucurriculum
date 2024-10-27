from typing import List

from autrainer.core.constants import AbstractConstants


class CurriculumConstants(AbstractConstants):
    """Singleton for managing the curriculum configurations of
    `aucurriculum`."""

    _config_dirs = [
        "curriculum",
        "curriculum/pacing",
        "curriculum/sampling",
        "curriculum/scoring",
    ]

    @property
    def CONFIG_DIRS(self) -> List[str]:
        """Get the curriculum configuration directories for Hydra
        configurations. Defaults to :attr:`["curriculum", "curriculum/pacing",
        "curriculum/sampling", "curriculum/scoring"]`.

        Returns:
            Curriculum configuration directories for Hydra configurations.
        """
        return self._config_dirs
