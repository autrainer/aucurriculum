from hydra_filter_sweeper import AbstractFilter
from omegaconf import DictConfig


class FilterPlaceholderScoringFunction(AbstractFilter):  # pragma: no cover
    def filter(self, config: DictConfig, directory: str) -> bool:
        """Filter out run configurations with placeholder (None) scoring
        functions.

        Args:
            config: Run configuration.
            directory: Directory where the run configuration is stored.

        Returns:
            Whether to filter out the run configuration.
        """
        return config.curriculum.scoring.id == "None"


class FilterPartialCurriculum(AbstractFilter):  # pragma: no cover
    def filter(self, config: DictConfig, directory: str) -> bool:
        """Filter out run configurations with partial curricula.
        A partial curriculum is one where the id is not "None" but one or more
        of the sampling, scoring, and pacing ids are "None".

        Args:
            config: Run configuration.
            directory: Directory where the run configuration is stored.

        Returns:
            Whether to filter out the run configuration.
        """
        c = config.curriculum
        if c.id == "None":
            if any(
                id != "None"
                for id in [c.sampling.id, c.scoring.id, c.pacing.id]
            ):
                return True
            return False
        if any(
            id == "None" for id in [c.sampling.id, c.scoring.id, c.pacing.id]
        ):
            return True
        return False
