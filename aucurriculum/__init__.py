from autrainer.core.constants import (
    ExportConstants,
    NamingConstants,
)

from aucurriculum.core.constants import CurriculumConstants
from aucurriculum.core.wrappers import main


# always overlay the autrainer constants
NamingConstants().CONFIG_DIRS = (
    NamingConstants().CONFIG_DIRS + CurriculumConstants().CONFIG_DIRS
)
NamingConstants().NAMING_CONVENTION = NamingConstants().NAMING_CONVENTION[
    :-1
] + [
    "curriculum",
    "curriculum.sampling",
    "curriculum.scoring",
    "curriculum.pacing",
    "curriculum.pacing.initial_size",
    "curriculum.pacing.final_iteration",
    "seed",
]
NamingConstants().VALID_AGGREGATIONS = list(
    set(NamingConstants().NAMING_CONVENTION)
    - set(NamingConstants().INVALID_AGGREGATIONS)
)
ExportConstants().LOGGING_DEPTH = 3
ExportConstants().IGNORE_PARAMS = ExportConstants().IGNORE_PARAMS + [
    "curriculum.short",
    "curriculum.sampling.short",
]

__version__ = "0.1.0"
__all__ = ["main"]
