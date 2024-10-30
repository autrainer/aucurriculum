from autrainer.core.scripts import (
    inference,
    rm_failed,
    rm_states,
)

from aucurriculum.core.scripts import (
    create,
    curriculum,
    fetch,
    group,
    postprocess,
    preprocess,
    show,
    train,
)
from aucurriculum.core.scripts import list_configs as list


__all__ = [
    "create",
    "curriculum",
    "fetch",
    "group",
    "inference",
    "list",
    "postprocess",
    "preprocess",
    "rm_failed",
    "rm_states",
    "show",
    "train",
]
