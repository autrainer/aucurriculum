import argparse
from typing import List

from autrainer.core.scripts import (
    AbstractScript,
    CommandLineError,
    DeleteFailedScript,
    DeleteStatesScript,
    FetchScript,
    InferenceScript,
    PreprocessScript,
)
from autrainer.core.scripts.cli import CLI as AutrainerCLI

import aucurriculum
from aucurriculum.core.scripts import (
    CreateCurriculumScript,
    CurriculumScript,
    GroupCurriculumScript,
    ListCurriculumScript,
    PostprocessCurriculumScript,
    ShowCurriculumScript,
    TrainCurriculumScript,
)


class CLI(AutrainerCLI):
    def __init__(self) -> None:
        super().__init__()
        self._update_metadata()

    def _init_scripts(self) -> List[AbstractScript]:
        return [
            # configuration management
            CreateCurriculumScript(),
            ListCurriculumScript(),
            ShowCurriculumScript(),
            # preprocessing
            FetchScript(),
            PreprocessScript(),
            # training
            TrainCurriculumScript(),
            # inference
            InferenceScript(),
            # postprocessing
            PostprocessCurriculumScript(),
            DeleteFailedScript(),
            DeleteStatesScript(),
            GroupCurriculumScript(),
            # curriculum learning
            CurriculumScript(),
        ]

    def _update_metadata(self) -> None:
        self.parser.prog = "aucurriculum"
        self.parser.description = (
            "A Curriculum Learning Toolkit for Deep Learning Tasks "
            "built on top of autrainer."
        )
        for action in self.parser._actions:
            if isinstance(action, argparse._VersionAction):
                action.version = f"%(prog)s {aucurriculum.__version__}"


def main() -> None:
    cli = CLI()
    try:
        cli.main()
    except CommandLineError as e:
        e.handle()


def get_parser() -> argparse.ArgumentParser:
    cli = CLI()  # pragma: no cover
    return cli.parser  # pragma: no cover


if __name__ == "__main__":
    main()
