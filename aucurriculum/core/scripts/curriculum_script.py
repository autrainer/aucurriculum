from typing import Optional

import autrainer
from autrainer.core.scripts import CommandLineError
from autrainer.core.scripts.abstract_script import AbstractScript, MockParser
from autrainer.core.scripts.utils import (
    add_hydra_args_to_sys,
    catch_cli_errors,
    run_hydra_cmd,
    running_in_notebook,
)
from omegaconf import DictConfig, OmegaConf


class CurriculumScript(AbstractScript):
    def __init__(self) -> None:
        super().__init__(
            "curriculum",
            "Launch a curriculum scoring function configuration (Hydra).",
            extended_description=(
                "For more information on Hydra's command line line flags, see:\n"
                "https://hydra.cc/docs/advanced/hydra-command-line-flags/."
            ),
            epilog="Example: aucurriculum curriculum -cn curriculum.yaml",
            unknown_args=True,
        )

    def main(self, args: dict) -> None:
        @autrainer.main("curriculum")
        def main(cfg: DictConfig) -> None:
            import hydra

            OmegaConf.set_struct(cfg, False)
            OmegaConf.resolve(cfg)
            output_directory = (
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )
            if cfg.curriculum.scoring.id == "None":
                raise CommandLineError(
                    MockParser(), "No scoring function provided."
                )

            from aucurriculum.curricula import CurriculumScoreManager

            cs = CurriculumScoreManager(cfg, output_directory)
            configs, runs = cs.preprocess()

            for config, run in zip(configs, runs):
                cs.run(config, run)

            cs.postprocess(
                score_id=cfg.curriculum.scoring.id,
                correlation=cfg.correlation,
            )

        main()


@catch_cli_errors
def curriculum(
    override_kwargs: Optional[dict] = None,
    config_name: str = "curriculum",
    config_path: Optional[str] = None,
) -> None:
    """Launch a curriculum scoring function configuration.

    Args:
        override_kwargs: Additional Hydra override arguments to pass to the
            train script.
        config_name: The name of the config (usually the file name without the
            .yaml extension). Defaults to "curriculum".
        config_path: The config path, a directory where Hydra will search for
            config files. If config_path is None no directory is added to the
            search path. Defaults to None.
    """
    if running_in_notebook():
        run_hydra_cmd(
            "curriculum",
            override_kwargs,
            config_name,
            config_path,
            cmd_prefix="aucurriculum",
        )
    else:
        add_hydra_args_to_sys(override_kwargs, config_name, config_path)
        script = CurriculumScript()
        script.parser = MockParser()
        script.main({})
