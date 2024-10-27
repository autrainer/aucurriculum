from typing import Optional

from autrainer.core.scripts import FetchScript, PreprocessScript
from autrainer.core.scripts.abstract_script import MockParser
from autrainer.core.scripts.fetch_script import PreprocessArgs as FetchArgs
from autrainer.core.scripts.preprocess_script import PreprocessArgs
from autrainer.core.scripts.utils import (
    add_hydra_args_to_sys,
    catch_cli_errors,
    run_hydra_cmd,
    running_in_notebook,
)


@catch_cli_errors
def fetch(
    override_kwargs: Optional[dict] = None,
    cfg_launcher: bool = False,
    config_name: str = "config",
    config_path: Optional[str] = None,
) -> None:
    """Fetch the datasets and models specified in a training configuration.

    Args:
        override_kwargs: Additional Hydra override arguments to pass to the
            train script.
        cfg_launcher: Use the launcher specified in the configuration instead
            of the Hydra basic launcher. Defaults to False.
        config_name: The name of the config (usually the file name without the
            .yaml extension). Defaults to "config".
        config_path: The config path, a directory where Hydra will search for
            config files. If config_path is None no directory is added to the
            search path. Defaults to None.
    """
    if running_in_notebook():
        run_hydra_cmd(
            "fetch -l" if cfg_launcher else "fetch",
            override_kwargs,
            config_name,
            config_path,
            cmd_prefix="aucurriculum",
        )

    else:
        add_hydra_args_to_sys(override_kwargs, config_name, config_path)
        script = FetchScript()
        script.parser = MockParser()
        script.main(FetchArgs(cfg_launcher))


@catch_cli_errors
def preprocess(
    override_kwargs: Optional[dict] = None,
    num_workers: int = 1,
    update_frequency: int = 1,
    cfg_launcher: bool = False,
    config_name: str = "config",
    config_path: Optional[str] = None,
) -> None:
    """Launch a data preprocessing configuration.

    Args:
        override_kwargs: Additional Hydra override arguments to pass to the
            train script.
        num_workers: Number of workers (threads) to use for preprocessing.
            Defaults to 1.
        update_frequency: Frequency of progress bar updates for each worker
            (thread). If 0, the progress bar will be disabled. Defaults to 1.
        cfg_launcher: Use the launcher specified in the configuration instead
            of the Hydra basic launcher. Defaults to False.
        config_name: The name of the config (usually the file name without the
            .yaml extension). Defaults to "config".
        config_path: The config path, a directory where Hydra will search for
            config files. If config_path is None no directory is added to the
            search path. Defaults to None.
    """
    if running_in_notebook():
        cmd = "preprocess"
        if cfg_launcher:
            cmd += " -l"
        cmd += f" -n {num_workers} -u {update_frequency}"
        run_hydra_cmd(
            cmd,
            override_kwargs,
            config_name,
            config_path,
            cmd_prefix="aucurriculum",
        )

    else:
        add_hydra_args_to_sys(override_kwargs, config_name, config_path)
        script = PreprocessScript()
        script.parser = MockParser()
        script.main(
            PreprocessArgs(cfg_launcher, num_workers, update_frequency)
        )
