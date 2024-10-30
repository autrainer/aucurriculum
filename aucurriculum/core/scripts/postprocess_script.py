from typing import List, Optional

from autrainer.core.scripts import PostprocessScript
from autrainer.core.scripts.abstract_script import MockParser
from autrainer.core.scripts.postprocess_script import PostprocessArgs
from autrainer.core.scripts.utils import catch_cli_errors


class PostprocessCurriculumScript(PostprocessScript):
    def _summarize(self, args: PostprocessArgs) -> None:
        from autrainer.postprocessing import SummarizeGrid

        from aucurriculum.postprocessing import (
            AggregateCurriculum,
            SummarizeCurriculum,
        )

        sg = SummarizeGrid(
            results_dir=args.results_dir,
            experiment_id=args.experiment_id,
            max_runs_plot=args.max_runs,
        )
        sg.summarize()
        sg.plot_aggregated_bars()
        sg.plot_metrics()

        sc = SummarizeCurriculum(
            results_dir=args.results_dir,
            experiment_id=args.experiment_id,
            max_runs_plot=args.max_runs,
        )
        sc.summarize()
        sc.plot_pace()

        if not args.aggregate:
            return
        for agg in args.aggregate:
            ag = AggregateCurriculum(
                results_dir=args.results_dir,
                experiment_id=args.experiment_id,
                aggregate_list=agg,
                max_runs_plot=args.max_runs,
            )
            ag.aggregate()
            ag.summarize()


@catch_cli_errors
def postprocess(
    results_dir: str,
    experiment_id: str,
    max_runs: Optional[int] = None,
    aggregate: Optional[List[List[str]]] = None,
) -> None:
    """Postprocess grid search results.

    Args:
        results_dir: Path to grid search results directory.
        experiment_id: ID of experiment to postprocess.
        max_runs: Maximum number of best runs to plot. Defaults to None.
        aggregate: Configurations to aggregate. One or more of:
            :const:`~autrainer.core.constants.NamingConstants.VALID_AGGREGATIONS`.
            Defaults to None.
    """
    script = PostprocessCurriculumScript()
    script.parser = MockParser()
    script.main(
        PostprocessArgs(
            results_dir,
            experiment_id,
            max_runs,
            aggregate,
        )
    )
