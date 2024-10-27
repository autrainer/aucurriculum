from typing import Dict, List

from autrainer.postprocessing import AggregateGrid, GroupGrid, SummarizeGrid
from omegaconf import DictConfig, ListConfig

from .summarize_curriculum import SummarizeCurriculum


class GroupCurriculum(GroupGrid):
    def _group_experiment(
        self,
        experiment: str,
        runs: ListConfig[DictConfig],
        create_summary: bool,
        mappings: Dict[str, List[str]],
    ) -> None:
        grouped_dict = {}
        for run in runs:
            base_runs = []
            for r in run.combine:
                base_runs.extend(mappings[r])

            grouped_dict.update({run.run_name: base_runs})
        if create_summary:
            sg = SummarizeGrid(
                results_dir=self.results_dir,
                experiment_id=experiment,
                max_runs_plot=self.max_runs,
            )
            sg.summarize()
            sg.plot_aggregated_bars()
            sg.plot_metrics()

            sc = SummarizeCurriculum(
                results_dir=self.results_dir,
                experiment_id=experiment,
                max_runs_plot=self.max_runs,
            )
            sc.summarize()
            sc.plot_pace()

        ag = AggregateGrid(
            results_dir=self.results_dir,
            experiment_id=experiment,
            aggregate_list="",
            aggregate_name="grouping",
            aggregated_dict=grouped_dict,
            max_runs_plot=self.max_runs,
            plot_params=self.plot_params,
        )
        ag.aggregate()
        ag.summarize()
