import os
import warnings

from autrainer.core.plotting import PlotBase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class CurriculumPlots(PlotBase):
    def __init__(
        self,
        output_directory: str,
        training_type: str,
        figsize: tuple,
        latex: bool,
        filetypes: list,
        pickle: bool,
        context: str,
        palette: str,
        replace_none: bool,
        add_titles: bool,
        add_xlabels: bool,
        add_ylabels: bool,
        rcParams: dict,
    ) -> None:
        """Plot the sample difficulty scores and pace of curricula.

        Args:
            output_directory: Output directory to save plots to.
            training_type: Type of training in ["Epoch", "Step"].
            figsize: Figure size in inches.
            latex: Whether to use LaTeX in plots. Requires the `latex` package.
                To install all necessary dependencies, run:
                `pip install autrainer[latex]`.
            filetypes: Filetypes to save plots as.
            pickle: Whether to save additional pickle files of the plots.
            context: Context for seaborn plots.
            palette: Color palette for seaborn plots.
            replace_none: Whether to replace "None" in labels with "~".
            add_titles: Whether to add titles to plots.
            add_xlabels: Whether to add x-labels to plots.
            add_ylabels: Whether to add y-labels to plots.
            rcParams: Additional Matplotlib rcParams to set.
        """
        super().__init__(
            output_directory,
            training_type,
            figsize,
            latex,
            filetypes,
            pickle,
            context,
            palette,
            replace_none,
            add_titles,
            add_xlabels,
            add_ylabels,
            rcParams,
        )

    def plot_score(self, df: pd.DataFrame, score_id: str, num_bins=20) -> None:
        """Plot the sample difficulty scores by creating histograms,
        facet grids, and ridge plots visualizing the global and per class
        sample difficulty distributions.

        Args:
            df: DataFrame containing the sample difficulty scores ("mean",
            "ranks") and corresponding labels ("decoded").
            score_id: ID of the scoring function.
            num_bins: Number of bins for the histogram. Defaults to 20.
        """
        df["decoded"] = df["decoded"].astype(str)
        order = sorted(df["decoded"].unique())
        sample_counts = df["decoded"].value_counts().sort_index().values

        self._plot_fn(
            df=df,
            score_id=score_id,
            name="histogram",
            x="mean",
            y_label="count",
            fn=sns.histplot,
            fn_kargs={"bins": num_bins},
        )
        self._plot_facet_grid(
            df=df,
            score_id=score_id,
            name="Count per Class",
            y_label="count",
            fn_map=sns.histplot,
            fn_kargs={
                "bins": np.arange(0, 1 + (1 / num_bins), 1 / num_bins),
                "stat": "count",
            },
            col_order=order,
        )
        percent_char = "\\%" if self.latex else "%"
        self._plot_facet_grid(
            df=df,
            score_id=score_id,
            name="Percent per Class",
            y_label=f"ratio [{percent_char}]",
            fn_map=sns.histplot,
            fn_kargs={
                "bins": np.arange(0, 1 + (1 / num_bins), 1 / num_bins),
                "stat": "percent",
            },
            col_order=order,
            sample_counts=sample_counts,
        )
        self._plot_ridge(
            df=df, score_id=score_id, name="Ridge per Class", row_order=order
        )

    def _plot_facet_grid(
        self,
        df: pd.DataFrame,
        score_id: str,
        name: str,
        y_label: str,
        fn_map: callable,
        fn_kargs: dict,
        col_order: list = None,
        sample_counts: list = None,
    ) -> None:
        fg = sns.FacetGrid(
            df,
            col="decoded",
            col_order=col_order,
            hue="decoded",
            col_wrap=5,
            height=self.figsize[1] / 2,
            aspect=self.figsize[0] / (self.figsize[1] * 2),
        )
        fg.set_titles(col_template="{col_name}")
        if sample_counts is not None:
            for ax, ct in zip(fg.axes.flat, sample_counts):
                ax.set_title(f"{ax.get_title()} ({ct} samples)")
        fg.map_dataframe(fn_map, x="ranks", **fn_kargs)
        if self.add_xlabels:
            fg.set_xlabels("ranks")
        else:
            fg.set_xlabels("")
        if self.add_ylabels:
            fg.set_ylabels(y_label)
        else:
            fg.set_ylabels("")
        if self.add_titles:
            fg.figure.suptitle(f"{score_id} {name}", y=0.975)
        else:
            fg.figure.suptitle("")
        path = os.path.join("_plots", score_id)
        self.save_plot(fg.figure, name.lower().replace(" ", "_"), path)

    def _plot_fn(
        self,
        df: pd.DataFrame,
        score_id: str,
        name: str,
        x: str,
        y_label: str,
        fn: callable,
        fn_kargs: dict,
    ) -> None:
        fig = plt.figure(figsize=self.figsize)
        fn(data=df, x=x, **fn_kargs)
        self._add_label(fig.gca(), x, y_label)
        path = os.path.join("_plots", score_id)
        self.save_plot(fig, name.lower().replace(" ", "_"), path)

    def _plot_ridge(
        self, df: pd.DataFrame, score_id: str, name: str, row_order: list
    ) -> None:
        _old_warn = warnings.filters[:]
        warnings.filterwarnings("ignore", category=UserWarning)
        height = (self.figsize[1] / 10) * 1.25
        fg = sns.FacetGrid(
            df,
            row="decoded",
            row_order=row_order,
            hue="decoded",
            height=height,
            aspect=(self.figsize[0] * 1.25) / height,
        )
        fg.map_dataframe(
            sns.kdeplot,
            x="ranks",
            bw_adjust=0.5,
            fill=True,
        )
        for ax in fg.axes.flat:
            ax.patch.set_facecolor("none")
            ax.xaxis.grid(False)
        fg.set_titles("")
        fg.set(yticks=[], ylabel="")

        def label(x, color, label):
            ax = plt.gca()
            ax.text(
                x=0.04,
                y=0,
                s=label,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
            )

        fg.map(label, "ranks")
        fg.despine(left=True)
        if self.add_titles:
            fg.figure.suptitle(f"{score_id} {name}", y=0.975)
        else:
            fg.figure.suptitle("")

        path = os.path.join("_plots", score_id)
        self.save_plot(fg.figure, name.lower().replace(" ", "_"), path)
        warnings.filters = _old_warn

    def plot_correlation_matrix(self, df: pd.DataFrame, name: str) -> None:
        """Create and plot the correlation matrix of the scoring functions.

        Args:
            df: DataFrame containing multiple scoring functions per column.
            name: Name of the correlation matrix plot.
        """
        df.sort_index(axis="columns", inplace=True)
        fig_height = int(1 + len(df.columns))
        fig = plt.figure(
            figsize=(max(self.figsize[0], fig_height), fig_height)
        )
        correlation_matrix = df.corr(method="spearman")
        correlation_matrix.to_csv(
            os.path.join(self.output_directory, f"{name}.csv")
        )
        percent_char = "\\%" if self.latex else "%"

        def escape_percent(x):
            return f"{(x*100):.2f}{percent_char}"

        sns.heatmap(
            correlation_matrix,
            annot=correlation_matrix.map(escape_percent),
            fmt="",
            cbar=False,
            cmap="crest",
        )
        fig.gca().set_xticklabels(df.columns, rotation=25)
        self.save_plot(fig, name)

    def plot_run(self, df: pd.DataFrame) -> None:
        """Plot the dynamic dataset size determined by the curriculum.

        Args:
            df: DataFrame containing the run metrics and dataset sizes.
        """
        fig = plt.figure(figsize=self.figsize)
        sns.lineplot(
            data=df["dataset_size"], dashes=False, drawstyle="steps-post"
        )
        self._add_label(fig.gca(), self.training_type, "dataset size")
        self.save_plot(fig, "pace")

    def plot_pace(self, df: pd.DataFrame) -> None:
        """Plot the dynamic dataset size of multiple runs.

        Args:
            df: DataFrame containing multiple runs and their dataset sizes.
        """
        fig = plt.figure(figsize=self.figsize)
        sns.lineplot(data=df, dashes=False, drawstyle="steps-post")
        self._add_label(fig.gca(), self.training_type, "dataset size")
        self.save_plot(fig, "pace")
