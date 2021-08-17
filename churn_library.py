# library doc string
"""
TODO:library docstring
"""

############# Imports #######################################

import math
import seaborn as sns

from utils import (
    read_config,
    set_plot,
    save_plot,
    set_subplots,
    set_target_plot,
    get_barplot,
    numericalize_target,
    set_cardinality_plot,
    set_target_correlations_plot,
    set_feature_correlations_plot,
)

# Read Configuration File

config = read_config()

################# PARAMETERS ###################################################

TARGET = config["EDA_TARGET"]["TARGET_NAME"]
POSITIVE_CLASS = config["EDA_TARGET"]["POSITIVE_CLASS"]

############################# EDA ###############################################


def eda_single_plot(
    *, data, kind, target=TARGET, positive_class=POSITIVE_CLASS, **kwargs
):

    """TODO: function docstring"""

    data_frame = data.copy()
    fig, ax = set_plot(**kwargs)

    if kind.lower() == "target_distribution":
        bar_data, filename = set_target_plot(data_frame, target)
        ax = get_barplot(data=bar_data, percentual=True)

    elif kind.lower() == "categorical_cardinality":
        bar_data, filename = set_cardinality_plot(data_frame, target)
        ax = get_barplot(data=bar_data, percentual=False)

    elif kind.lower() == "target_correlations":
        bar_data, filename = set_target_correlations_plot(
            data_frame, positive_class, target
        )
        ax = get_barplot(data=bar_data, percentual=True)

    elif kind.lower() == "feature_correlations":
        (
            midpoint,
            corr_data,
            non_exaustive_corr,
            filename,
        ) = set_feature_correlations_plot(data_frame, positive_class, target)
        ax = sns.heatmap(
            corr_data,
            cmap="Blues",
            mask=non_exaustive_corr,
            cbar=False,
            center=midpoint,
            annot=True,
        )

    save_plot(fig, filename)


def eda_grid_plot(
    *, data, kind, target=TARGET, positive_class=POSITIVE_CLASS, **kwargs
):

    """TODO: function docstring"""

    data_frame = data.copy()

    if (kind.lower() == "rate") or (kind.lower() == "distribution"):
        features, n_plots, n_cols, fig = set_subplots(
            data, target, "categorical", **kwargs
        )
        data_frame[target] = numericalize_target(data_frame, target, positive_class)

    elif (
        (kind.lower() == "strip")
        or (kind.lower() == "histogram")
        or (kind.lower() == "boxplot")
    ):
        features, n_plots, n_cols, fig = set_subplots(
            data, target, "numerical", **kwargs
        )

    for counter, feat in zip(range(1, n_plots + 1), features):

        ax = fig.add_subplot(math.ceil(n_plots / n_cols), n_cols, counter)

        if kind.lower() == "rate":
            bar_data = data_frame.groupby(feat)[target].mean()
            ax = get_barplot(data=bar_data)
            set_plot(grid=True, ax=ax, fig=fig, title=feat)
            filename = config["EDA_CATEGORICAL"]["CATEGORICAL_RATE_FILENAME"]

        elif kind.lower() == "distribution":
            bar_data = data_frame[feat].value_counts(normalize=True)
            ax = get_barplot(data=bar_data)
            set_plot(grid=True, ax=ax, fig=fig, title=feat)
            filename = config["EDA_CATEGORICAL"]["CATEGORICAL_DISTRIBUTION_FILENAME"]

        elif kind.lower() == "strip":
            ax = sns.stripplot(x=target, y=feat, data=data_frame, alpha=0.1)
            set_plot(grid=True, ax=ax, fig=fig, title=feat)
            filename = config["EDA_NUMERICAL"]["STRIPPLOT_FILENAME"]

        elif kind.lower() == "histogram":
            ax = data_frame[feat].plot(kind="hist")
            set_plot(grid=True, ax=ax, fig=fig, title=feat)
            filename = config["EDA_NUMERICAL"]["HISTOGRAM_FILENAME"]

        elif kind.lower() == "boxplot":
            ax = sns.boxplot(x=data_frame[feat])
            set_plot(grid=True, ax=ax, fig=fig, title=feat)
            filename = config["EDA_NUMERICAL"]["BOXPLOT_FILENAME"]

    save_plot(fig, filename)


#################################################################################################

############################# Model Training ####################################################
