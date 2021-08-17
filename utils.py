# module doc string
"""
TODO:library docstring
"""


import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

####################  Utility Functions ######################################################


def read_config(config_file="config.ini"):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def import_data(filepath):
    """
    returns dataframe for the csv found at filename_path

    input:
            filename_path (str): a path to the csv
            ex.: ./data/input_data.csv
    output:
            df: pandas dataframe
    """

    data_frame = pd.read_csv(filepath)

    return data_frame


def split_data_target(target_col, filepath):
    """
    split data and target columns from a dataframe at filename_path

    input:
            target_col (str): target column name
            filename_path (str): a path to the csv
            ex.: ./data/input_data.csv

    output:
            data: pandas dataframe excluding the target_col column.
            target: pandas series containing the target
    """

    data_frame = import_data(filepath)
    data_frame, target = data_frame.drop(target_col, axis=1), data_frame[target_col]

    return data_frame, target


def get_categorical(data, target):
    data_frame = data.copy()
    config = read_config()
    categorical = [
        feature
        for feature in data_frame
        if (
            data_frame[feature].dtype == "object"
            or data_frame[feature].nunique()
            < int(config["EDA_CATEGORICAL"]["CARDINALITY_THRESHOLD"])
        )
        and feature != target
    ]
    return categorical


def get_numerical(data, target):
    data_frame = data.copy()
    cat = get_categorical(data_frame, target)
    numerical = [
        feature for feature in data_frame if feature not in cat and feature != target
    ]
    return numerical


def set_plot(
    plotsize=(15, 7),
    title="",
    title_size=15,
    grid=False,
    ax=None,
    fig=None,
    xlabel="",
    ylabel="",
):

    if grid:
        ax.set_title(title + "\n", size=title_size)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()

    else:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(plotsize)
        ax.set_title(title + "\n", size=title_size)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.set_tight_layout(True)

    return fig, ax


def set_fig(gridsize=(15, 7), gridtitle=""):
    fig = plt.figure()
    fig.set_size_inches(gridsize)
    fig.suptitle(gridtitle + "\n", size=20)
    return fig


def save_plot(fig, filename):
    config = read_config()
    fig.savefig(config["GENERAL"]["EDA_PATH"] + filename)


def set_subplots(data, target, features_type, ncols=3, **kwargs):
    data_frame = data.copy()
    if features_type.lower() == "categorical":
        features = get_categorical(data_frame, target)
    elif features_type.lower() == "numerical":
        features = get_numerical(data_frame, target)

    n_features = len(features)
    grid_cols = min(ncols, n_features)
    fig = set_fig(**kwargs)

    return features, n_features, grid_cols, fig


def get_barplot(data, percentual=True):
    data_frame = data.copy()
    if percentual:
        ax = data_frame.sort_values().mul(100).plot(kind="barh")
    else:
        ax = data_frame.sort_values().plot(kind="barh")

    return ax


def get_histogram(data):
    data_frame = data.copy()
    ax = data_frame.plot(kind="hist")
    return ax


def numericalize_target(data, target, positive_class):
    data_frame = data.copy()
    data_frame[target] = (data_frame[target] == positive_class).astype(int)
    return data_frame[target]


def set_target_plot(data, target):
    data_frame = data.copy()
    config = read_config()
    bar_data = data_frame[target].value_counts(normalize=True, dropna=False)
    filename = config["EDA_TARGET"]["TARGET_DISTRIBUTION_FILENAME"]
    return bar_data, filename


def set_cardinality_plot(data, target):
    data_frame = data.copy()
    config = read_config()
    categorical_features = get_categorical(data_frame, target)
    bar_data = data_frame[categorical_features].nunique()
    filename = config["EDA_CATEGORICAL"]["CATEGORICAL_CARDINALITY_FILENAME"]
    return bar_data, filename


def set_target_correlations_plot(data, positive_class, target):

    data_frame = data.copy()
    config = read_config()
    data_frame[target] = numericalize_target(data_frame, target, positive_class)
    numerical_features = get_numerical(data_frame, target)
    bar_data = data_frame[numerical_features].corrwith(data_frame[target]).sort_values()
    filename = config["EDA_TARGET"]["TARGET_CORRELATIONS_FILENAME"]
    return bar_data, filename


def set_feature_correlations_plot(data, positive_class, target):
    data_frame = data.copy()
    config = read_config()
    data_frame[target] = numericalize_target(data_frame, target, positive_class)
    numerical = get_numerical(data_frame, target)
    midpoint = data_frame[numerical].corr().mean().mean()
    corr_data = data_frame[numerical + [target]].corr()
    non_exaustive_corr = np.triu(corr_data)
    filename = config["EDA_NUMERICAL"]["NUMERICAL_FEATURES_CORRELATIONS_FILENAME"]
    return midpoint, corr_data, non_exaustive_corr, filename


#####################################################################################
