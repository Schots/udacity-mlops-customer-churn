import configparser
from churn_library import (import_data,plot_target_distribution,plot_target_correlations,
                        plot_features_correlations,plot_categorical_cardinality,plot_grid)

from utils import read_config

# Read Configuration File
config=read_config()

def run_eda():
    config=read_config()

    data = import_data(config["DATA"]["PATH"])

    plot_target_distribution(data=data,
                        title=config["EDA_TARGET"]["TARGET_DISTRIBUTION_TITLE"],
                        xlabel="% of observations",
                        )

    plot_target_correlations(data=data,
                            title=config["EDA_TARGET"]["TARGET_CORRELATIONS_TITLE"])

    plot_features_correlations(data=data,
                            title=config["EDA_NUMERICAL"]["NUMERICAL_FEATURES_CORRELATIONS_TITLE"],
                                )

    plot_categorical_cardinality(data=data,
                                title=config["EDA_CATEGORICAL"]["CATEGORICAL_CARDINALITY_TITLE"],
                            )

    plot_grid(data=data,
            gridtitle=config["EDA_CATEGORICAL"]["GRID_TITLE"],
            kind="Rate"
            )

    plot_grid(data=data,
            gridtitle=config["EDA_NUMERICAL"]["GRID_TITLE"],
            kind="distribution",
            )

    plot_grid(data=data,
            kind="histogram",
            gridtitle=config["EDA_NUMERICAL"]["HISTOGRAM_TITLE"])

            
    plot_grid(data=data,
            kind="strip",
            gridtitle=config["EDA_NUMERICAL"]["STRIPPLOT_TITLE"])

if __name__ == "__main__":
    run_eda()