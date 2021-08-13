from churn_library import (import_data,eda_single_plot,eda_grid_plot)

from utils import read_config,has_categorical,has_numerical


def run_eda():

    config=read_config()

    data = import_data(config["DATA"]["PATH"])


    eda_single_plot(data=data,
                        kind="target_distribution",
                        title=config["EDA_TARGET"]["TARGET_DISTRIBUTION_TITLE"],
                        xlabel="% of observations",
                        )

    if has_numerical:
        
        eda_single_plot(data=data,
                        kind="target_correlations",
                        title=config["EDA_TARGET"]["TARGET_CORRELATIONS_TITLE"],
                        xlabel="% of observations",
                        )

        eda_single_plot(data=data,
                        kind="feature_correlations",
                        title=config["EDA_NUMERICAL"]["NUMERICAL_FEATURES_CORRELATIONS_TITLE"],
                                )

        eda_grid_plot(data=data,
            gridtitle=config["EDA_NUMERICAL"]["GRID_TITLE"],
            kind="distribution",
            )

        eda_grid_plot(data=data,
            kind="histogram",
            gridtitle=config["EDA_NUMERICAL"]["HISTOGRAM_TITLE"])

            
        eda_grid_plot(data=data,
            kind="strip",
            gridtitle=config["EDA_NUMERICAL"]["STRIPPLOT_TITLE"])

    
    if has_categorical:

        eda_single_plot(data=data,
                        kind="categorical_cardinality",
                        title=config["EDA_CATEGORICAL"]["CATEGORICAL_CARDINALITY_TITLE"],
                            )

        eda_grid_plot(data=data,
            gridtitle=config["EDA_CATEGORICAL"]["GRID_TITLE"],
            kind="Rate",
            )

        eda_grid_plot(data=data,
                    gridtitle=config["EDA_NUMERICAL"]["BOXPLOT_TITLE"],
                    kind="boxplot")


if __name__ == "__main__":
    run_eda()