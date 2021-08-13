# library doc string


############# Imports #######################################

import pandas as pd
import seaborn as sns
import numpy as np
import math

from utils import (read_config,import_data,get_categorical,get_numerical,set_plot,save_plot,
                    set_subplots,set_target_plot,get_barplot,get_histogram,numericalize_target,set_cardinality_plot,
                    set_target_correlations_plot,set_feature_correlations_plot)

# Read Configuration File

config=read_config()

################# PARAMETERS ###################################################

TARGET=config["EDA_TARGET"]["TARGET_NAME"]
POSITIVE_CLASS=config["EDA_TARGET"]["POSITIVE_CLASS"]

############################# EDA ###############################################

def eda_single_plot(*,data,
                    kind,
                    target=TARGET,
                    positive_class=POSITIVE_CLASS,
                    **kwargs):

    df=data.copy() 
    fig,ax = set_plot(**kwargs)

    if kind.lower() =="target_distribution":
        bar_data,filename= set_target_plot(data,target)
        ax=get_barplot(data=bar_data,percentual=True)
        
    elif kind.lower() =="categorical_cardinality":
        bar_data,filename=set_cardinality_plot(data,target)
        ax=get_barplot(data=bar_data,percentual=False)

    elif kind.lower() == "target_correlations":
        bar_data,filename=set_target_correlations_plot(data,positive_class,target)
        ax=get_barplot(data=bar_data,percentual=True)


    elif kind.lower() == "feature_correlations":
        midpoint,corr_data,non_exaustive_corr,filename = set_feature_correlations_plot(data,
                                                                                    positive_class,
                                                                                    target)
        ax=sns.heatmap(corr_data,
                    cmap='Blues',
                    mask=non_exaustive_corr,
                    cbar=False,
                    center=midpoint,
                    annot=True)

    save_plot(fig,filename)

def eda_grid_plot(*,data,
            kind,
            title_size=15,
            target=TARGET,
            positive_class=POSITIVE_CLASS,
            **kwargs):
    
    df=data.copy()
        
    if (kind.lower() == "rate") or (kind.lower() == "distribution"):
        features,n_plots,n_cols,fig = set_subplots(data,target,"categorical",**kwargs)
        df[target] = numericalize_target(df,target,positive_class)
        
    elif (kind.lower() == "strip") or (kind.lower() == "histogram"):
        features,n_plots,n_cols,fig = set_subplots(data,target,"numerical",**kwargs)
    
    for counter,feat in zip(range(1,n_plots+1),features):
        
        ax = fig.add_subplot(math.ceil(n_plots/n_cols),n_cols,counter)
        

        if kind.lower() =="rate":
            bar_data=df.groupby(feat)[target].mean()
            ax = get_barplot(data=bar_data)
            set_plot(grid=True,ax=ax,fig=fig,title=feat)
            filename=config["EDA_CATEGORICAL"]["CATEGORICAL_RATE_FILENAME"]
                        
        elif kind.lower() == "distribution":
            bar_data=df[feat].value_counts(normalize=True)
            ax = get_barplot(data=bar_data)
            set_plot(grid=True,ax=ax,fig=fig,title=feat)
            filename=config["EDA_CATEGORICAL"]["CATEGORICAL_DISTRIBUTION_FILENAME"]
                                            
        elif kind.lower()== "strip":
            ax = sns.stripplot(x=target, y=feat, data=df)
            set_plot(grid=True,ax=ax,fig=fig,title=feat)
            filename=config["EDA_NUMERICAL"]["STRIPPLOT_FILENAME"]
            
        elif kind.lower() == "histogram":
            ax = df[feat].plot(kind="hist")
            set_plot(grid=True,ax=ax,fig=fig,title=feat)
            filename=config["EDA_NUMERICAL"]["HISTOGRAM_FILENAME"]
            
    save_plot(fig,filename)  

#################################################################################################