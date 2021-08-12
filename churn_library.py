# library doc string


############# Imports #######################################

import pandas as pd
import seaborn as sns
import numpy as np

from utils import (read_config,import_data,get_categorical,get_numerical,set_plot,save_plot,
                    set_subplots,get_barplot,get_histogram,numericalize_target)

# Read Configuration File

config=read_config()

################# PARAMETERS ###################################################

TARGET=config["EDA_TARGET"]["TARGET_NAME"]
POSITIVE_CLASS=config["EDA_TARGET"]["POSITIVE_CLASS"]

############################# EDA ###############################################

def plot_distribution(*,data,
                    kind,
                    target=TARGET,
                    **kwargs):

    df=data.copy()
    fig,ax = set_plot(**kwargs)
    if kind.lower() =="target":
        bar_data= df[target].value_counts(normalize=True,dropna=False)
        ax=get_barplot(data=bar_data,percentual=True)
        filename=config["EDA_TARGET"]["TARGET_DISTRIBUTION_FILENAME"]
        
    elif kind.lower() =="cardinality":
        categorical=get_categorical(df,target)
        bar_data= df[categorical].nunique()
        ax=get_barplot(data=bar_data,percentual=False)
        filename=config["EDA_CATEGORICAL"]["CATEGORICAL_CARDINALITY_FILENAME"]
        
    save_plot(fig,filename)

def plot_grid(*,data,
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
        
        ax = fig.add_subplot(round(n_plots/n_cols),n_cols,counter)
        
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

def plot_target_correlations(*,data,
                            positive_class=POSITIVE_CLASS,
                            target=TARGET,
                            **kwargs):
    df=data.copy()
    df[target] = numericalize_target(df,target,positive_class)
    numerical=get_numerical(df,target)
    fig,ax = set_plot(**kwargs)
    ax = df[numerical].corrwith((data[target]==positive_class).astype(int)).sort_values().plot(kind="barh")
    save_plot(fig,config["EDA_TARGET"]["TARGET_CORRELATIONS_FILENAME"]) 
    
def plot_features_correlations(*,data,
                            positive_class=POSITIVE_CLASS,
                            target=TARGET,
                            annot=True,
                            **kwargs):
    df=data.copy()
    df[target] = numericalize_target(df,target,positive_class)
    numerical=get_numerical(data,target)
    midpoint=data[numerical].corr().mean().mean()
    fig,ax = set_plot(**kwargs)
    correlations = df[numerical + [target]].corr()
    non_exaustive_corr = np.triu(correlations)
    ax=sns.heatmap(correlations,cmap='Blues',mask=non_exaustive_corr,cbar=False,center=midpoint,annot=annot)
    save_plot(fig,config["EDA_NUMERICAL"]["NUMERICAL_FEATURES_CORRELATIONS_FILENAME"])

#################################################################################################