# library doc string


# import libraries
import os
import logging
import pandas as pd
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
os.environ['QT_QPA_PLATFORM']='offscreen'



def import_data(pth: str) -> pd.core.frame.DataFrame :
    """ returns dataframe for the csv found at pth

    Args:
        pth (str): path to CSV file containing data.

    Raises:
        FileNotFoundError: Raised in case the CSV file does not exist

    Returns:
        pd.core.frame.DataFrame: The dataframe for the given CSV data
    """
    if not os.path.exists(pth):
        raise FileNotFoundError(f"The file at path {pth} does not exist.")
    return pd.read_csv(pth)


def perform_eda(df: pd.core.frame.DataFrame, path_plots="images/eda"):
    """ Perform eda on df and save figures to images folder.

    Args:
        df (pd.core.frame.DataFrame): pandas dataframe
        path_plots (str, optional): Path to folder where eda plots should be stored in. Defaults to "images/eda".
    """
    # cleanup the directory before creating new plots
    if os.path.exists(path_plots):
        shutil.rmtree(path_plots)
    
    # create the directory to save the plots
    os.makedirs(path_plots)

    # loop through each column in the dataframe and create a histogram
    for feature in df.columns:
        
        fig, ax = plt.subplots(figsize=(8, 6))  # adjust the figure size
        
        if df[feature].dtype != "object":
            ax.hist(df[feature])
            filename = f"hist_{feature}.png"
        else:
            df[feature].value_counts("normalize").plot(kind='bar', ax=ax)
            filename = f"bar_{feature}.png"
        
        ax.set_title(feature)
        ax.set_xlabel('Value')  # add a label to the x axis
        ax.set_ylabel('Frequency')  # add a label to the y axis
        plt.tight_layout()
        fig.savefig(os.path.join(path_plots, filename), dpi=300)
        plt.close(fig)  # close the figure to avoid memory leaks
    
    # create and plot heatmap
    fig = plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    fig.savefig(os.path.join(path_plots, "feature_heatmap.png"), dpi=300)
    


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass

if __name__ == "__main__":
    data = import_data(r"./data/bank_data.csv")
    perform_eda(data)