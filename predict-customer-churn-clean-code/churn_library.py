# library doc string

from __future__ import annotations

# import libraries
import os
import logging
import pandas as pd
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

os.environ["QT_QPA_PLATFORM"]="offscreen"



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
    return add_target_col_to_df(pd.read_csv(pth))
    

def add_target_col_to_df(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """Helper function to add the target column with name Churn which will be used for training later.

    Args:
        df (pd.core.frame.DataFrame): pandas dataframe

    Returns:
        pd.core.frame.DataFrame: pandas dataframe with added column Churn
    """
    df["Churn"] = df["Attrition_Flag"].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return df
    

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
            df[feature].value_counts("normalize").plot(kind="bar", ax=ax)
            filename = f"bar_{feature}.png"
        
        ax.set_title(feature)
        ax.set_xlabel("Value")  # add a label to the x axis
        ax.set_ylabel("Frequency")  # add a label to the y axis
        plt.tight_layout()
        fig.savefig(os.path.join(path_plots, filename), dpi=300)
        plt.close(fig)  # close the figure to avoid memory leaks
    
    # create and plot heatmap
    fig = plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths = 2)
    fig.savefig(os.path.join(path_plots, "feature_heatmap.png"), dpi=300)
    


def encoder_helper(df: pd.core.frame.DataFrame, category_lst: list[str], response: str) -> pd.core.frame.DataFrame:
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    Args:
        df (pd.core.frame.DataFrame): pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used for naming variables or index y column]

    Returns:
        df (pd.core.frame.DataFrame): pandas dataframe with new columns for
    """
    for category in category_lst:
        logging.info(f"Add proportion of churn for category {category}...")
        category_mean = df.groupby(category)["Churn"].transform("mean")
        df[category + "_Churn"] = category_mean
    
    return df.drop(category_lst, axis=1)


def perform_feature_engineering(df: pd.core.frame.DataFrame, response):
    """
    apply feature engineering on dataset before training in order to prepare training dataset
    Args:
        df (pd.core.frame.DataFrame): pandas dataframe
        response: string of response name [optional argument that could be used for naming variables or index y column]

    Returns:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """
    cat_cols = get_categorical_cols(df)
    df = encoder_helper(df, category_lst=cat_cols, response=response)
    
    y = df[response]
    X = df.drop(response, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

def get_categorical_cols(df):
    """Get the list of categorical columns in the dataframe.
    
    Args:
    df (pandas DataFrame): Input dataframe.
    
    Returns:
    list: List of categorical columns.
    """
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            cat_cols.append(col)
    return cat_cols

def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    Args:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure
    """
    pass

def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    Args:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    # dict for grid search parameters
    param_grid = { 
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth" : [4,5,100],
        "criterion" :["gini", "entropy"]
    }

    # Train random forest model using grid seach
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    
    # Train logistic regression model
    lrc.fit(X_train, y_train)
    
    # get predictions 
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    
def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                path_plots="images/results"):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    Args:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest
        path_plots (str, optional): Folder to store the created image in. Defaults to "images/results".
    """
    # classification report train results
    clf_report_train_lr = classification_report(y_train,
                                                y_train_preds_lr,
                                                output_dict=True)
    classification_report_to_image(clf_report_train_lr, 
                                   filename="clf_lr_train_results",
                                   path_plots=path_plots)
    
    clf_report_train_rf = classification_report(y_train,
                                                y_train_preds_rf,
                                                output_dict=True)
    classification_report_to_image(clf_report_train_rf, 
                                   filename="clf_rf_train_results",
                                   path_plots=path_plots)
    
    # classification report test results
    clf_report_test_lr = classification_report(y_test,
                                                y_test_preds_lr,
                                                output_dict=True)
    classification_report_to_image(clf_report_test_lr, 
                                   filename="clf_lr_test_results",
                                   path_plots=path_plots)
    
    clf_report_test_rf = classification_report(y_test,
                                                y_test_preds_rf,
                                                output_dict=True)
    classification_report_to_image(clf_report_test_rf, 
                                   filename="clf_rf_test_results",
                                   path_plots=path_plots)


def classification_report_to_image(clf_report: dict, filename: str, path_plots="images/results"):
    """ Helper function that takes in output dict of classification report and saves it to an image using seaborn.

    Args:
        clf_report (dict): Output dict of sklearn's classification report function
        filename (str): filename that should be used to save that image
        path_plots (str, optional): Folder to store the created image in. Defaults to "images/results".
    """
    
    if not os.path.exists(path_plots):
        os.makedirs(path_plots)
        
    fig = plt.figure(figsize=(20,10))
    plt.tight_layout()
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    fig.savefig(os.path.join(path_plots, filename + ".png"), dpi=300)    

if __name__ == "__main__":
    data = import_data(r"./data/bank_data.csv")
    perform_eda(data)
    X_train, X_test, y_train, y_test = perform_feature_engineering(data, response="Churn")
    train_models(X_train, X_test, y_train, y_test)