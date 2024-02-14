# library doc string

from __future__ import annotations

# import libraries
import os
import shap
import joblib
import logging
import pandas as pd
import shutil
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

os.environ["QT_QPA_PLATFORM"]="offscreen"

class ChurnPredictor:
    # define statics
    MODEL_PATH = r"./models/rfc_model.pkl"
    TARGET_COLUMN = "Churn"
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.clf_rf = None
        self.clf_baseline = None
    
    def import_data(self):
        """ 
        Load the data from the CSV file into a pandas dataframe.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"The file at path {self.data_path} does not exist.")
        
        self.df = pd.read_csv(self.data_path)
        self._add_target_col_to_df()

    def perform_eda(self,
                    path_plots="images/eda"):
        """ Perform eda on df and save figures to images folder.

        Args:
            path_plots (str, optional): Path to folder where eda plots should be stored in. Defaults to "images/eda".
        """
        # cleanup the directory before creating new plots
        if os.path.exists(path_plots):
            shutil.rmtree(path_plots)
        
        # create the directory to save the plots
        os.makedirs(path_plots)

        # loop through each column in the dataframe and create a histogram
        for feature in self.df.columns:
            
            fig, ax = plt.subplots(figsize=(8, 6))  # adjust the figure size
            
            if self.df[feature].dtype != "object":
                ax.hist(self.df[feature])
                filename = f"hist_{feature}.png"
            else:
                self.df[feature].value_counts("normalize").plot(kind="bar", ax=ax)
                filename = f"bar_{feature}.png"
            
            ax.set_title(feature)
            ax.set_xlabel("Value")  # add a label to the x axis
            ax.set_ylabel("Frequency")  # add a label to the y axis
            plt.tight_layout()
            fig.savefig(os.path.join(path_plots, filename), dpi=300)
            plt.close(fig)  # close the figure to avoid memory leaks
        
        # create and plot heatmap
        fig = plt.figure(figsize=(20,10)) 
        sns.heatmap(self.df.corr(), annot=False, cmap="Dark2_r", linewidths = 2)
        fig.savefig(os.path.join(path_plots, "feature_heatmap.png"), dpi=300)
        
    def perform_feature_engineering(self):
        """
        apply feature engineering on dataset before training in order to prepare training dataset
        """
        cat_cols = self._get_categorical_cols()
        self._encoder_helper(category_lst=cat_cols)
        
        y = self.df[self.TARGET_COLUMN]
        X = self.df.drop(self.TARGET_COLUMN, axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    def train_models(self):
        """
        train, store model results: images + scores, and store models
        """
        
        rfc = RandomForestClassifier(random_state=42)
        self.clf_baseline = LogisticRegression(solver="lbfgs", max_iter=3000)

        # dict for grid search parameters
        param_grid = { 
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth" : [4, 5, 100],
            "criterion" :["gini", "entropy"]
        }

        # Train random forest model using grid seach
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(self.X_train, self.y_train)
        self.clf_rf = cv_rfc.best_estimator_
        
        # Train logistic regression model
        self.clf_baseline.fit(self.X_train, self.y_train)
        
        # save best model
        joblib.dump(self.clf_rf, self.MODEL_PATH)
        joblib.dump(self.clf_baseline, self.MODEL_PATH)
        
        # get predictions 
        y_train_preds_rf =  self.clf_rf.predict(self.X_train)
        y_test_preds_rf =  self.clf_rf.predict(self.X_test)
        y_train_preds_lr = self.clf_baseline.predict(self.X_train)
        y_test_preds_lr = self.clf_baseline.predict(self.X_test)
        
        self._classification_report_image(y_train_preds_lr,
                                          y_train_preds_rf,
                                          y_test_preds_lr,
                                          y_test_preds_rf)
        
        self._roc_curve_image("roc_curve_test")
        
        self._feature_importance_plot(self.clf_rf)
        
    def _add_target_col_to_df(self):
        """
        Helper function to add the target column with name Churn which will be used for training later.
        """
        self.df[self.TARGET_COLUMN] = self.df["Attrition_Flag"].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    def _get_categorical_cols(self):
        """Get the list of categorical columns in the dataframe.
        
        Returns:
        list: List of categorical columns.
        """
        cat_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(self.df[col]):
                cat_cols.append(col)
        return cat_cols

    def _encoder_helper(self,
                       category_lst: list[str]):
        """
        helper function to turn each categorical column into a new column with
        propotion of churn for each category

        Args:
            category_lst: list of columns that contain categorical features
        """
        for category in category_lst:
            logging.info(f"Add proportion of churn for category {category}...")
            category_mean = self.df.groupby(category)[self.TARGET_COLUMN].transform("mean")
            self.df[category + "_" + self.TARGET_COLUMN] = category_mean
        
        self.df = self.df.drop(category_lst, axis=1)
        
    def _classification_report_image(self,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf,
                                    path_plots="images/results"):
        """
        produces classification report for training and testing results and stores report as image
        in images folder
        Args:
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            path_plots (str, optional): Folder to store the created image in. Defaults to "images/results".
        """
        # classification report train results
        clf_report_train_lr = classification_report(self.y_train,
                                                    y_train_preds_lr,
                                                    output_dict=True)
        self._classification_report_to_image(clf_report_train_lr, 
                                    filename="clf_lr_train_results",
                                    path_plots=path_plots)
        
        clf_report_train_rf = classification_report(self.y_train,
                                                    y_train_preds_rf,
                                                    output_dict=True)
        self._classification_report_to_image(clf_report_train_rf, 
                                    filename="clf_rf_train_results",
                                    path_plots=path_plots)
        
        # classification report test results
        clf_report_test_lr = classification_report(self.y_test,
                                                    y_test_preds_lr,
                                                    output_dict=True)
        self._classification_report_to_image(clf_report_test_lr, 
                                    filename="clf_lr_test_results",
                                    path_plots=path_plots)
        
        clf_report_test_rf = classification_report(self.y_test,
                                                    y_test_preds_rf,
                                                    output_dict=True)
        self._classification_report_to_image(clf_report_test_rf, 
                                    filename="clf_rf_test_results",
                                    path_plots=path_plots)


    def _classification_report_to_image(self, 
                                       clf_report: dict, 
                                       filename: str, 
                                       path_plots="images/results"):
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

    def _feature_importance_plot(self,
                                model,
                                path_plots="images/results"):
        """
        creates and stores the feature importances in pth
        Args:
            model: model object containing feature_importances_
            path_plots (str, optional): Folder to store the created image in. Defaults to "images/results".
        """
        # plot shapley values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_test)
        shap.summary_plot(shap_values, self.X_test, plot_type="bar")
        plt.tight_layout()
        plt.savefig(os.path.join(path_plots, "feature_importance_shap.png"))
        plt.close()
        
        # Calculate feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [self.X_test.columns[i] for i in indices]
        plt.figure(figsize=(20,5))
        plt.title("Feature Importance")
        plt.ylabel("Importance")
        plt.bar(range(self.X_test.shape[1]), importances[indices])
        plt.xticks(range(self.X_test.shape[1]), names, rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(path_plots, "feature_importance.png"))
        plt.close()

    def _roc_curve_image(self,
                        filename,
                        path_plots="images/results"):
        """ 
        Plot ROC curve for given X and y using two estimators: 
        he baseline one and the one that shall be compared to the baseline model.

        Args:
            estimator (_type_): _description_
            estimator_baseline (_type_): _description_
            X (_type_): _description_
            y (_type_): _description_
            filename (_type_): _description_
            path_plots (str, optional): _description_. Defaults to "images/results".
        """
        if not os.path.exists(path_plots):
            os.makedirs(path_plots)
            
        plt.figure(figsize=(15, 8))
        lrc_plot = plot_roc_curve(self.clf_baseline, self.X_test, self.y_test)
        ax = plt.gca()
        plot_roc_curve(self.clf_rf, self.X_test, self.y_test, ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path_plots, filename + ".png"))
        plt.close()
    

if __name__ == "__main__":
    DATA_PATH = r"./data/bank_data.csv"
    churn_predictor = ChurnPredictor(data_path=DATA_PATH)
    churn_predictor.import_data()
    churn_predictor.perform_eda()
    churn_predictor.perform_feature_engineering()
    churn_predictor.train_models()