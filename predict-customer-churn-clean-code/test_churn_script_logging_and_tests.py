"""
This script contains the unit tests for testing the proper functioning of the
ChurnPredictor class.
It makes use of the pytest library.

Author: Patrick Brus

Date: 2/15/2024
"""

import os
import shutil
import logging
from unittest.mock import patch
import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from churn_library import ChurnPredictor


@pytest.fixture
def path():
    """
    Return the path to the test csv data file.
    """
    return r"./data/bank_data.csv"


@pytest.fixture
def path_test_images_eda():
    """
    Return path to folder for storing eda test images.
    """
    return r"./test_images/eda"


@pytest.fixture
def path_test_images_results():
    """
    Return path to folder for storing result test images.
    """
    return r"./test_images/results"


@pytest.fixture
def sample_df():
    """
    Return a test dataframe with some sample data.
    """
    return pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [2, 4, 6, 8, 10],
            "col3": ["A", "B", "C", "D", "E"],
        }
    )


@pytest.fixture
def sample_data_train():
    """
    Create sample train and test data for running training on.
    """
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def test_import_data_returns_dataframe(path):
    """
    Test that import data can load the CSV file into a pandas dataframe.
    """
    logging.info("Start test import data returns dataframe...")
    churn_predictor = ChurnPredictor(path)
    churn_predictor.import_data()

    logging.info("Data import for real data succeeded.")

    assert isinstance(churn_predictor.df, pd.DataFrame)

    logging.info("Data is of instance pandas dataframe.")


def test_import_data_raises_exception():
    """
    Test that exception of "FileNotFoundError" gets raised in case the CSV
    file does not exist.
    """
    logging.info("Start test import data raises exception...")
    with pytest.raises(FileNotFoundError):
        churn_predictor = ChurnPredictor("non_existent_file.csv")
        churn_predictor.import_data()
        logging.info(
            "Test case successfully raised exception FileNotFoundError."
        )


def test_import_data_reads_csv(path):
    """
    Test that the pandas.read_csv function gets called.
    """
    logging.info("Start test import data reads csv...")
    with patch("pandas.read_csv") as mock_csv:
        churn_predictor = ChurnPredictor(path)
        churn_predictor.import_data()
        mock_csv.assert_called_once_with(path)
        logging.info(
            "Test successful. Pandas read csv function called at least once."
        )


def test_add_churn_col():
    """
    Test that the target column Churn gets added correctly.
    """
    logging.info("Start test add churn col...")
    churn_predictor = ChurnPredictor("test.csv")

    # Create a test CSV file
    test_csv = pd.DataFrame(
        {
            "Attrition_Flag": [
                "Existing Customer",
                "Existing Customer",
                "Attrited Customer",
            ]
        }
    )
    test_csv.to_csv("test.csv", index=False)

    # Call the import_data method
    churn_predictor.import_data()

    # Check if the dataframe is loaded properly
    assert len(churn_predictor.df) == 3
    assert churn_predictor.df.columns.tolist() == ["Attrition_Flag", "Churn"]
    assert churn_predictor.df["Attrition_Flag"].tolist() == [
        "Existing Customer",
        "Existing Customer",
        "Attrited Customer",
    ]
    assert churn_predictor.df["Churn"].tolist() == [0, 0, 1]

    # Clean up
    os.remove("test.csv")

    logging.info("Test add churn col PASSED.")


def test_perform_eda(sample_df, path_test_images_eda):
    """
    Test that the EDA function correctly creates histogram files for numerical
    columns and bar plots for categorical columns.
    """
    logging.info("Start test perform EDA...")
    churn_predictor = ChurnPredictor("")
    churn_predictor.df = sample_df
    churn_predictor.perform_eda(path_test_images_eda)

    assert os.path.exists(os.path.join(path_test_images_eda, "hist_col1.png"))
    assert os.path.exists(os.path.join(path_test_images_eda, "hist_col2.png"))
    assert os.path.exists(os.path.join(path_test_images_eda, "bar_col3.png"))
    assert os.path.exists(
        os.path.join(path_test_images_eda, "correlation_heatmap.png")
    )

    shutil.rmtree(path_test_images_eda)

    logging.info("Test perform EDA PASSED.")


def test_get_categorical_cols():
    """
    Test that the function get_categorical_cols correctly returns a list of
    all categorical columns.
    """
    logging.info("Start test get categorical cols...")
    df = pd.DataFrame(
        {
            "col1": ["A", "B", "C"],
            "col2": [1, 2, 3],
            "col3": pd.Categorical(["X", "Y", "X"]),
        }
    )

    churn_predictor = ChurnPredictor("")
    churn_predictor.df = df
    assert churn_predictor._get_categorical_cols() == ["col1", "col3"]

    logging.info("Test get categorical cols PASSED.")


def test_encode_features():
    """
    Test that the encode_features function correctly one-hot encodes the
    categorical column.
    """
    logging.info("Start test encode features...")
    df = pd.DataFrame(
        {
            "Gender": ["M", "F", "M", "M", "F"],
            "Churn": [1, 0, 1, 1, 1],
            "num_col1": [1, 2, 3, 4, 5],
        }
    )

    churn_predictor = ChurnPredictor("")
    churn_predictor.df = df
    churn_predictor._encode_features(["Gender"])

    assert sorted(churn_predictor.df.columns) == [
        "Churn",
        "Gender_F",
        "Gender_M",
        "num_col1",
    ]
    assert (churn_predictor.df["Gender_F"] == [0, 1, 0, 0, 1]).all()
    assert (churn_predictor.df["Gender_M"] == [1, 0, 1, 1, 0]).all()

    logging.info("Test encode features PASSED.")


def test_perform_feature_engineering():
    """
    Test that the feature engineering function works as expected.
    """
    logging.info("Start test perform feature engineering...")
    df = pd.DataFrame(
        {
            "cat_col1": ["A", "B", "C"],
            "cat_col2": ["X", "Y", "X"],
            "num_col1": [1, 2, 3],
            "Churn": [1, 0, 1],
        }
    )

    churn_predictor = ChurnPredictor("")
    churn_predictor.df = df

    # Test the function output
    churn_predictor.perform_feature_engineering()
    assert isinstance(churn_predictor.X_train, pd.DataFrame)
    assert isinstance(churn_predictor.X_test, pd.DataFrame)
    assert isinstance(churn_predictor.y_train, pd.Series)
    assert isinstance(churn_predictor.y_test, pd.Series)
    assert len(churn_predictor.X_train) == 2
    assert len(churn_predictor.X_test) == 1
    assert len(churn_predictor.y_train) == 2
    assert len(churn_predictor.y_test) == 1

    logging.info("Test perform feature engineering PASSED.")


# Test classification_report_image function
def test_classification_report_image(path_test_images_results):
    """
    Test that the classification reports for all classifiers and for train and
    test data get saved to disk at the expected path.
    """
    logging.info("Start test classification report image...")
    # Create sample data
    y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_test = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    y_train_preds_lr = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_train_preds_rf = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_test_preds_lr = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    y_test_preds_rf = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0])

    # Test the function output
    churn_predictor = ChurnPredictor("")
    churn_predictor.y_train = y_train
    churn_predictor.y_test = y_test
    churn_predictor._classification_report_image(
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
        path_plots=path_test_images_results,
    )

    # Check if the images have been created in the expected directory
    assert os.path.exists(
        os.path.join(path_test_images_results, "clf_lr_train_results.png")
    )
    assert os.path.exists(
        os.path.join(path_test_images_results, "clf_rf_train_results.png")
    )
    assert os.path.exists(
        os.path.join(path_test_images_results, "clf_lr_test_results.png")
    )
    assert os.path.exists(
        os.path.join(path_test_images_results, "clf_rf_test_results.png")
    )

    shutil.rmtree(path_test_images_results)

    logging.info("Test classification report image PASSED.")


# Test classification_report_to_image function
def test_classification_report_to_image(path_test_images_results):
    """
    Test that the classification report image for one classifier gets created.
    """
    logging.info("Start test classification report to image...")
    # Create sample data
    clf_report = {
        "precision": {
            "0": 0.6,
            "1": 0.75,
            "macro avg": 0.675,
            "weighted avg": 0.675,
        },
        "recall": {"0": 0.5, "1": 0.9, "macro avg": 0.7, "weighted avg": 0.7},
        "f1-score": {
            "0": 0.5454545454545454,
            "1": 0.8181818181818181,
            "macro avg": 0.6818181818181818,
            "weighted avg": 0.6818181818181818,
        },
        "support": {"0": 4, "1": 5, "macro avg": 9, "weighted avg": 9},
    }

    # Test the function output
    churn_predictor = ChurnPredictor("")
    churn_predictor._classification_report_to_image(
        clf_report, "test_report", path_plots=path_test_images_results
    )

    # Check if the image has been created in the expected directory
    assert os.path.exists(
        os.path.join(path_test_images_results, "test_report.png")
    )

    shutil.rmtree(path_test_images_results)

    logging.info("Test classification report to image PASSED.")


def test_roc_curve_image(sample_data_train, path_test_images_results):
    """
    Test that the ROC curve image gets created and saved to disk.
    """
    logging.info("Start test roc curve image...")
    filename = "test_roc_curve"
    X_train, X_test, y_train, y_test = sample_data_train

    # Train classifiers
    estimator_baseline = LogisticRegression(random_state=42)
    estimator = RandomForestClassifier(random_state=42)
    estimator_baseline.fit(X_train, y_train)
    estimator.fit(X_train, y_train)

    churn_predictor = ChurnPredictor("")
    churn_predictor.clf_baseline = estimator_baseline
    churn_predictor.clf_rf = estimator
    churn_predictor.X_test = X_test
    churn_predictor.y_test = y_test
    churn_predictor._roc_curve_image(filename, path_test_images_results)

    # Check if the image has been created in the expected directory
    assert os.path.exists(
        os.path.join(path_test_images_results, filename + ".png")
    )

    shutil.rmtree(path_test_images_results)

    logging.info("Test roc curve image PASSED.")


def test_train_logistic_regression(sample_data_train):
    """
    Test that checks whether the logistic regression training works as
    expected and that the final model gets saved as pkl file to disk and
    can be loaded again correctly.
    """
    logging.info("Start test train logistic regression...")
    X_train, X_test, y_train, y_test = sample_data_train

    churn_predictor = ChurnPredictor("", "test_models")
    churn_predictor.X_train = X_train
    churn_predictor.X_test = X_test
    churn_predictor.y_train = y_train
    churn_predictor.y_test = y_test

    churn_predictor._train_logistic_regression()

    # Check if the model is trained and saved properly
    assert os.path.exists("test_models")
    assert os.path.getsize("test_models") > 0
    with open(
        os.path.join("test_models", "logistic_regression.pkl"), "rb"
    ) as file:
        clf_rf = joblib.load(file)
        assert "LogisticRegression" in str(type(clf_rf))

    shutil.rmtree("test_models")

    logging.info("Test train logistic regression PASSED.")


def test_train_random_forest(sample_data_train):
    """
    Test that checks whether the random forest training works as
    expected and that the final model gets saved as pkl file to disk and
    can be loaded again correctly.
    """
    logging.info("Start test train random forest...")
    X_train, X_test, y_train, y_test = sample_data_train

    churn_predictor = ChurnPredictor("", "test_models")
    churn_predictor.X_train = X_train
    churn_predictor.X_test = X_test
    churn_predictor.y_train = y_train
    churn_predictor.y_test = y_test

    churn_predictor._train_random_forest()

    # Check if the model is trained and saved properly
    assert os.path.exists("test_models")
    assert os.path.getsize("test_models") > 0
    with open(os.path.join("test_models", "random_forest.pkl"), "rb") as file:
        clf_rf = joblib.load(file)
        assert "RandomForestClassifier" in str(type(clf_rf))

    shutil.rmtree("test_models")

    logging.info("Test train random forest PASSED.")
