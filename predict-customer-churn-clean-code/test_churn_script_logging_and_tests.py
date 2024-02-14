import os
import joblib
import shutil
import logging
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from unittest.mock import patch, Mock
from churn_library import ChurnPredictor

logging.basicConfig(
    filename="./logs/churn_library.log",
    level = logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s")

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
def df_test_churn():
    return pd.DataFrame({"Attrition_Flag": ["Existing Customer", "Existing Customer", "Attrited Customer"]})

@pytest.fixture
def sample_df():
    return pd.DataFrame({"col1": [1, 2, 3, 4, 5], 
                         "col2": [2, 4, 6, 8, 10],
                         "col3": ["A", "B", "C", "D", "E"]})

def test_import_data_returns_dataframe(path):
    churn_predictor = ChurnPredictor(path)
    churn_predictor.import_data()
    
    assert isinstance(churn_predictor.df, pd.DataFrame)
    
def test_import_data_raises_exception():
    with pytest.raises(FileNotFoundError):
        churn_predictor = ChurnPredictor("non_existent_file.csv")
        churn_predictor.import_data()
        
def test_import_data_reads_csv(path):
    with patch("pandas.read_csv") as mock_csv:
        churn_predictor = ChurnPredictor(path)
        churn_predictor.import_data()
        mock_csv.assert_called_once_with(path)

def test_add_churn_col():
    churn_predictor = ChurnPredictor("test.csv")
    
    # Create a test CSV file
    test_csv = pd.DataFrame({"Attrition_Flag": ["Existing Customer", "Existing Customer", "Attrited Customer"]})
    test_csv.to_csv("test.csv", index=False)

    # Call the import_data method
    churn_predictor.import_data()
    
    # Check if the dataframe is loaded properly
    assert len(churn_predictor.df) == 3
    assert churn_predictor.df.columns.tolist() == ["Attrition_Flag", "Churn"]
    assert churn_predictor.df["Attrition_Flag"].tolist() == ["Existing Customer", "Existing Customer", "Attrited Customer"]
    assert churn_predictor.df["Churn"].tolist() == [0, 0, 1]

    # Clean up
    os.remove("test.csv")
        
def test_perform_eda(sample_df, path_test_images_eda):
    churn_predictor = ChurnPredictor("")
    churn_predictor.df = sample_df
    churn_predictor.perform_eda(path_test_images_eda)

    assert os.path.exists(os.path.join(path_test_images_eda, "hist_col1.png"))
    assert os.path.exists(os.path.join(path_test_images_eda, "hist_col2.png"))
    assert os.path.exists(os.path.join(path_test_images_eda, "bar_col3.png"))
    assert os.path.exists(os.path.join(path_test_images_eda, "feature_heatmap.png"))

    shutil.rmtree(path_test_images_eda)
    
def test_get_categorical_cols():
    df = pd.DataFrame({"col1": ["A", "B", "C"], 
                       "col2": [1, 2, 3], 
                       "col3": pd.Categorical(["X", "Y", "X"])})
    
    churn_predictor = ChurnPredictor("")
    churn_predictor.df = df
    assert churn_predictor._get_categorical_cols() == ["col1", "col3"]

def test_encoder_helper():
    df = pd.DataFrame({"Gender": ["M", "F", "M", "M", "F"], 
                       "Churn": [1, 0, 1, 1, 1],
                       "num_col1": [1, 2, 3, 4, 5]})
    
    churn_predictor = ChurnPredictor("")
    churn_predictor.df = df
    churn_predictor._encoder_helper(["Gender"])
    
    assert sorted(churn_predictor.df.columns) == ['Churn', 'Gender_Churn', 'num_col1']
    assert (churn_predictor.df["Gender_Churn"] == [1.0, 0.5, 1.0, 1.0, 0.5]).all()
    
def test_perform_feature_engineering():
    # Test perform_feature_engineering function
    # Create a sample dataframe
    df = pd.DataFrame({'cat_col1': ['A', 'B', 'C'], 
                       'cat_col2': ['X', 'Y', 'X'],
                       'num_col1': [1, 2, 3],
                       'Churn': [1, 0, 1]})
    
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
    
# Test classification_report_image function
def test_classification_report_image(path_test_images_results):
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
    churn_predictor._classification_report_image(y_train_preds_lr,
                                                 y_train_preds_rf,
                                                 y_test_preds_lr,
                                                 y_test_preds_rf,
                                                 path_plots=path_test_images_results)
    
    # Check if the images have been created in the expected directory
    assert os.path.exists(os.path.join(path_test_images_results, "clf_lr_train_results.png"))
    assert os.path.exists(os.path.join(path_test_images_results, "clf_rf_train_results.png"))
    assert os.path.exists(os.path.join(path_test_images_results, "clf_lr_test_results.png"))
    assert os.path.exists(os.path.join(path_test_images_results, "clf_rf_test_results.png"))
    
    shutil.rmtree(path_test_images_results)

# Test classification_report_to_image function
def test_classification_report_to_image(path_test_images_results):
    # Create sample data
    clf_report = {
        'precision': {'0': 0.6, '1': 0.75, 'macro avg': 0.675, 'weighted avg': 0.675},
        'recall': {'0': 0.5, '1': 0.9, 'macro avg': 0.7, 'weighted avg': 0.7},
        'f1-score': {'0': 0.5454545454545454, '1': 0.8181818181818181, 'macro avg': 0.6818181818181818, 'weighted avg': 0.6818181818181818},
        'support': {'0': 4, '1': 5, 'macro avg': 9, 'weighted avg': 9}
    }
    
    # Test the function output
    churn_predictor = ChurnPredictor("")
    churn_predictor._classification_report_to_image(clf_report, "test_report", path_plots=path_test_images_results)
    
    # Check if the image has been created in the expected directory
    assert os.path.exists(os.path.join(path_test_images_results,"test_report.png"))

def test_roc_curve_image_old(path_test_images_results):
    filename="test_roc_curve"
    # Create sample data
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    assert os.path.exists(os.path.join(path_test_images_results, filename + ".png"))
    
def test_train_models(path_test_images_results):
    # Create sample data
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    churn_predictor = ChurnPredictor("")
    churn_predictor.X_train = X_train
    churn_predictor.X_test = X_test
    churn_predictor.y_train = y_train
    churn_predictor.y_test = y_test
    
    
    churn_predictor.train_models()
    
    # Check if the models are trained and saved properly
    assert os.path.exists(churn_predictor.MODEL_PATH)
    assert os.path.getsize(churn_predictor.MODEL_PATH) > 0
    with open(churn_predictor.MODEL_PATH, "rb") as f:
        clf_rf = joblib.load(f)
        clf_baseline = joblib.load(f)
        assert "RandomForestClassifier" in str(type(clf_rf))
        assert "LogisticRegression" in str(type(clf_baseline))