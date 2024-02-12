import logging
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from churn_library import *

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
def df_test_churn():
    return pd.DataFrame({"Attrition_Flag": ["Existing Customer", "Existing Customer", "Attrited Customer"]})

@pytest.fixture
def sample_df():
    return pd.DataFrame({"col1": [1, 2, 3, 4, 5], 
                         "col2": [2, 4, 6, 8, 10],
                         "col3": ["A", "B", "C", "D", "E"]})

def test_import_data_returns_dataframe(path):
    data = import_data(path)
    assert isinstance(data, pd.DataFrame)
    
def test_import_data_raises_exception():
    with pytest.raises(FileNotFoundError):
        import_data("non_existent_file.csv")
        
def test_import_data_reads_csv(path):
    with patch("pandas.read_csv") as mock_csv:
        import_data(path)
        mock_csv.assert_called_once_with(path)

def test_add_churn_col():
    df_test_churn = pd.DataFrame({"Attrition_Flag": ["Existing Customer", "Existing Customer", "Attrited Customer"]})
    extended_df = add_target_col_to_df(df_test_churn)
    
    assert extended_df["Churn"].tolist() == [0, 0, 1]
        
def test_perform_eda(sample_df):
    test_path_plots = "test_images/eda"
    perform_eda(sample_df, test_path_plots)

    assert os.path.exists(os.path.join(test_path_plots, "hist_col1.png"))
    assert os.path.exists(os.path.join(test_path_plots, "hist_col2.png"))
    assert os.path.exists(os.path.join(test_path_plots, "bar_col3.png"))
    assert os.path.exists(os.path.join(test_path_plots, "feature_heatmap.png"))

    shutil.rmtree(test_path_plots)
    
def test_get_categorical_cols():
    df = pd.DataFrame({"col1": ["A", "B", "C"], 
                       "col2": [1, 2, 3], 
                       "col3": pd.Categorical(["X", "Y", "X"])})
    
    assert get_categorical_cols(df) == ["col1", "col3"]

def test_encoder_helper():
    df = pd.DataFrame({"Gender": ["M", "F", "M", "M", "F"], 
                       "Churn": [1, 0, 1, 1, 1],
                       "num_col1": [1, 2, 3, 4, 5]})
    
    df_encoded = encoder_helper(df, ["Gender"], "Churn")
    
    assert sorted(df_encoded.columns) == ['Churn', 'Gender_Churn', 'num_col1']
    assert (df_encoded["Gender_Churn"] == [1.0, 0.5, 1.0, 1.0, 0.5]).all()
    
def test_perform_feature_engineering():
    # Test perform_feature_engineering function
    # Create a sample dataframe
    df = pd.DataFrame({'cat_col1': ['A', 'B', 'C'], 
                       'cat_col2': ['X', 'Y', 'X'],
                       'num_col1': [1, 2, 3],
                       'Churn': [1, 0, 1]})
    
    # Test the function output
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(X_train) == 2
    assert len(X_test) == 1
    assert len(y_train) == 2
    assert len(y_test) == 1