import logging
import pytest
import pandas as pd
from unittest.mock import patch
from churn_library import *

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture
def path():
    """
    Return the path to the test csv data file.
    """
    return r"./data/bank_data.csv"

@pytest.fixture
def sample_df():
    return pd.DataFrame({'col1': [1, 2, 3, 4, 5], 
                         'col2': [2, 4, 6, 8, 10],
                         'col3': ['A', 'B', 'C', 'D', 'E']})

def test_import_data_returns_dataframe(path):
    data = import_data(path)
    assert isinstance(data, pd.DataFrame)
    
def test_import_data_raises_exception():
    with pytest.raises(FileNotFoundError):
        import_data('non_existent_file.csv')
        
def test_import_data_reads_csv(path):
    with patch("pandas.read_csv") as mock_csv:
        import_data(path)
        mock_csv.assert_called_once_with(path)
        
def test_perform_eda(sample_df):
    # create temporary directory for test images
    test_path_plots = 'test_images/eda'

    # run the function
    perform_eda(sample_df, test_path_plots)

    # check that the expected files have been created
    assert os.path.exists(os.path.join(test_path_plots, "hist_col1.png"))
    assert os.path.exists(os.path.join(test_path_plots, "hist_col2.png"))
    assert os.path.exists(os.path.join(test_path_plots, "bar_col3.png"))
    assert os.path.exists(os.path.join(test_path_plots, "feature_heatmap.png"))

    # cleanup the directory after the test
    shutil.rmtree(test_path_plots)        