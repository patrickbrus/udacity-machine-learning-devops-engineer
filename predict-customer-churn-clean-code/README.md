# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, the aim is to identify credit card customers that are most likely to churn. Churn, in this context, refers to customers who stop using their credit card or close their credit card account. Identifying customers who are likely to churn can helps to take appropriate actions to retain them, thereby increasing customer loyalty and revenue.

The project includes a Python package that uses machine learning to predict credit card customer churn. The package follows coding best practices, such as PEP8 style guide and modular, documented, and tested code.

The dataset for this project was pulled from Kaggle and contains information about credit card customers, including demographic information, credit card usage, and payment history. This dataset is used to train and test machine learning models that can predict which customers are most likely to churn.

## Project Structure

The project code is organized into the following modules and directories:

- `churn_notebook.ipynb`: A Jupyter notebook that contains the initial solution for identifying credit card customers that are most likely to churn.
- `churn_library.py`: The main module that defines the `ChurnPredictor` class, which encapsulates the machine learning models and their training and evaluation procedures. The module also defines functions for loading, cleaning, and transforming the dataset.
- `data`: A directory that contains the dataset used in the project (`data/bank_data.csv`).
- `images`: A directory that contains the images generated during the analysis and evaluation of the models. The images are organized into subdirectories (`images/eda` and `images/results`) according to their purpose.
- `logs`: A directory that contains the log file generated during the execution of the package (`logs/churn_library.log`).
- `models`: A directory that contains the trained machine learning models saved to disk (`models/logistic_regression.pkl` and `models/random_forest.pkl`).
- `test_churn_script_logging_and_tests.py`: A module that defines unit tests for the various functions and methods in the project.
- `requirements_py3.6.txt` and `requirements_py3.8.txt`: Text files that list the dependencies required by the project for Python 3.6 and 3.8, respectively.
- `Dockerfile`: Dockerfile to create an image that runs the churn predictor training when started. The build process only succeeds in case the unit tests are successful

## Installation

To install the package and its dependencies, run the following command:

```bash
pip install -r requirements_py3.6.txt # for Python3.6
pip install -r requirements_py3.8.txt # for Python3.8
```

## Usage

To use the package, import the `ChurnPredictor` class from the `churn_library` module and create an instance of it. 

You can check the `churn_library.py` file to get more information about how to use that module. You can also run this file, as it contains a main function when executed:

```bash
cd /path/to/this/folder
python churn_library.py
```

If you want to run the unittests, then please run the following command inside a terminal:

```bash
cd /path/to/this/folder
pytest
```

Alternatively, I have added a Dockerfile as well in case you would like to run everything with Docker. The docker build step will only succeed, if the 
unit tests succeed. Running the docker container will then execute the training process.

```bash
docker build -t churn-predictor .
docker run --rm -v ./images:/app/images -v ./models:/app/models -v ./logs:/app/logs churn-predictor
```

