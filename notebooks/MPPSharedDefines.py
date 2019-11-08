import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import math
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
import os

pd.set_option('display.max_columns', None)

DATA_PATH = os.path.realpath(os.getcwd()+"/../data/")+"/"  # Data source directory
TEMP_PATH = os.path.realpath(os.getcwd()+"/../temp/")+"/"  # Temporary directory

TRAIN_DATA_FN = 'train_values.csv'  # Train data CSV
TRAIN_LABELS_FN = "train_labels.csv"  # Label data CSV
CLEANED_OUTPUT_FN = "train_cleaned.csv"  # Cleaned training data
CLEANED_OUTPUT_FN_XLS = "train_cleaned.xlsx"  # Cleaned training data
TEST_VALUES_FN = "test_values.csv"  # Test value CSV