# -*- coding: utf-8 -*-
"""BAWA_Stepup_Loading_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17LQgMZbQvupdS8vjrs6ZTDs668779cOO
"""



"""LOADS SHIT THAT NEEDS TO DO OTHER SHIT"""

import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from google.colab import drive
drive.mount('/content/drive')

import os

# Path to the directory
directory_path = '/content/drive/My Drive/94_betting_model/'

# List all files in the directory
file_list = os.listdir(directory_path)
print(file_list)