from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import numpy as np

train_path = 'train.csv'
#feat=pd.
test_path = 'test.csv'

train_data = pd.read_csv(train_path)
test_data  = pd.read_csv(test_path)

feat = test_data.columns

print(train_data.describe())
print(feat)

#train_X, val_X, train_y, val_y = tts(train_data, y, random_state=1)
