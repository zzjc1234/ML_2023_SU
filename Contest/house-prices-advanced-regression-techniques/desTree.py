#Import Module
from sklearn.tree import DecisionTreeRegressor as dt
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
import numpy as np

#Set to print the whole table
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Import Data
train_path = 'train.csv'
test_path = 'test.csv'

train_data = pd.read_csv(train_path)
test_data  = pd.read_csv(test_path)
feat = test_data.columns
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

#Get Summary
# print(train_data.describe())

# Process Data
X=train_data.to_numpy()
Y=X[:, -1]
X=X[:, :-1]

#Fit Model
train_X, val_X, train_y, val_y = tts(X, Y, random_state=1)
model=dt()
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=k_fold)
absolute_mae_scores = -mae_scores
average_mae = absolute_mae_scores.mean()

print("Average MAE:", average_mae)