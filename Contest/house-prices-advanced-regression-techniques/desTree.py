#Import Module
from sklearn.tree import DecisionTreeRegressor as dt
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error as mae
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
print(train_data.describe())

# Process Data
X=train_data.to_numpy()
Y=X[:, -1]
X=X[:, :-1]
# print(X)
# print(Y)

#Fit Model
train_X, val_X, train_y, val_y = tts(X, Y, random_state=1)
model=dt(random_state=3)
model.fit(train_X,train_y)

#Validation
val_predictions = model.predict(val_X)
val_mae = mae(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))