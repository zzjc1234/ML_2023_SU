import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.feature_selection as featsel

# import data
column_to_fill =['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','FireplaceQu','GarageFinish','GarageCond','PoolQC','Fence','MiscFeature']
fill_value = 'No'

# process Na
data = pd.read_csv('~/Desktop/ML_2023_SU/Contest/house-prices-advanced-regression-techniques/train.csv')
data[column_to_fill] = data[column_to_fill].fillna(fill_value)
data.dropna()

# display the Data
print(data.describe())
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(data.shape)
print(data.isnull().sum())

train = pd.get_dummies(data)
Y = train["SalePrice"]
X = train.drop('SalePrice', axis = 1)

X = X.to_numpy()
Y = Y.to_numpy()

# feature selection