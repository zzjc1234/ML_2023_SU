# import data
import pandas as pd
train_path = 'train.csv'
train_data = pd.read_csv(train_path)
train_data = pd.get_dummies(train_data)
#train_data.to_csv("testfile.csv",index=False)
X = train_data.drop('SalePrice', axis=1)
print(X.shape)
X = X.dropna(axis=1)
y=train_data["SalePrice"]

print(X.shape)
print(y.shape)
