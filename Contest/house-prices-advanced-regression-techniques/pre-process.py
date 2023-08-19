import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.feature_selection as featsel


def remove_low_variance_features(data, threshold=0.01):
    variance_threshold = featsel.VarianceThreshold(threshold=threshold)
    data_without_low_variance = variance_threshold.fit_transform(data)
    return data_without_low_variance

def remove_constant_features(data):
    constant_feature_indices = []  # 用于存储常量特征的列索引

    for col in data.columns:
        unique_values = data[col].unique()
        if len(unique_values) == 1 or (len(unique_values) == 2 and pd.isna(unique_values).any()):
            # 如果特征的元素只有一个或有两个但包括NaN，认为是常量特征
            constant_feature_indices.append(data.columns.get_loc(col))

    # 从数据中移除常量特征
    data_without_constants = data.drop(data.columns[constant_feature_indices], axis=1)

    return data_without_constants, constant_feature_indices

# INFO: import data
column_to_fill =['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','FireplaceQu','GarageFinish','GarageCond','PoolQC','Fence','MiscFeature']
fill_value = "No"

# INFO: process Na
data = pd.read_csv('~/Desktop/ML_2023_SU/Contest/house-prices-advanced-regression-techniques/train.csv')
data[column_to_fill] = data[column_to_fill].fillna(fill_value)
data, index=remove_constant_features(data)
data.dropna(inplace = True)

# INFO: display the Data
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

# INFO: feature selection

# TEST: Built-in test

# WARNING: Since the variance of dumb variable is really small, these features are removed

T=remove_low_variance_features(X)

f_scores, _ = featsel.f_classif(T, Y)
mi_scores = featsel.mutual_info_classif(T, Y)
chi2_scores, _ = featsel.chi2(T, Y)

f_scores/= np.max(f_scores)
mi_scores/=np.max(mi_scores)
chi2_scores/=np.max(chi2_scores)

plt.figure(figsize=(10, 6))
plt.plot(range(len(f_scores)), f_scores, marker='o', label='F-test')
plt.plot(range(len(mi_scores)), mi_scores, marker='s', label='Mutual Info Test')
plt.plot(range(len(chi2_scores)), chi2_scores, marker='^', label='Chi2 Test')

plt.xlabel('Feature Index')
plt.ylabel('Score')
plt.title('Feature Selection Scores Comparison')
plt.xticks(range(T.shape[1]), [f'Feature {i}' for i in range(T.shape[1])])
plt.legend()
plt.grid()
plt.show()

# TEST: Recursive Feature Elimination


