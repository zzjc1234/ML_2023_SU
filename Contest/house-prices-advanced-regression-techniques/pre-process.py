import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.feature_selection as featsel

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

def import_data():
# INFO: mark special data
    column_to_fill =['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','FireplaceQu','GarageFinish','GarageCond','PoolQC','Fence','MiscFeature']
    fill_value = "No"

# INFO: import data
    data = pd.read_csv('~/Desktop/ML_2023_SU/Contest/house-prices-advanced-regression-techniques/train.csv')

# INFO: process Na
    data[column_to_fill] = data[column_to_fill].fillna(fill_value)
    data, _ =remove_constant_features(data)
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

    return X,Y

# INFO: Use the buildtInTest to analyze the feature
def built_in_test(X,Y):
# WARNING: Since the variance of dumb variable is really small, these features are removed
    f_scores, _ = featsel.f_regression(X, Y)
    mi_scores = featsel.mutual_info_regression(X, Y)
    chi2_scores = featsel.r_regression(X, Y)

    f_scores/= np.max(f_scores)
    mi_scores/=np.max(mi_scores)
    chi2_scores/=np.max(chi2_scores)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(f_scores)), f_scores, marker='o', label='F-Regression')
    plt.plot(range(len(mi_scores)), mi_scores, marker='s', label='Mutual Info Regression')
    plt.plot(range(len(chi2_scores)), chi2_scores, marker='^', label='Chi2 Regression')

    plt.xlabel('Feature Index')
    plt.ylabel('Score')
    plt.title('Feature Selection Scores Comparison')
    plt.xticks(range(X.shape[1]), [f'Feature {i}' for i in range(X.shape[1])])
    plt.legend()
    plt.grid()
    plt.show()

# INFO: subplot
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Plot F-Regression on the first subplot
    axes[0, 0].plot(range(len(f_scores)), f_scores, marker='o', label='F-Regression', color='blue')
    axes[0, 0].set_title('F-Regression')
    axes[0, 0].set_xlabel('Feature Index')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()

# Plot Mutual Info Regression on the second subplot
    axes[0, 1].plot(range(len(mi_scores)), mi_scores, marker='s', label='Mutual Info Regression', color='green')
    axes[0, 1].set_title('Mutual Info Regression')
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()

# Plot Chi2 Regression on the third subplot
    axes[1, 0].plot(range(len(chi2_scores)), chi2_scores, marker='^', label='Chi2 Regression', color='orange')
    axes[1, 0].set_title('Chi2 Regression')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()

# Plot the overlay of the first three plots on the fourth subplot
    axes[1, 1].plot(range(len(f_scores)), f_scores, marker='o', label='F-Regression', color='blue')
    axes[1, 1].plot(range(len(mi_scores)), mi_scores, marker='s', label='Mutual Info Regression', color='green')
    axes[1, 1].plot(range(len(chi2_scores)), chi2_scores, marker='^', label='Chi2 Regression', color='orange')
    axes[1, 1].set_title('Combined Scores')
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()

    plt.tight_layout()

    plt.show()

# INFO: Use REF to select features
def refSel(X,Y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    min_features_to_select = 1  # Minimum number of features to consider
    clf = LogisticRegression()
    cv = StratifiedKFold(5)

    rfecv = featsel.RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
        n_jobs=2,
    )
    rfecv.fit(X, Y)

    print(f"Optimal number of features: {rfecv.n_features_}")

    n_scores = len(rfecv.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.plot(
        range(min_features_to_select, n_scores + min_features_to_select),
        rfecv.cv_results_["mean_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()

X, Y=import_data()

# INFO: feature selection

# TEST: Built-in test

#built_in_test(X,Y)

# TEST: REF

refSel(X,Y)
