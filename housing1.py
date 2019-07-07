import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
df = data.copy()

p = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']

q = ['LotFrontage', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType',
     'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']

# for i in q:
#     print(i, ': ', data[i].unique())
#     print(df.isna().sum()[i])

# print(len(data))

df = df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu',
              'LotFrontage', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond'], axis=1)

df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

# for i in df.columns:
#     print(i, ': ', df[i].unique())


df1 = df.select_dtypes(include=['object']).copy()

df = df.drop(df1.columns, axis=1)

a1 = df1.copy()
for i in df1.columns:
    a1[i] = a1[i].astype('category')
    a1[i] = a1[i].cat.codes


# df_y = df['SalePrice']
# df = df.drop(['SalePrice'], axis=1)

df = df.join(a1)


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# scaler = MinMaxScaler()
# df = scaler.fit_transform(df)
#
# df = pd.DataFrame(df)
# df_y = pd.DataFrame(df_y)

# df = normalize(df)

# df = df.join(df_y)

# df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

# print(df.info())


# plt.figure(figsize=(12,10))
cor = df.corr()
# sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
# plt.show()

# Correlation with output variable
cor_target = abs(cor['SalePrice'])

# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.5]
print(relevant_features)
print(len(relevant_features))


# df = df[['OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath',
#          'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'SalePrice', 'ExterQual', 'BsmtQual', 'KitchenQual']]

X = df.drop(['SalePrice'], axis=1)
Y = df['SalePrice']


# Alley, PoolQC, Fence, MiscFeature

# FireplaceQu

# 'LotFrontage', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
# 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond'


# LotFrontage - (259 NaN) - int -- TRY TAKING MEAN
# GarageType - (81 NaN) - object
# GarageYrBlt - (81 NaN) - object
# GarageCond - (81 NaN) - object
# GarageQual - (81 NaN) - object
# GarageFinish - (81 NaN) - object
# BsmtQual - (37 NaN) - object
# BsmtCond - (38 NaN) - object
# BsmtExposure - (37 NaN) - object
# BsmtFinType1 - (37 NaN) - object
# BsmtFinType2 - (38 NaN) - object


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# model = LinearRegression()
# model.fit(x_train, y_train)
#
# y_pred = model.predict(x_test)



model = pickle.load(open('linear_regression_model.sav', 'rb'))
y_pred = model.predict(x_test)

# filename = 'linear_regression_model.sav'
# pickle.dump(model, open(filename, 'wb'))



print(type(y_pred))
print(type(y_test))

df_ans= pd.DataFrame({'y_pred': y_pred, 'y_test': y_test})
# df_ans = y_test.join(y_pred)
# print(df_ans)

y_pred = pd.DataFrame(y_pred)
y_test = pd.DataFrame(y_test)

a1 = mean_squared_error(y_test, y_pred)
a2 = np.power(mean_squared_error(y_test, y_pred), 0.5)
a3 = r2_score(y_test, y_pred)

print('mse: ', a1)
print('rmse: ', a2)
print('R-squared: ', a3)  # higher the R-squared, the better the model fits data
