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

data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
print(len(data))
df = data.copy()

df = df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu',
              'LotFrontage', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond'], axis=1)


# df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

df1 = df.select_dtypes(include=['object']).copy()

df = df.drop(df1.columns, axis=1)

a1 = df1.copy()
for i in df1.columns:
    a1[i] = a1[i].astype('category')
    a1[i] = a1[i].cat.codes

df = df.join(a1)
for i in df.columns:
    df[i].fillna((df[i].mean()), inplace=True)

X = df.copy()

model = pickle.load(open('linear_regression_model_2.sav', 'rb'))
y_pred = model.predict(X)

y_pred = pd.DataFrame({'SalePrice': y_pred})
print(len(y_pred))

e = y_pred.to_csv('submit3.csv', header=True)
print(y_pred)
