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


### READ TRAINING DATA ###

data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
df = data.copy()


### UNIQUE VALUES OF FEATURES ###

for i in df.columns:
    print(i, ': ( ', data[i].isna().sum(), ' NaNs ) ', data[i].unique())


### FEATURES TO BE DROPPED ###

# Alley - (1369 NaN) - object
# FireplaceQu - (690 NaN) - object
# PoolQC - (1453 NaN) - object
# Fence - (1179 NaN) - object
# MiscFeature - (1406 NaN) - object

# LotFrontage - (259 NaN) - int
# GarageType - (81 NaN) - object
# GarageYrBlt - (81 NaN) - object
# GarageCond - (81 NaN) - object
# GarageQual - (81 NaN) - object
# GarageFinish - (81 NaN) - object

# BsmtCond - (38 NaN) - object
# BsmtFinType2 - (38 NaN) - object
# BsmtExposure - (37 NaN) - object
# BsmtFinType1 - (37 NaN) - object
# BsmtQual - (37 NaN) - object

# MasVnrType - (8 NaN) - object
# MasVnrArea - (8 NaN) - float
# Electrical - (1 NaN) - object


f1 = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']
f2 = ['LotFrontage', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']
f3 = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
f4 = ['MasVnrType', 'MasVnrArea', 'Electrical']

df = df.drop(f1, axis=1)
# df = df.drop(f2, axis=1)

### REMOVING NaNs ###

# df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]


### LABEL ENCODING ###

df1 = df.select_dtypes(include=['object']).copy()
df = df.drop(df1.columns, axis=1)

a1 = df1.copy()
for i in df1.columns:
    a1[i] = a1[i].astype('category')
    a1[i] = a1[i].cat.codes

df = df.join(a1)


### REPLACING NaNs WITH MEAN VALUE ###

for i in df.columns:
    df[i].fillna((df[i].mean()), inplace=True)


### NORMALIZING AND SCALING ###

# df_y = df['SalePrice']
# df = df.drop(['SalePrice'], axis=1)

### Normalizing the Data ###

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# df = normalize(df)


### Scaling the Data ###

# scaler = MinMaxScaler()
# df = scaler.fit_transform(df)


# df = pd.DataFrame(df)
# df_y = pd.DataFrame(df_y)
# df = df.join(df_y)


### REMOVING NaNs ###

# df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]


### FEATURE EXTRACTION - USING PEARSON'S CORRELATION ###

cor = df.corr()
# plt.figure(figsize=(12,10))
# sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
# plt.show()

# Correlation with output variable
cor_target = abs(cor['SalePrice'])

# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.5]
print('Relevant Features: \n', relevant_features)

# OverallQual     0.790982
# YearBuilt       0.522897
# YearRemodAdd    0.507101
# TotalBsmtSF     0.613581
# 1stFlrSF        0.605852
# GrLivArea       0.708624
# FullBath        0.560664
# TotRmsAbvGrd    0.533723
# GarageCars      0.640409
# GarageArea      0.623431
# SalePrice       1.000000
# ExterQual       0.636884
# KitchenQual     0.589189

print('No. of Relevant Features:', len(relevant_features))  # 13


### PREPARING TRAINING AND VALIDATION DATA ###

X = df.drop(['SalePrice'], axis=1)
Y = df['SalePrice']

x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=7)

print('X_TRAIN: ', x_train.shape)
print('X_VALID: ', x_valid.shape)
print('Y_TRAIN: ', y_train.shape)
print('Y_VALID: ', y_valid.shape)


### DEFINING AND TRAINING MODEL ###

# model = LinearRegression()
# model.fit(x_train, y_train)


### PREDICTIONS ###

# y_pred = model.predict(x_valid)


### SAVING MODEL ###

# filename = 'linear_regression_model_3.sav'
# pickle.dump(model, open(filename, 'wb'))


### LOADING SAVED MODEL ###

model = pickle.load(open('linear_regression_model_2.sav', 'rb'))
y_pred = model.predict(x_valid)


### VALIDATING MODEL PERFORMANCE ###

df_ans = pd.DataFrame({'y_pred': y_pred, 'y_valid': y_valid})

a1 = mean_squared_error(df_ans['y_valid'], df_ans['y_pred'])
a2 = np.power(mean_squared_error(df_ans['y_valid'], df_ans['y_pred']), 0.5)
a3 = r2_score(df_ans['y_valid'], df_ans['y_pred'])

print('mse: ', a1)
print('rmse: ', a2)
print('R-squared: ', a3)  # higher the R-squared, the better the model fits data
