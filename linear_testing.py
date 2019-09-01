import pickle
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


### READ TESTING DATA ###

data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
df = data.copy()


### DROP UNWANTED FEATURES ###

f1 = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']
f2 = ['LotFrontage', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']
f3 = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
f4 = ['MasVnrType', 'MasVnrArea', 'Electrical']

df = df.drop(f1, axis=1)
df = df.drop(f2, axis=1)


### REMOVING NaNs ###

# df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]


### LABEL ENCODING ###

df1 = df.select_dtypes(include=['object']).copy()
df = df.drop(df1.columns, axis=1)

for i in df1.columns:
    df1[i].fillna((df1[i].mode()), inplace=True)

for i in df.columns:
    df[i].fillna((df[i].median()), inplace=True)

a1 = df1.copy()
for i in df1.columns:
    a1[i] = a1[i].astype('category')
    a1[i] = a1[i].cat.codes

df = df.join(a1)


### NORMALIZING AND SCALING ###

# df_y = df['SalePrice']
# df = df.drop(['SalePrice'], axis=1)


### Scaling the Data ###

# scaler = MinMaxScaler()
# df = scaler.fit_transform(df)
# df = pd.DataFrame(df)


### Normalizing the Data ###

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# df = normalize(df)


# df = pd.DataFrame(df)
# df_y = pd.DataFrame(df_y)
# df = df.join(df_y)


### REPLACING NaNs WITH MEAN VALUE ###

# for i in df.columns:
#     df[i].fillna((df[i].mean()), inplace=True)


### PREPARING TESTING DATA ###

X = df.copy()


### LOADING SAVED MODEL ###

model = pickle.load(open('linear_regression_model_2.sav', 'rb'))
y_pred = model.predict(X)


### STORING RESULTS IN A .csv FILE ###

y_pred = pd.DataFrame({'SalePrice': y_pred})

e = y_pred.to_csv('submit.csv')
