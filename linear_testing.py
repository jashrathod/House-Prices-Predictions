import pickle
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


### READ TESTING DATA ###

data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
df = data.copy()


### DROP UNWANTED FEATURES ###

df = df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1)


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


### PREPARING TESTING DATA ###

X = df.copy()


### LOADING SAVED MODEL ###

model = pickle.load(open('linear_regression_model_3.sav', 'rb'))
y_pred = model.predict(X)


### STORING RESULTS IN A .csv FILE ###

y_pred = pd.DataFrame({'SalePrice': y_pred})

e = y_pred.to_csv('submit3.csv')
