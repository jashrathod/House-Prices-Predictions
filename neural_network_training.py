import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from keras.losses import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pylab import rcParams
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, r2_score
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed
import warnings

warnings.filterwarnings('ignore')

### READ TRAINING DATA ###

data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
df = data.copy()

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

for i in df.columns:
    print(i, ': ( ', df[i].isna().sum(), ' NaNs ) ')

X = df.drop(['SalePrice'], axis=1)
Y = df['SalePrice']

x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=7)



nb_epoch = 2000
batch_size = 100
input_dim = x_train.shape[1]  # num of predictor variables,
dim = 64
hidden_dim_1 = int(dim / 2)
hidden_dim_2 = int(hidden_dim_1 / 2)
learning_rate = 0.001

input_layer = Input(shape=(input_dim,))
h1 = Dense(dim, activation='relu', activity_regularizer=regularizers.l1(learning_rate))(input_layer)
h2 = Dense(hidden_dim_1, activation='relu')(h1)
h3 = Dense(hidden_dim_2, activation='relu')(h2)
output_layer = Dense(1)(h3)

model = load_model('keras_model_1.h5')

# model = Model(inputs=input_layer, outputs=output_layer)
#
# model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
# cp = ModelCheckpoint(filepath="keras_model.h5", save_best_only=True, verbose=0)
# tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
#
#
# history = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size, shuffle=True,
#                           # validation_data=(x_valid, y_valid),
#                           verbose=1, callbacks=[cp, tb]).history
#
# model.save('keras_model_1.h5')

valid_x_predictions = model.predict(x_valid)
print('valid x predictions: ', type(valid_x_predictions))

y_pred = valid_x_predictions

y_pred = pd.DataFrame(y_pred)
y_valid = pd.DataFrame(y_valid)

a1 = mean_squared_error(y_valid, y_pred)
a2 = np.power(mean_squared_error(y_valid, y_pred), 0.5)
a3 = r2_score(y_valid, y_pred)

print('mse: ', a1)
print('rmse: ', a2)
print('R-squared: ', a3)  # higher the R-squared, the better the model fits data
print(input_dim)
