import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
from keras.models import load_model
from numpy.random import seed

seed(1)
import warnings

warnings.filterwarnings('ignore')


### READ TRAINING DATA ###

data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
df = data.copy()

### LABEL ENCODING ###

df1 = df.select_dtypes(include=['object']).copy()
df = df.drop(df1.columns, axis=1)

df1[pd.isnull(df1)] = 'NaN'

a1 = df1.copy()
for i in df1.columns:
    a1[i] = a1[i].astype('category')
    a1[i] = a1[i].cat.codes

df = df.join(a1)

# for i in df.columns:
#     df[i].fillna((df[i].mean()), inplace=True)

# for i in df.columns:
#     print(i, ': ( ', df[i].isna().sum(), ' NaNs ) ')

# X = df.drop(['SalePrice'], axis=1)
# Y = df['SalePrice']
#
# x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=7)

x = df.copy()

# nb_epoch = 2000
# batch_size = 64
# input_dim = x_train.shape[1]  # num of predictor variables,
# dim = 64
# hidden_dim_1 = int(dim / 2)
# hidden_dim_2 = int(hidden_dim_1 / 2)
# learning_rate = 0.001
#
# input_layer = Input(shape=(input_dim,))
# h1 = Dense(dim, activation='relu', activity_regularizer=regularizers.l1(learning_rate))(input_layer)
# h2 = Dense(hidden_dim_1, activation='relu')(h1)
# h3 = Dense(hidden_dim_2, activation='relu')(h2)
# output_layer = Dense(1)(h3)

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

valid_x_predictions = model.predict(x)

y_pred = pd.DataFrame(valid_x_predictions, columns=['SalePrice'])

for i in y_pred.columns:
    y_pred[i].fillna((y_pred[i].mean()), inplace=True)

e = y_pred.to_csv('submit.csv')
