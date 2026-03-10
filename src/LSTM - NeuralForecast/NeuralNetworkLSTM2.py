import warnings
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

os.environ['KERAS_BACKEND'] = "tensorflow"

# Desativar warnings do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.keras.backend.clear_session()
import pandas as pd
import numpy as np
import keras

# Carregar os dados
dataset = pd.read_csv('lstm.csv', index_col='Timestamp', parse_dates=True)

# Escolher a coluna desejada
ppv = dataset['Ppv']

ppv = pd.DataFrame({'Ppv':ppv})
ppv['Seconds'] = ppv.index.map(pd.Timestamp.timestamp)

day = 60*60*24
year = 365.2425*day

ppv['Day sin'] = np.sin(ppv['Seconds'] * (2 * np.pi / day))
ppv['Day cos'] = np.cos(ppv['Seconds'] * (2 * np.pi / day))
ppv['Year sin'] = np.sin(ppv['Seconds'] * (2 * np.pi / year))
ppv['Year cos'] = np.cos(ppv['Seconds'] * (2 * np.pi / year))

ppv = ppv.drop('Seconds', axis=1)

# Função para transformar dataset em janelas (X) e rótulos (Y)
def dataset_to_X_Y2(dataset, windows_size):
    dataset_as_np = dataset.to_numpy()
    X = []
    Y = []
    for i in range(len(dataset_as_np) - windows_size):
        row = [r for r in dataset_as_np[i:i+windows_size]]
        X.append(row)
        label = dataset_as_np[i+windows_size][0]
        Y.append(label)
    return np.array(X), np.array(Y)

WINDOW_SIZE = 6
X2, Y2 = dataset_to_X_Y2(ppv, WINDOW_SIZE)

# Dividir em treino, validação e teste
X2_train, Y2_train = X2[:600], Y2[:600]
X2_val, Y2_val = X2[600:700], Y2[600:700]
X2_test, Y2_test = X2[700:], Y2[700:]

ppv_training_mean = np.mean(X2_train[:, :, 0])
ppv_training_std = np.std(X2_train[:, :, 0])

def preprocess(X):
    X[:, :, 0] = (X[:, :, 0] - ppv_training_mean) / ppv_training_std
    return X

preprocess(X2_train)
preprocess(X2_val)
preprocess(X2_test)


# Importar os pacotes corretos
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

# Construir o modelo LSTM
model2 = Sequential()
model2.add(InputLayer((WINDOW_SIZE, 5)))
model2.add(LSTM(64))
model2.add(Dense(8, 'relu'))
model2.add(Dense(1, 'linear'))

# Resumo do modelo
model2.summary()

# Compilar o modelo
cp2 = ModelCheckpoint('model2/model_checkpoint.keras', save_best_only=True)
model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0005), metrics=[RootMeanSquaredError()])

# Treinamento do modelo
model2.fit(X2_train, Y2_train, validation_data=(X2_val, Y2_val), epochs=100, callbacks=[cp2])

train2_predictions = model2.predict(X2_train).flatten()
train2_results = pd.DataFrame(data = {'Train Predictions':train2_predictions, 'Actuals':Y2_train})

val2_predictions = model2.predict(X2_val).flatten()
val2_results = pd.DataFrame(data = {'Val Predictions':val2_predictions, 'Actuals':Y2_val})

test2_predictions = model2.predict(X2_test).flatten()
test2_results = pd.DataFrame(data = {'Test Predictions':test2_predictions, 'Actuals':Y2_test})

# Plotar os resultados de treino
plt.figure(figsize=(10, 6))
plt.plot(train2_results['Train Predictions'], label='Previsões do Treino', color='blue', linestyle='--')
plt.plot(train2_results['Actuals'], label='Valores Reais', color='green')
plt.title('Previsões do Treino vs Valores Reais')
plt.xlabel('Índice')
plt.ylabel('Ppv')
plt.legend()
plt.show()

# Plotar os resultados de validação
plt.figure(figsize=(10, 6))
plt.plot(val2_results['Val Predictions'], label='Previsões da Validação', color='orange', linestyle='--')
plt.plot(val2_results['Actuals'], label='Valores Reais', color='green')
plt.title('Previsões da Validação vs Valores Reais')
plt.xlabel('Índice')
plt.ylabel('Ppv')
plt.legend()
plt.show()

# Plotar os resultados de teste
plt.figure(figsize=(10, 6))
plt.plot(test2_results['Test Predictions'], label='Previsões do Teste', color='red', linestyle='--')
plt.plot(test2_results['Actuals'], label='Valores Reais', color='green')
plt.title('Previsões do Teste vs Valores Reais')
plt.xlabel('Índice')
plt.ylabel('Ppv')
plt.legend()
plt.show()