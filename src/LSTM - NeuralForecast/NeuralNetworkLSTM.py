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

# Função para transformar dataset em janelas (X) e rótulos (Y)
def dataset_to_X_Y(dataset, windows_size):
    dataset_as_np = dataset.to_numpy()
    X = []
    Y = []
    for i in range(len(dataset_as_np) - windows_size):
        row = [[a] for a in dataset_as_np[i:i+windows_size]]
        X.append(row)
        label = dataset_as_np[i+windows_size]
        Y.append(label)
    return np.array(X), np.array(Y)

WINDOW_SIZE = 5
X, Y = dataset_to_X_Y(ppv, WINDOW_SIZE)

# Dividir em treino, validação e teste
X_train, Y_train = X[:600], Y[:600]
X_val, Y_val = X[600:700], Y[600:700]
X_test, Y_test = X[700:], Y[700:]

# Importar os pacotes corretos
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

# Construir o modelo LSTM
model1 = Sequential()
model1.add(InputLayer((WINDOW_SIZE, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

# Resumo do modelo
model1.summary()

# Compilar o modelo
cp = ModelCheckpoint('model1/model_checkpoint.keras', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0005), metrics=[RootMeanSquaredError()])

# Treinamento do modelo
model1.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, callbacks=[cp])

train_predictions = model1.predict(X_train).flatten()
train_results = pd.DataFrame(data = {'Train Predictions':train_predictions, 'Actuals':Y_train})

val_predictions = model1.predict(X_val).flatten()
val_results = pd.DataFrame(data = {'Val Predictions':val_predictions, 'Actuals':Y_val})

test_predictions = model1.predict(X_test).flatten()
test_results = pd.DataFrame(data = {'Test Predictions':test_predictions, 'Actuals':Y_test})

# Plotar os resultados de treino
plt.figure(figsize=(10, 6))
plt.plot(train_results['Train Predictions'], label='Previsões do Treino', color='blue', linestyle='--')
plt.plot(train_results['Actuals'], label='Valores Reais', color='green')
plt.title('Previsões do Treino vs Valores Reais')
plt.xlabel('Índice')
plt.ylabel('Ppv')
plt.legend()
plt.show()

# Plotar os resultados de validação
plt.figure(figsize=(10, 6))
plt.plot(val_results['Val Predictions'], label='Previsões da Validação', color='orange', linestyle='--')
plt.plot(val_results['Actuals'], label='Valores Reais', color='green')
plt.title('Previsões da Validação vs Valores Reais')
plt.xlabel('Índice')
plt.ylabel('Ppv')
plt.legend()
plt.show()

# Plotar os resultados de teste
plt.figure(figsize=(10, 6))
plt.plot(test_results['Test Predictions'], label='Previsões do Teste', color='red', linestyle='--')
plt.plot(test_results['Actuals'], label='Valores Reais', color='green')
plt.title('Previsões do Teste vs Valores Reais')
plt.xlabel('Índice')
plt.ylabel('Ppv')
plt.legend()
plt.show()
