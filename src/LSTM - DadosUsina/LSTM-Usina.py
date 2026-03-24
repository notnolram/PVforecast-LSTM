import warnings
import os
import matplotlib.pyplot as plt
from pathlib import Path

os.environ['KERAS_BACKEND'] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.keras.backend.clear_session()

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


# =========================
# 1. Carregar os dados
# =========================
arquivo = Path("data/Usina/Dados_coletados_csv/dados_consolidados_2024_2025.csv")
dataset = pd.read_csv(arquivo)

dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values('Date')
dataset = dataset.set_index('Date')

# Tratar zeros
dataset.loc[dataset['Generation(kWh)'] == 0, 'Generation(kWh)'] = np.nan
dataset['Generation(kWh)'] = dataset['Generation(kWh)'].interpolate()

# Adicionar sazonalidade
dataset['dayofyear'] = dataset.index.dayofyear
dataset['sin_doy'] = np.sin(2 * np.pi * dataset['dayofyear'] / 365)
dataset['cos_doy'] = np.cos(2 * np.pi * dataset['dayofyear'] / 365)

dataset['lag1'] = dataset['Generation(kWh)'].shift(1)

# Selecionar colunas de entrada
data = dataset[['Generation(kWh)', 'sin_doy', 'cos_doy', 'lag1']]
data = data.dropna()


# =========================
# 2. Função para criar janelas
# =========================
def dataset_to_X_Y(dataset, window_size):
    data_as_np = dataset.to_numpy()
    X = []
    Y = []

    for i in range(len(data_as_np) - window_size):
        X.append(data_as_np[i:i+window_size])
        Y.append(data_as_np[i+window_size, 0])  # prever apenas Generation(kWh)

    return np.array(X), np.array(Y)


# =========================
# 3. Parâmetros
# =========================
WINDOW_SIZE = 15


# =========================
# 4. Divisão temporal
# =========================
n = len(data)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = data.iloc[:train_end]
val = data.iloc[train_end:val_end]
test = data.iloc[val_end:]


# =========================
# 5. Normalização sem vazamento
# =========================
scaler = MinMaxScaler()

train_scaled = pd.DataFrame(
    scaler.fit_transform(train),
    index=train.index,
    columns=train.columns
)

val_scaled = pd.DataFrame(
    scaler.transform(val),
    index=val.index,
    columns=val.columns
)

test_scaled = pd.DataFrame(
    scaler.transform(test),
    index=test.index,
    columns=test.columns
)
# =========================
# 6. Criar janelas
# =========================
X_train, Y_train = dataset_to_X_Y(train_scaled, WINDOW_SIZE)
X_val, Y_val = dataset_to_X_Y(val_scaled, WINDOW_SIZE)
X_test, Y_test = dataset_to_X_Y(test_scaled, WINDOW_SIZE)

print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_val:", X_val.shape, "Y_val:", Y_val.shape)
print("X_test:", X_test.shape, "Y_test:", Y_test.shape)


# =========================
# 7. Construir o modelo
# =========================
model1 = Sequential()
model1.add(InputLayer((WINDOW_SIZE, X_train.shape[2])))
model1.add(LSTM(32))
model1.add(Dropout(0.2))
model1.add(Dense(8, activation='relu'))
model1.add(Dense(1, activation='linear'))

model1.summary()

model1.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=0.0005),
    metrics=[RootMeanSquaredError()]
)


# =========================
# 8. Callbacks
# =========================
os.makedirs("model1", exist_ok=True)

cp = ModelCheckpoint(
    'model1/model_checkpoint.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

es = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)


# =========================
# 9. Treinar
# =========================
history = model1.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=100,
    batch_size=16,
    callbacks=[cp, es],
    verbose=1
)


# =========================
# 10. Previsões
# =========================
train_predictions = model1.predict(X_train).flatten()
val_predictions = model1.predict(X_val).flatten()
test_predictions = model1.predict(X_test).flatten()


# =========================
# 11. Desnormalizar
# =========================
# =========================
# 11. Desnormalizar
# =========================
def inverse_generation_only(y_scaled, scaler, n_features):
    temp = np.zeros((len(y_scaled), n_features))
    temp[:, 0] = y_scaled  # geração está na coluna 0
    return scaler.inverse_transform(temp)[:, 0]

n_features = X_train.shape[2]

train_predictions_inv = inverse_generation_only(train_predictions, scaler, n_features)
val_predictions_inv = inverse_generation_only(val_predictions, scaler, n_features)
test_predictions_inv = inverse_generation_only(test_predictions, scaler, n_features)

Y_train_inv = inverse_generation_only(Y_train, scaler, n_features)
Y_val_inv = inverse_generation_only(Y_val, scaler, n_features)
Y_test_inv = inverse_generation_only(Y_test, scaler, n_features)


# =========================
# 12. Datas correspondentes
# =========================
train_dates = train.index[WINDOW_SIZE:]
val_dates = val.index[WINDOW_SIZE:]
test_dates = test.index[WINDOW_SIZE:]


# =========================
# 13. DataFrames de resultados
# =========================
train_results = pd.DataFrame({
    'Date': train_dates,
    'Train Predictions': train_predictions_inv,
    'Actuals': Y_train_inv
}).set_index('Date')

val_results = pd.DataFrame({
    'Date': val_dates,
    'Val Predictions': val_predictions_inv,
    'Actuals': Y_val_inv
}).set_index('Date')

test_results = pd.DataFrame({
    'Date': test_dates,
    'Test Predictions': test_predictions_inv,
    'Actuals': Y_test_inv
}).set_index('Date')


# =========================
# 14. Métricas
# =========================
def print_metrics(y_true, y_pred, nome):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mask = y_true > 1
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print(f"\n{nome}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

print_metrics(Y_train_inv, train_predictions_inv, "Treino")
print_metrics(Y_val_inv, val_predictions_inv, "Validação")
print_metrics(Y_test_inv, test_predictions_inv, "Teste")


# =========================
# 15. Baseline ingênuo
# =========================
baseline_test = test['Generation(kWh)'].shift(1).iloc[WINDOW_SIZE:]
baseline_real = test['Generation(kWh)'].iloc[WINDOW_SIZE:]

baseline_mae = mean_absolute_error(baseline_real, baseline_test)
baseline_rmse = np.sqrt(mean_squared_error(baseline_real, baseline_test))

print("\nBaseline ingênuo (previsão = valor do dia anterior)")
print(f"MAE : {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")


# =========================
# 16. Gráficos
# =========================
plt.figure(figsize=(12, 6))
plt.plot(train_results.index, train_results['Actuals'], label='Valores Reais')
plt.plot(train_results.index, train_results['Train Predictions'], label='Previsões do Treino', linestyle='--')
plt.title('Treino: Previsões vs Valores Reais')
plt.xlabel('Data')
plt.ylabel('Generation(kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(val_results.index, val_results['Actuals'], label='Valores Reais')
plt.plot(val_results.index, val_results['Val Predictions'], label='Previsões da Validação', linestyle='--')
plt.title('Validação: Previsões vs Valores Reais')
plt.xlabel('Data')
plt.ylabel('Generation(kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(test_results.index, test_results['Actuals'], label='Valores Reais')
plt.plot(test_results.index, test_results['Test Predictions'], label='Previsões do Teste', linestyle='--')
plt.title('Teste: Previsões vs Valores Reais')
plt.xlabel('Data')
plt.ylabel('Generation(kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()