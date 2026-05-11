import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
from data_loader import (
    load_nsrdb_csv,
    prepare_nsrdb_daily,
    load_usina_csv,
    add_seasonality_features,
    create_sequences,
    inverse_transform_target,
    print_metrics
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

path_nsrdb = Path("data/NSRDB/9317325_-19.40_-40.05_2024.csv")
path_usina = Path("data/Usina/Dados_coletados_csv/dados_consolidados_2024_2025.csv")

# ================================
# CONFIGURAÇÃO DO EXPERIMENTO
# ================================

target_col = "Generation(kWh)"
window = 3

lstm_units = 64
learning_rate = 0.0005
epochs = 200
batch_size = 16

# Quais dados vão pra teste
input_features = [
    "GHI_sum",
    "Temperature_mean",
    "Relative Humidity_mean",
    "Cloud Type_mean",
    "sin_day",
    "cos_day"
]

# Monta a lista final garantindo que o target fique na primeira posição
features = list(dict.fromkeys([target_col] + input_features))

dt = load_nsrdb_csv(path_nsrdb)
dt_daily = prepare_nsrdb_daily(dt)

usina = load_usina_csv(
    path_usina,
    target_col,
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Encontra apenas as datas que existem nas duas bases
datas_comuns = usina.index.intersection(dt_daily.index)

# Verifica se há interseção
if len(datas_comuns) == 0:
    raise ValueError("Não há datas em comum entre usina e NSRDB.")

# Filtra as duas bases para o mesmo conjunto de datas
usina_filtrada = usina.loc[datas_comuns].copy()
clima_filtrado = dt_daily.loc[datas_comuns].copy()

# Junta geração e clima em um único dataset
dataset = usina_filtrada.join(clima_filtrado, how="inner")
dataset = add_seasonality_features(dataset)

# Salva o dataset final já alinhado
dataset.to_csv("data/Usina/dataset_nsrdb_2024.csv")

# ================================
# SELEÇÃO DE FEATURES
# ================================

# Mantém apenas as colunas que realmente existem
features = [c for c in features if c in dataset.columns]

dataset_final = dataset[features].copy()

if dataset_final.isna().values.any():
    print("⚠️ Existem valores NaN no dataset!")

# Separa entradas (X) e alvo (y)
X_df = dataset_final[input_features].copy()
y_df = dataset_final[[target_col]].copy()

print("Features usadas:", features)
print("Window:", window)

# ================================
# SPLIT TEMPORAL
# ================================

n = len(X_df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train_df = X_df.iloc[:train_end].copy()
X_val_df = X_df.iloc[train_end:val_end].copy()
X_test_df = X_df.iloc[val_end:].copy()

y_train_df = y_df.iloc[:train_end].copy()
y_val_df = y_df.iloc[train_end:val_end].copy()
y_test_df = y_df.iloc[val_end:].copy()

# ================================
# NORMALIZAÇÃO
# ================================

X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train_scaled = pd.DataFrame(
    X_scaler.fit_transform(X_train_df),
    index=X_train_df.index,
    columns=X_train_df.columns
)

X_val_scaled = pd.DataFrame(
    X_scaler.transform(X_val_df),
    index=X_val_df.index,
    columns=X_val_df.columns
)

X_test_scaled = pd.DataFrame(
    X_scaler.transform(X_test_df),
    index=X_test_df.index,
    columns=X_test_df.columns
)

y_train_scaled = pd.DataFrame(
    y_scaler.fit_transform(y_train_df),
    index=y_train_df.index,
    columns=y_train_df.columns
)

y_val_scaled = pd.DataFrame(
    y_scaler.transform(y_val_df),
    index=y_val_df.index,
    columns=y_val_df.columns
)

y_test_scaled = pd.DataFrame(
    y_scaler.transform(y_test_df),
    index=y_test_df.index,
    columns=y_test_df.columns
)


# ================================
# CRIAÇÃO DAS JANELAS
# ================================

X_train, y_train = create_sequences(X_train_scaled.values, y_train_scaled.values, window)
X_val, y_val = create_sequences(X_val_scaled.values, y_val_scaled.values, window)
X_test, y_test = create_sequences(X_test_scaled.values, y_test_scaled.values, window)

# ================================
# MODELO LSTM
# ================================

model = Sequential([
    InputLayer(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64),
    Dense(1, activation='linear')
])


cp = ModelCheckpoint('model1.keras', save_best_only=True)


model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=MeanSquaredError(),
    #loss=Huber(delta=0.5),
    metrics=[
        RootMeanSquaredError(name='rmse'),
        MeanAbsoluteError(name='mae'),
        MeanAbsolutePercentageError(name='mape')
    ]
)

model.summary()

# ================================
# TREINAMENTO
# ================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    mode="min",
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[cp, early_stop],
    verbose=1
)

# ================================
# PLOTS DE TREINAMENTO
# ================================

plt.figure(figsize=(12, 5))

# LOSS
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Loss (MSE) durante o treinamento')
#plt.title('Loss (Huber) durante o treinamento')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# RMSE
plt.subplot(1, 2, 2)
plt.plot(history.history['rmse'], label='Treino')
plt.plot(history.history['val_rmse'], label='Validação')
plt.title('RMSE durante o treinamento')
plt.xlabel('Épocas')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ================================
# PREVISÕES
# ================================

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# ================================
# INVERSÃO DA ESCALA
# ================================

y_train_real = inverse_transform_target(y_train, y_scaler)
y_val_real = inverse_transform_target(y_val, y_scaler)
y_test_real = inverse_transform_target(y_test, y_scaler)

y_train_pred_real = inverse_transform_target(y_train_pred, y_scaler)
y_val_pred_real = inverse_transform_target(y_val_pred, y_scaler)
y_test_pred_real = inverse_transform_target(y_test_pred, y_scaler)

print("\n===== CONFIGURAÇÃO DO MODELO =====")
print(f"Entradas (X): {input_features}")
print(f"Saída (y): {target_col}")
print(f"Timesteps (window): {window}")
 
print_metrics("Treino", y_train_real, y_train_pred_real)
print_metrics("Validação", y_val_real, y_val_pred_real)
print_metrics("Teste", y_test_real, y_test_pred_real)

def mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ================================
# AJUSTE DAS DATAS (por causa do window)
# ================================

train_dates = y_train_df.index[window:]
val_dates = y_val_df.index[window:]
test_dates = y_test_df.index[window:]

# ================================
# FUNÇÃO DE PLOT
# ================================

def plot_real_vs_pred(dates, y_true, y_pred, title):
    plt.figure(figsize=(14, 5))
    plt.plot(dates, y_true, label='Real', linewidth=2)
    plt.plot(dates, y_pred, label='Previsto', linewidth=2)
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Geração (kWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ================================
# PLOTS
# ================================

plot_real_vs_pred(train_dates, y_train_real, y_train_pred_real, 'Treino - Real vs Previsto')
plot_real_vs_pred(val_dates, y_val_real, y_val_pred_real, 'Validação - Real vs Previsto')
plot_real_vs_pred(test_dates, y_test_real, y_test_pred_real, 'Teste - Real vs Previsto')
