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
# 1. Carregar dados
# =========================

# USINA
arquivo_usina = Path("data/Usina/Dados_coletados_csv/dados_consolidados_2024_2025.csv")
usina = pd.read_csv(arquivo_usina)

usina['Date'] = pd.to_datetime(usina['Date'])
usina = usina.set_index('Date').sort_index()

# CLIMA (Solcast)
arquivo_clima = Path("data/Solcast/Ifes_Solcast_PT15M_2023.csv")
clima = pd.read_csv(arquivo_clima)

clima['PeriodStart'] = pd.to_datetime(clima['PeriodStart'], utc=True)
clima['PeriodStart'] = clima['PeriodStart'].dt.tz_convert(None)
clima = clima.set_index('PeriodStart').sort_index()


# =========================
# 2. Transformar clima em diário
# =========================

clima_daily = clima.resample('D').agg({
    'Ghi': 'sum',
    'Dni': 'sum',
    'Dhi': 'sum',
    'CloudOpacity': 'mean',
    'AirTemp': ['mean', 'max'],
    'RelativeHumidity': 'mean'
})

clima_daily.columns = [
    'ghi_sum',
    'dni_sum',
    'dhi_sum',
    'cloud_mean',
    'temp_mean',
    'temp_max',
    'humidity_mean'
]

clima_daily.index = clima_daily.index.normalize()


# =========================
# 3. Alinhar datas
# =========================

usina.index = usina.index.normalize()
datas_comuns = usina.index.intersection(clima_daily.index)

usina_filtrada = usina.loc[datas_comuns].copy()
clima_filtrado = clima_daily.loc[datas_comuns].copy()

dataset = usina_filtrada.join(clima_filtrado, how='inner')

saida = Path("data/Usina/dataset_solcast_filtrado.csv")
dataset.to_csv(saida)

print("Período usina:", usina.index.min(), "até", usina.index.max())
print("Período clima:", clima_daily.index.min(), "até", clima_daily.index.max())
print("Datas em comum:", len(datas_comuns))
print("Colunas do dataset:", dataset.columns.tolist())


# =========================
# 4. Tratamento da geração
# =========================

dataset.loc[dataset['Generation(kWh)'] == 0, 'Generation(kWh)'] = np.nan
dataset['Generation(kWh)'] = dataset['Generation(kWh)'].interpolate()

# opcional: remover nans restantes nas bordas
dataset['Generation(kWh)'] = dataset['Generation(kWh)'].bfill().ffill()


# =========================
# 5. Seleção de features
# =========================

features = [
    'Generation(kWh)',
    'ghi_sum',
    'cloud_mean',
    'dni_sum',
    'dhi_sum'
]

dataset_final = dataset[features].dropna().copy()

print("\nDataset final:")
print(dataset_final.head())
print("Shape:", dataset_final.shape)

if len(dataset_final) < 30:
    raise ValueError(
        "Poucas amostras no dataset_final. Verifique a sobreposição temporal entre usina e clima."
    )


# =========================
# 6. Escalonamento
# =========================

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset_final)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=features,
    index=dataset_final.index
)


# =========================
# 7. Criar janelas multivariadas
# =========================

def create_multivariate_dataset(data, window):
    X, y = [], []

    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window, 0])  # target = Generation(kWh)

    return np.array(X), np.array(y)


WINDOW = 5
X, y = create_multivariate_dataset(scaled_df.values, WINDOW)

print("\nShapes após criação das janelas:")
print("X shape:", X.shape)
print("y shape:", y.shape)

if len(X) < 10:
    raise ValueError(
        "Quantidade de amostras após criação das janelas é muito pequena."
    )


# índices correspondentes aos alvos y
target_index = dataset_final.index[WINDOW:]


# =========================
# 8. Split temporal: treino / validação / teste
# =========================

n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

index_train = target_index[:train_end]
index_val = target_index[train_end:val_end]
index_test = target_index[val_end:]

print("\nDivisão temporal:")
print("Treino:", X_train.shape, y_train.shape)
print("Validação:", X_val.shape, y_val.shape)
print("Teste:", X_test.shape, y_test.shape)


# =========================
# 9. Modelo LSTM
# =========================

model = Sequential([
    InputLayer(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=MeanSquaredError(),
    metrics=[RootMeanSquaredError()]
)

model.summary()


# =========================
# 10. Callbacks
# =========================

cp = ModelCheckpoint(
    "best_model.keras",
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    mode="min",
    verbose=1
)


# =========================
# 11. Treinamento
# =========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[cp, early_stop],
    verbose=1,
    shuffle=False
)


# =========================
# 12. Previsões
# =========================

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)


# =========================
# 13. Inversão da escala
# =========================

def inverse_transform_target(y_scaled, scaler, n_features):
    temp = np.zeros((len(y_scaled), n_features))
    temp[:, 0] = y_scaled.flatten()
    return scaler.inverse_transform(temp)[:, 0]

n_features = len(features)

y_train_real = inverse_transform_target(y_train.reshape(-1, 1), scaler, n_features)
y_val_real = inverse_transform_target(y_val.reshape(-1, 1), scaler, n_features)
y_test_real = inverse_transform_target(y_test.reshape(-1, 1), scaler, n_features)

y_train_pred_inv = inverse_transform_target(y_train_pred, scaler, n_features)
y_val_pred_inv = inverse_transform_target(y_val_pred, scaler, n_features)
y_test_pred_inv = inverse_transform_target(y_test_pred, scaler, n_features)


# =========================
# 14. Métricas
# =========================

def print_metrics(nome, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{nome} -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")

print("\nMétricas:")
print_metrics("Treino", y_train_real, y_train_pred_inv)
print_metrics("Validação", y_val_real, y_val_pred_inv)
print_metrics("Teste", y_test_real, y_test_pred_inv)


# =========================
# 15. Plot da curva de treinamento
# =========================

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Loss durante o treinamento')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================
# 16. Plot treino
# =========================

plt.figure(figsize=(12, 5))
plt.plot(index_train, y_train_real, label='Real')
plt.plot(index_train, y_train_pred_inv, label='Previsto')
plt.title('Treino - Real vs Previsto')
plt.xlabel('Data')
plt.ylabel('Generation (kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================
# 17. Plot validação
# =========================

plt.figure(figsize=(12, 5))
plt.plot(index_val, y_val_real, label='Real')
plt.plot(index_val, y_val_pred_inv, label='Previsto')
plt.title('Validação - Real vs Previsto')
plt.xlabel('Data')
plt.ylabel('Generation (kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================
# 18. Plot teste
# =========================

plt.figure(figsize=(12, 5))
plt.plot(index_test, y_test_real, label='Real')
plt.plot(index_test, y_test_pred_inv, label='Previsto')
plt.title('Teste - Real vs Previsto')
plt.xlabel('Data')
plt.ylabel('Generation (kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================
# 19. Plot geral
# =========================

plt.figure(figsize=(14, 6))
plt.plot(index_train, y_train_real, label='Treino Real')
plt.plot(index_train, y_train_pred_inv, label='Treino Previsto')

plt.plot(index_val, y_val_real, label='Validação Real')
plt.plot(index_val, y_val_pred_inv, label='Validação Previsto')

plt.plot(index_test, y_test_real, label='Teste Real')
plt.plot(index_test, y_test_pred_inv, label='Teste Previsto')

plt.title('Comparação geral - Treino, Validação e Teste')
plt.xlabel('Data')
plt.ylabel('Generation (kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()