import warnings
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Define o backend do Keras/TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

# Reduz mensagens de log do TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ignora warnings para deixar a saída mais limpa
warnings.filterwarnings("ignore")

import tensorflow as tf

# Limpa qualquer sessão anterior do Keras/TensorFlow
# Isso evita que modelos antigos fiquem na memória em execuções repetidas
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


# =========================================================
# 1. CONFIGURAÇÕES
# =========================================================

# Caminho do arquivo com dados da usina
ARQUIVO_USINA = Path("data/Usina/Dados_coletados_csv/dados_consolidados_2024_2025.csv")

# Caminho do arquivo climático do NSRDB
ARQUIVO_NSRDB = Path("data/NSRDB/9317325_-19.40_-40.05_2024.csv")

# Coluna alvo que o modelo vai tentar prever
TARGET_COL = "Generation(kWh)"

# Tamanho da janela temporal
# Ex.: se WINDOW = 3, o modelo usa 3 dias anteriores para prever o próximo
WINDOW = 2

# Define se as variáveis de nuvem serão usadas ou não
# True  -> usa cloud_clear_mean, cloud_medium_mean, etc.
# False -> ignora essas variáveis
USE_CLOUD_FEATURES = False

# Lista das variáveis climáticas esperadas no arquivo NSRDB
CLIMATE_FEATURES = [
    "GHI",
    "DNI",
    "DHI",
    "Temperature",
    "Relative Humidity",
    "Wind Speed",
    "Cloud Type",
]

# Define como cada variável climática será agregada diariamente
# Exemplo:
# - GHI, DNI e DHI serão somados no dia
# - temperatura terá média e máximo
# - umidade e vento terão média
DAILY_AGG = {
    "GHI": "sum",
    "DNI": "sum",
    "DHI": "sum",
    "Temperature": ["mean", "max"],
    "Relative Humidity": "mean",
    "Wind Speed": "mean",
}


# =========================================================
# 2. FUNÇÕES AUXILIARES
# =========================================================

def load_usina_csv(path: Path) -> pd.DataFrame:
    """
    Carrega o arquivo da usina.

    Etapas:
    1. Lê o CSV
    2. Converte a coluna Date para datetime
    3. Ordena por data e coloca Date como índice
    4. Trata zeros da geração como ausentes
    5. Interpola valores faltantes
    6. Normaliza o índice para ficar só com a data (sem horário)
    """
    df = pd.read_csv(path)

    # Converte coluna de data para datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Ordena e define como índice temporal
    df = df.sort_values("Date").set_index("Date")

    # Considera geração zero como dado faltante
    # Isso foi escolhido porque, no seu caso, zeros podem indicar falha/ausência
    df.loc[df[TARGET_COL] == 0, TARGET_COL] = np.nan

    # Interpola valores ausentes e, se ainda faltar algo no início/fim,
    # preenche com bfill/ffill
    df[TARGET_COL] = df[TARGET_COL].interpolate().bfill().ffill()

    # Remove parte de hora/minuto/segundo do índice, deixando só a data
    df.index = df.index.normalize()

    return df


def load_nsrdb_csv(path: Path) -> pd.DataFrame:
    """
    Carrega o arquivo do NSRDB.

    Alguns arquivos já vêm com cabeçalho padrão.
    Outros possuem metadados nas primeiras linhas.
    Então:
    - tenta ler normalmente
    - se não encontrar as colunas esperadas, tenta de novo pulando 2 linhas
    """
    try:
        df = pd.read_csv(path)

        # Conjunto mínimo de colunas de tempo esperadas
        expected = {"Year", "Month", "Day", "Hour"}

        # Se as colunas existem, o arquivo foi lido corretamente
        if expected.issubset(set(df.columns)):
            return df
    except Exception:
        pass

    # Segunda tentativa: pula 2 linhas iniciais
    df = pd.read_csv(path, skiprows=2)
    return df


def prepare_nsrdb_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara os dados do NSRDB em resolução diária.

    Etapas:
    1. Renomeia colunas, se necessário
    2. Confere se as colunas de tempo existem
    3. Cria timestamp
    4. Define timestamp como índice
    5. Cria variáveis de nuvem derivadas de Cloud Type
    6. Agrega as variáveis por dia
    7. Achata nomes de colunas agregadas
    """
    rename_map = {
        "Temperature": "Temperature",
        "Relative Humidity": "Relative Humidity",
        "Wind Speed": "Wind Speed",
        "GHI": "GHI",
        "DNI": "DNI",
        "DHI": "DHI",
        "Cloud Type": "Cloud Type",
    }

    # Faz uma cópia e renomeia
    df = df.rename(columns=rename_map).copy()

    # Garante que as colunas de tempo essenciais existem
    required_time_cols = ["Year", "Month", "Day", "Hour"]
    for col in required_time_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente no NSRDB: {col}")

    # Se não houver a coluna Minute, cria com zero
    if "Minute" not in df.columns:
        df["Minute"] = 0

    # Cria um timestamp completo com ano, mês, dia, hora e minuto
    df["timestamp"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour", "Minute"]],
        errors="coerce"
    )

    # Remove timestamps inválidos, usa timestamp como índice e ordena
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    # Verifica quais features climáticas realmente existem no arquivo
    existing_features = [c for c in CLIMATE_FEATURES if c in df.columns]

    # Se poucas features foram encontradas, aborta
    if len(existing_features) < 3:
        raise ValueError(
            f"Poucas features climáticas encontradas. Encontradas: {existing_features}"
        )

    # Se Cloud Type existir, cria colunas categóricas simplificadas
    if "Cloud Type" in df.columns:
        # Converte para número
        df["Cloud Type"] = pd.to_numeric(df["Cloud Type"], errors="coerce")

        # Céu limpo
        df["cloud_clear"] = (df["Cloud Type"] == 0).astype(int)

        # Nuvens/fenômenos leves a moderados
        df["cloud_medium"] = df["Cloud Type"].isin([1, 2, 3, 4, 5, 6]).astype(int)

        # Nuvens mais densas
        df["cloud_dense"] = df["Cloud Type"].isin([7, 8, 9, 10]).astype(int)

        # Poeira/fumaça/extremos
        df["cloud_extreme"] = df["Cloud Type"].isin([11, 12]).astype(int)

    # Monta o dicionário de agregações apenas para colunas existentes
    agg_map = {}
    for col, agg in DAILY_AGG.items():
        if col in df.columns:
            agg_map[col] = agg

    # Agregações diárias das variáveis de nuvem, se existirem
    if "cloud_clear" in df.columns:
        agg_map["cloud_clear"] = "mean"
    if "cloud_medium" in df.columns:
        agg_map["cloud_medium"] = "mean"
    if "cloud_dense" in df.columns:
        agg_map["cloud_dense"] = "mean"
    if "cloud_extreme" in df.columns:
        agg_map["cloud_extreme"] = "mean"

    # Reamostra para frequência diária usando as agregações definidas
    clima_daily = df.resample("D").agg(agg_map)

    # Após agregação com múltiplas operações, algumas colunas viram MultiIndex
    # Este bloco transforma nomes como ('Temperature', 'mean') em 'Temperature_mean'
    flat_cols = []
    for col in clima_daily.columns:
        if isinstance(col, tuple):
            if col[1] == "":
                flat_cols.append(col[0])
            else:
                flat_cols.append(f"{col[0]}_{col[1]}")
        else:
            flat_cols.append(col)

    clima_daily.columns = flat_cols

    # Normaliza o índice diário
    clima_daily.index = clima_daily.index.normalize()

    return clima_daily


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis sazonais cíclicas com base no dia do ano.

    Usamos seno e cosseno para representar a sazonalidade anual
    sem criar uma quebra artificial entre o fim e o início do ano.
    """
    df = df.copy()

    # Extrai o dia do ano (1 a 366 no caso de ano bissexto)
    day_of_year = df.index.dayofyear

    # Como 2024 é bissexto, usamos 366 no denominador
    df["sin_day"] = np.sin(2 * np.pi * day_of_year / 366)
    df["cos_day"] = np.cos(2 * np.pi * day_of_year / 366)

    return df


def create_multivariate_dataset(data: np.ndarray, window: int):
    """
    Converte uma série multivariada em amostras para LSTM.

    Exemplo:
    Se window = 3, cada X terá 3 linhas consecutivas do passado.
    O y será o valor da próxima linha, na coluna 0 (target).

    Retorna:
    - X com shape (amostras, window, n_features)
    - y com shape (amostras,)
    """
    X, y = [], []

    for i in range(len(data) - window):
        # Janela de entrada
        X.append(data[i:i + window])

        # Alvo: próximo valor da coluna 0
        y.append(data[i + window, 0])

    return np.array(X), np.array(y)


def inverse_transform_target(y_scaled, scaler, n_features):
    """
    Desfaz a normalização apenas da variável alvo.

    Como o scaler foi treinado com todas as features,
    é necessário montar uma matriz temporária com o mesmo número de colunas.
    A primeira coluna recebe o target escalado e as demais ficam zeradas.
    """
    temp = np.zeros((len(y_scaled), n_features))
    temp[:, 0] = y_scaled.flatten()
    return scaler.inverse_transform(temp)[:, 0]


def print_metrics(nome, y_true, y_pred):
    """
    Calcula e imprime métricas de erro:
    - MAE
    - RMSE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{nome} -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")


# =========================================================
# 3. CARREGAR DADOS
# =========================================================

# Carrega dados da usina
usina = load_usina_csv(ARQUIVO_USINA)

# Carrega dados brutos do NSRDB
nsrdb_raw = load_nsrdb_csv(ARQUIVO_NSRDB)

# Transforma o NSRDB em base diária agregada
clima_daily = prepare_nsrdb_daily(nsrdb_raw)

# Mostra período disponível em cada base
print("Período usina :", usina.index.min(), "até", usina.index.max())
print("Período NSRDB :", clima_daily.index.min(), "até", clima_daily.index.max())


# =========================================================
# 4. ALINHAR DATAS
# =========================================================

# Encontra datas que existem em ambas as bases
datas_comuns = usina.index.intersection(clima_daily.index)

# Se não houver datas em comum, não é possível continuar
if len(datas_comuns) == 0:
    raise ValueError("Não há datas em comum entre usina e NSRDB.")

# Filtra ambas as bases para manter apenas as datas em comum
usina_filtrada = usina.loc[datas_comuns].copy()
clima_filtrado = clima_daily.loc[datas_comuns].copy()

# Junta as duas bases em um único dataset
dataset = usina_filtrada.join(clima_filtrado, how="inner")

# Adiciona variáveis sazonais
dataset = add_seasonality_features(dataset)

# Salva o dataset resultante para inspeção externa
dataset.to_csv("data/Usina/dataset_nsrdb_filtrado.csv")

# Mostra algumas informações do dataset unido
print("Datas em comum:", len(datas_comuns))
print("Colunas finais:", dataset.columns.tolist())


# =========================================================
# 5. SELEÇÃO DE FEATURES
# =========================================================

# Features básicas do experimento
base_features = [
    TARGET_COL,
    "GHI_sum",
    "DNI_sum",
    "DHI_sum",
    "Temperature_mean",
    "Temperature_max",
    "Relative Humidity_mean",
    "Wind Speed_mean",
    "sin_day",
    "cos_day",
]

# Features derivadas de nuvem
cloud_features = [
    "cloud_clear_mean",
    "cloud_medium_mean",
    "cloud_dense_mean",
    "cloud_extreme_mean",
]

# Começa com as features básicas
candidate_features = base_features.copy()

# Se configurado, adiciona as features de nuvem
if USE_CLOUD_FEATURES:
    candidate_features += cloud_features

# Mantém apenas as colunas que realmente existem
features = [c for c in candidate_features if c in dataset.columns]

# Remove linhas com NaN e cria o dataset final do modelo
dataset_final = dataset[features].dropna().copy()

# Segurança mínima: evita treinar com muito poucos dados
if len(dataset_final) < 30:
    raise ValueError(
        "Poucas amostras após cruzamento e limpeza. Verifique o período do NSRDB."
    )

# Mostra como o experimento foi configurado
print("\nConfiguração do experimento:")
print("USE_CLOUD_FEATURES =", USE_CLOUD_FEATURES)

print("\nFeatures usadas:")
print(features)

print("\nDataset final:")
print(dataset_final.head())
print("Shape:", dataset_final.shape)


# =========================================================
# 6. SPLIT TEMPORAL ANTES DA NORMALIZAÇÃO
# =========================================================

# Número total de amostras
n = len(dataset_final)

# Define os pontos de corte do split temporal
# 70% treino, 15% validação, 15% teste
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Divide em blocos temporais, sem embaralhar
train_df = dataset_final.iloc[:train_end].copy()
val_df = dataset_final.iloc[train_end:val_end].copy()
test_df = dataset_final.iloc[val_end:].copy()

# Cria scaler MinMax
scaler = MinMaxScaler()

# Ajusta o scaler apenas com o treino
# Isso evita vazamento de informação
train_scaled = pd.DataFrame(
    scaler.fit_transform(train_df),
    index=train_df.index,
    columns=train_df.columns
)

# Aplica o scaler já ajustado ao conjunto de validação
val_scaled = pd.DataFrame(
    scaler.transform(val_df),
    index=val_df.index,
    columns=val_df.columns
)

# Aplica o scaler já ajustado ao conjunto de teste
test_scaled = pd.DataFrame(
    scaler.transform(test_df),
    index=test_df.index,
    columns=test_df.columns
)


# =========================================================
# 7. CRIAR JANELAS
# =========================================================

# Converte os dataframes escalados em sequências para a LSTM
X_train, y_train = create_multivariate_dataset(train_scaled.values, WINDOW)
X_val, y_val = create_multivariate_dataset(val_scaled.values, WINDOW)
X_test, y_test = create_multivariate_dataset(test_scaled.values, WINDOW)

# Verifica se sobrou amostra suficiente depois da janela
if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise ValueError("A divisão ficou pequena demais para criar janelas.")

# Guarda os índices correspondentes aos alvos após a criação das janelas
# Ex.: se WINDOW=3, os 3 primeiros dias viram entrada e o índice começa no 4º
index_train = train_scaled.index[WINDOW:]
index_val = val_scaled.index[WINDOW:]
index_test = test_scaled.index[WINDOW:]

# Mostra o formato final das entradas/saídas
print("\nShapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)


# =========================================================
# 8. MODELO
# =========================================================

# Cria a rede LSTM empilhada
model = Sequential([
    # Define o shape de entrada: (window, número de features)
    InputLayer(shape=(X_train.shape[1], X_train.shape[2])),

    # Primeira LSTM retorna sequência para alimentar a próxima LSTM
    LSTM(32, return_sequences=True),

    # Dropout para reduzir overfitting
    Dropout(0.1),

    # Segunda LSTM recebe a sequência e retorna um vetor final
    LSTM(16),

    # Novo dropout
    Dropout(0.1),

    # Camada final para prever 1 valor contínuo
    Dense(1)
])

# Compila o modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=MeanSquaredError(),
    metrics=[RootMeanSquaredError()]
)

# Mostra resumo da arquitetura
model.summary()


# =========================================================
# 9. CALLBACKS
# =========================================================

# Garante que a pasta do modelo exista
os.makedirs("model1", exist_ok=True)

# Salva automaticamente o melhor modelo com base no menor val_loss
cp = ModelCheckpoint(
    "model1/best_model_nsrdb.keras",
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1
)

# Para o treinamento se o val_loss parar de melhorar
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    mode="min",
    verbose=1
)


# =========================================================
# 10. TREINAMENTO
# =========================================================

# Treina o modelo com os dados de treino
# e avalia a cada época no conjunto de validação
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[cp, early_stop],
    verbose=1,
    shuffle=False  # importante em série temporal
)


# =========================================================
# 11. PREVISÕES
# =========================================================

# Gera previsões em escala normalizada
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Número total de features usadas no scaler
n_features = len(features)

# Desfaz a normalização dos valores reais
y_train_real = inverse_transform_target(y_train.reshape(-1, 1), scaler, n_features)
y_val_real = inverse_transform_target(y_val.reshape(-1, 1), scaler, n_features)
y_test_real = inverse_transform_target(y_test.reshape(-1, 1), scaler, n_features)

# Desfaz a normalização das previsões
y_train_pred_inv = inverse_transform_target(y_train_pred, scaler, n_features)
y_val_pred_inv = inverse_transform_target(y_val_pred, scaler, n_features)
y_test_pred_inv = inverse_transform_target(y_test_pred, scaler, n_features)


# =========================================================
# 12. MÉTRICAS
# =========================================================

# Imprime métricas de desempenho
print("\nMétricas:")
print_metrics("Treino", y_train_real, y_train_pred_inv)
print_metrics("Validação", y_val_real, y_val_pred_inv)
print_metrics("Teste", y_test_real, y_test_pred_inv)


# =========================================================
# 13. PLOTS
# =========================================================

# Gráfico da loss ao longo das épocas
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Treino")
plt.plot(history.history["val_loss"], label="Validação")
plt.title("Loss durante o treinamento")
plt.xlabel("Épocas")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico treino: real vs previsto
plt.figure(figsize=(12, 5))
plt.plot(index_train, y_train_real, label="Real")
plt.plot(index_train, y_train_pred_inv, label="Previsto")
plt.title("Treino - Real vs Previsto")
plt.xlabel("Data")
plt.ylabel(TARGET_COL)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico validação: real vs previsto
plt.figure(figsize=(12, 5))
plt.plot(index_val, y_val_real, label="Real")
plt.plot(index_val, y_val_pred_inv, label="Previsto")
plt.title("Validação - Real vs Previsto")
plt.xlabel("Data")
plt.ylabel(TARGET_COL)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico teste: real vs previsto
plt.figure(figsize=(12, 5))
plt.plot(index_test, y_test_real, label="Real")
plt.plot(index_test, y_test_pred_inv, label="Previsto")
plt.title("Teste - Real vs Previsto")
plt.xlabel("Data")
plt.ylabel(TARGET_COL)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico geral comparando treino, validação e teste
plt.figure(figsize=(14, 6))
plt.plot(index_train, y_train_real, label="Treino Real")
plt.plot(index_train, y_train_pred_inv, label="Treino Previsto")
plt.plot(index_val, y_val_real, label="Validação Real")
plt.plot(index_val, y_val_pred_inv, label="Validação Previsto")
plt.plot(index_test, y_test_real, label="Teste Real")
plt.plot(index_test, y_test_pred_inv, label="Teste Previsto")
plt.title("Comparação geral - Treino, Validação e Teste")
plt.xlabel("Data")
plt.ylabel(TARGET_COL)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()