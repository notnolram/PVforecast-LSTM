from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_nsrdb_csv(path: Path) -> pd.DataFrame:
    """
    Carrega o CSV do NSRDB.

    Observação:
    - O arquivo possui linhas iniciais de metadados,
      por isso usamos skiprows=2.
    """
    df = pd.read_csv(path, skiprows=2)
    return df


def prepare_nsrdb_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte dados horários do NSRDB em dados diários.

    Etapas:
    1. Criação de timestamp (data + hora)
    2. Definição do índice temporal
    3. Agregação diária das variáveis
    4. Renomeação das colunas
    """

    # Trabalha com uma cópia para evitar modificar o original
    df = df.copy()

    # Cria coluna de data/hora completa
    df["timestamp"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour", "Minute"]],
        errors="coerce"
    )

    # Remove registros inválidos
    df = df.dropna(subset=["timestamp"])

    # Define o timestamp como índice e ordena
    df = df.set_index("timestamp").sort_index()

    # Agregação diária:
    # - Irradiação: soma (energia acumulada no dia)
    # - Temperatura: média e máximo
    # - Umidade e vento: média
    # - Cloud Type: média (representação geral do dia)
    daily = df.resample("D").agg({
        "GHI": "sum",
        "DNI": "sum",
        "DHI": "sum",
        "Temperature": ["mean", "max"],
        "Relative Humidity": "mean",
        "Wind Speed": "mean",
        "Cloud Type": "mean",
    })

    # Após múltiplas agregações, o pandas cria colunas com MultiIndex
    # Aqui transformamos para nomes simples
    daily.columns = [
        "GHI_sum",
        "DNI_sum",
        "DHI_sum",
        "Temperature_mean",
        "Temperature_max",
        "Relative Humidity_mean",
        "Wind Speed_mean",
        "Cloud Type_mean",
    ]

    # Remove informação de horário (fica só a data)
    daily.index = daily.index.normalize()

    return daily

def load_usina_csv(
    path: Path,
    target_col: str,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Carrega o CSV da usina e prepara a série diária da variável alvo.

    Etapas:
    1. Lê o arquivo
    2. Valida se a coluna alvo existe
    3. Converte a coluna Date para datetime
    4. Ordena por data
    5. Define Date como índice
    6. Trata zeros como ausentes na coluna alvo
    7. Interpola valores faltantes
    8. Filtra o período, se informado
    9. Normaliza o índice para manter apenas a data
    """
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(f"Coluna {target_col} não encontrada no dataset da usina.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Trata valores baixos como ausentes
    low_threshold = 20
    df.loc[df[target_col] <= low_threshold, target_col] = pd.NA

    # Interpola os valores ausentes
    df[target_col] = df[target_col].interpolate().bfill().ffill()

    if start_date is not None or end_date is not None:
        df = df.loc[start_date:end_date]

    return df[[target_col]]

def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona variáveis sazonais cíclicas com base no dia do ano.

    Essas variáveis ajudam o modelo a representar a sazonalidade anual
    de forma contínua.
    """
    df = df.copy()

    # Extrai o dia do ano
    day_of_year = df.index.dayofyear

    # Como 2024 é bissexto, usamos 366
    df["sin_day"] = np.sin(2 * np.pi * day_of_year / 366)
    df["cos_day"] = np.cos(2 * np.pi * day_of_year / 366)

    return df

def create_sequences(X, y, window: int):
    """
    Cria janelas para LSTM usando X e y separados.

    X: array 2D com shape (amostras, n_features)
    y: array 2D com shape (amostras, 1)
    """
    X_seq, y_seq = [], []

    for i in range(len(X) - window):
        X_seq.append(X[i:i + window])
        y_seq.append(y[i + window, 0])

    return np.array(X_seq), np.array(y_seq)

def inverse_transform_target(y_scaled, y_scaler):
    return y_scaler.inverse_transform(
        y_scaled.reshape(-1, 1)
    ).flatten()

def print_metrics(nome, y_true, y_pred):
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (com proteção contra divisão por zero)
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    print(f"{nome} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")