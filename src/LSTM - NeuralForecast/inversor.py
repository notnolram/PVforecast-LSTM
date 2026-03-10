import pandas as pd
import numpy as np

# Carregar os dados
dataset = pd.read_csv('../../data/Inversor/Out22_15min_SG5K-D_001_001.xlsx.csv', skiprows=1)

# Selecionar apenas as colunas que queremos manter
columns_to_keep = ['Horário', 'SG5K-D_001_001/Potência CC total(kW)']

# Filtrar as colunas que existem no dataset
dataset = dataset[[col for col in columns_to_keep if col in dataset.columns]]

# Criar coluna de data e hora
dataset['Timestamp'] = pd.to_datetime(dataset['Horário'], errors='coerce')  # Handle parsing issues

# Definir o Timestamp como índice
dataset.set_index('Timestamp', inplace=True)
dataset.drop(columns=['Horário'], inplace=True)

# Converter a coluna para numérico
dataset['SG5K-D_001_001/Potência CC total(kW)'] = pd.to_numeric(dataset['SG5K-D_001_001/Potência CC total(kW)'].str.replace(',', '.', regex=False), errors='coerce')

# Filtrar apenas entre 6:00 e 19:00
dataset = dataset.between_time('06:00', '19:00')

# Filtrar as datas para outubro de 2022
dataset = dataset[(dataset.index >= '2022-10-01') & (dataset.index < '2022-11-01')]

# Selecionar os horários onde os minutos são 00 ou 30
dataset = dataset[dataset.index.minute.isin([0, 30])].copy()

# Substituir zeros por NaN para aplicar a lógica de preenchimento
dataset.replace(0, np.nan, inplace=True)

# Extrair o tempo do índice Timestamp (apenas HH:MM:SS)
remaining_data = dataset.copy()
remaining_data['time'] = remaining_data.index.time

# Função para aplicar bfill e preencher valores NaN com o anterior valor válido no mesmo horário
def fill_na_by_time(group):
    return group.bfill()

# Aplicar a função nos grupos de horários
remaining_data['SG5K-D_001_001/Potência CC total(kW)'] = remaining_data.groupby('time')['SG5K-D_001_001/Potência CC total(kW)'].transform(fill_na_by_time)

# Remover a coluna auxiliar de tempo
remaining_data.drop(columns=['time'], inplace=True)

# Preencher valores nulos com 0
remaining_data['SG5K-D_001_001/Potência CC total(kW)'] = remaining_data['SG5K-D_001_001/Potência CC total(kW)'].fillna(0)

# Renomear a coluna
remaining_data.rename(columns={'SG5K-D_001_001/Potência CC total(kW)': 'kW'}, inplace=True)

# Resumir as primeiras 5 linhas
print(remaining_data.head())

# Salvar em arquivo
remaining_data.to_csv('inversor.csv')
