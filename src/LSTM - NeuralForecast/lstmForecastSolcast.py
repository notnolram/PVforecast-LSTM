import pandas as pd
from calculatePOA import calculate_POA_for_year
from PVGeneration import PVGeneration

# Carregar os dados
dataset = pd.read_csv('data/Solcast/-19.407482_-40.045201_Solcast_PT15M.csv')
dataset2 = pd.read_csv('data/Solcast/Ifes_Solcast_PT15M_2023.csv')

# Selecionar apenas as colunas que queremos manter
columns_to_keep = ['PeriodStart', 'AirTemp', 'Azimuth', 'CloudOpacity', 'Ghi', 'PrecipitableWater', 
                   'RelativeHumidity', 'WindDirection10m', 'WindSpeed10m', 'Zenith', 'Dni', 'Dhi', 'DewpointTemp']

# Filtrar as colunas que existem em ambos os datasets
dataset = dataset[[col for col in columns_to_keep if col in dataset.columns]]
dataset2 = dataset2[[col for col in columns_to_keep if col in dataset2.columns]]

# Concatenar os dois datasets
dataset_concat = pd.concat([dataset, dataset2], ignore_index=True)

# Criar coluna de data e hora
dataset_concat['Timestamp'] = pd.to_datetime(dataset_concat['PeriodStart'], errors='coerce')  # Tratar erros de parsing
dataset_concat['Timestamp'] = dataset_concat['Timestamp'].dt.tz_localize(None)  # Remover timezone (GMT)

# Definir o Timestamp como índice
dataset_concat.set_index('Timestamp', inplace=True)

# Filtrar as datas para outubro de 2022
dataset_concat = dataset_concat[(dataset_concat.index >= '2022-10-01') & (dataset_concat.index < '2022-11-01')]

# Filtrar apenas entre 8:30 e 17:00
dataset_concat = dataset_concat.between_time('06:00', '19:00')

# Selecionar apenas as colunas restantes que são climáticas
dataset_concat = dataset_concat[['Dni','Dhi', 'Ghi', 'AirTemp', 'DewpointTemp', 'WindSpeed10m', 'WindDirection10m', 'RelativeHumidity']]

# Selecionar os horários onde os minutos são 00 ou 30
dataset_pares = dataset_concat[dataset_concat.index.minute.isin([0, 30])].copy()  # Usar .copy() para garantir que não seja uma cópia de fatia

# Parâmetros físicos do modelo do módulo PV
Npv = 12            # number of modules
Efficiency = 0.204   # module efficiency
tilt = 10            # Tilt angle
azimuth = 12          # Azimuth angle
latitude = -19.39

# Module data
Vmpp = 41.1
Impp = 10.96
Voc = 49.1
Isc = 11.60
Kv = -0.27
Ki = 0.05

#calcular POA e PVGeneration
dataset_pares = calculate_POA_for_year(dataset_pares, tilt, azimuth, latitude)
dataset_calculado = PVGeneration(dataset_pares, Npv, Vmpp, Impp, Voc, Isc, Kv, Ki)

# Resumir as primeiras 5 linhas
print(dataset_calculado.head())

# Salvar em arquivo
dataset_calculado.to_csv('lstm.csv')