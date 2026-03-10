import pandas as pd

# Carregar os dados processados
inversor_data = pd.read_csv('inversor.csv', index_col='Timestamp', parse_dates=True)
lstm_data = pd.read_csv('lstm.csv', index_col='Timestamp', parse_dates=True)

# Garantir que ambos os datasets tenham o mesmo índice de tempo (intersecção de datas)
merged_data = pd.merge(inversor_data, lstm_data, left_index=True, right_index=True, how='inner')

# Calcular a correlação entre Potência do Inversor e GHI
correlation = merged_data['kW'].corr(merged_data['Ppv'])

# Exibir a correlação
print(f"Correlação entre a Potência do Inversor e o GHI: {correlation:.4f}")

# Exibir uma amostra dos dados combinados
print(merged_data.head())