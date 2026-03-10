import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados processados
inversor_data = pd.read_csv('inversor.csv', index_col='Timestamp', parse_dates=True)
lstm_data = pd.read_csv('lstm.csv', index_col='Timestamp', parse_dates=True)

# Garantir que ambos os datasets tenham o mesmo índice de tempo (intersecção de datas)
merged_data = pd.merge(inversor_data, lstm_data, left_index=True, right_index=True, how='inner')

# Plotar os dados originais no mesmo gráfico
plt.figure(figsize=(10, 6))

# Plot da potência do inversor
plt.plot(merged_data.index, merged_data['kW'], label='Potência CC total (kW)', color='blue')

# Plot do Ghi_kW
plt.plot(merged_data.index, merged_data['Ghi_kW'], label='Ghi_kW (kW)', color='orange')

# Configurações do gráfico
plt.title('Comparação de Potência do Inversor e Ghi_kW - Outubro 2022')
plt.xlabel('Data')
plt.ylabel('Potência (kW)')
plt.legend()
plt.grid(True)

# Exibir o gráfico
plt.show()
