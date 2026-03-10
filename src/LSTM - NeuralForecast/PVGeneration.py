def PVGeneration(df, Npv, Vmpp, Impp, Voc, Isc, Kv, Ki):
    # Fator de Forma (Fill Factor)
    FF = (Vmpp * Impp) / (Voc * Isc)
    Not = 42  # Temperatura de operação nominal

    # Cálculo de Tc, V, I e Ppv
    df['Tc'] = df['DewpointTemp'] + (df['POA'] / 1000) * (Not - 20) / 0.8
    df['V'] = Voc * (1 + Kv * (df['Tc'] - 25) / 100)
    df['I'] = (df['POA'] / 1000) * Isc * (1 + Ki * (df['Tc'] - 25) / 100)
    df['Ppv'] = 0.9 * Npv * df['V'] * df['I'] * FF / 1000  # Ppv em kW

    # Remover colunas intermediárias Tc, V e I
    df.drop(columns=['Tc', 'V', 'I'], inplace=True)
    df = df.round(2)

    # Retornar o DataFrame completo com a nova coluna 'Ppv'
    return df
