import pandas as pd
import numpy as np

def calculate_POA_for_year(df, tilt, azimuth, latitude):
    # Converter graus para radianos
    tilt_rad = np.deg2rad(tilt)
    azimuth_rad = np.deg2rad(azimuth)
    latitude_rad = np.deg2rad(latitude)
    
    # Inicialização da coluna POA no DataFrame como float64
    df['POA'] = 0.0

    # Iterar sobre cada dia único no DataFrame
    for day in df.index.normalize().unique():
        # Filtrar o DataFrame para o dia atual e horários entre 06:00 e 19:00
        df_day = df.loc[(df.index >= day + pd.Timedelta(hours=6)) & 
                        (df.index <= day + pd.Timedelta(hours=19))]

        # Calcular o dia do ano para a declinação solar
        dayOfYear = day.dayofyear

        # Declinação solar
        solarDeclination = 0.409 * np.sin(2 * np.pi * (dayOfYear - 81) / 368)

        for index, row in df_day.iterrows():
            # Calcular a hora a partir do índice datetime
            hour = index.hour + index.minute / 60.0

            # Ângulo horário (15° por hora a partir do meio-dia solar)
            hourAngle = np.deg2rad(15 * (hour - 12))

            # Cálculo do ângulo de elevação solar
            solarElevationAngle = np.arcsin(
                np.sin(latitude_rad) * np.sin(solarDeclination) + 
                np.cos(latitude_rad) * np.cos(solarDeclination) * np.cos(hourAngle)
            )

            # Cálculo do ângulo de incidência
            cosIncidenceAngle = (
                np.sin(solarElevationAngle) * np.cos(tilt_rad) + 
                np.cos(solarElevationAngle) * np.sin(tilt_rad) * np.cos(azimuth_rad - hourAngle)
            )

            # Calcular e atribuir o valor de POA (evitar valores negativos)
            df.at[index, 'POA'] = row['Ghi'] * max(cosIncidenceAngle, 0)

    # Arredondar todas as colunas numéricas para duas casas decimais
    df = df.round(2)

    return df
