import random
from datetime import datetime, timedelta
import pandas as pd
import calendar

# Função para gerar valores aleatórios dentro de um intervalo
def random_within_range(min_val, max_val, decimal_places=2):
    return round(random.uniform(min_val, max_val), decimal_places)

# Definição do cabeçalho
header1 = ["99999",'"IFES LINHARES"',"ES","-3.0","-19.40","-40.04","295"]
header2 = ["Date (MM/DD/YYYY),Time (HH:MM),ETR (W/m^2),ETRN (W/m^2),GHI (W/m^2),GHI source,GHI uncert (%),DNI (W/m^2),DNI source,DNI uncert (%),DHI (W/m^2),DHI source,DHI uncert (%),GH illum (lx),GH illum source,Global illum uncert (%),DN illum (lx),DN illum source,DN illum uncert (%),DH illum (lx),DH illum source,DH illum uncert (%),Zenith lum (cd/m^2),Zenith lum source,Zenith lum uncert (%),TotCld (tenths),TotCld source,TotCld uncert (code),OpqCld (tenths),OpqCld source,OpqCld uncert (code),Dry-bulb (C),Dry-bulb source,Dry-bulb uncert (code),Dew-point (C),Dew-point source,Dew-point uncert (code),RHum (%),RHum source,RHum uncert (code),Pressure (mbar),Pressure source,Pressure uncert (code),Wdir (degrees),Wdir source,Wdir uncert (code),Wspd (m/s),Wspd source,Wspd uncert (code),Hvis (m),Hvis source,Hvis uncert (code),CeilHgt (m),CeilHgt source,CeilHgt uncert (code),Pwat (cm),Pwat source,Pwat uncert (code),AOD (unitless),AOD source,AOD uncert (code),Alb (unitless),Alb source,Alb uncert (code),Lprecip depth (mm),Lprecip quantity (hr),Lprecip source,Lprecip uncert (code),PresWth (METAR code),PresWth source,PresWth uncert (code)"]

# Definição da data de início e intervalo de tempo
year = ['2018', '2019', '2020', '2021']
#start_date = datetime(int(year[0])-1, 12, 31, 23, 00, 0, 0)
start_date = datetime(int(year[0]), 1, 1, 0, 00, 0, 0)
current_time = start_date
#start_date = datetime(2020, 12, 31)
time_interval = timedelta(hours=0,minutes=30)

# Lista para armazenar os dados
data = []
ghi = []
dni = []
dhi = []
Albedo = []
dry_bulb_temp = []
dew_point_temp = []
Wind_Speed = []
Wind_Dir = []
R_H = []
Pressure = []

for i in range(len(year)):
    filename = "./NSRDB/" + year[i] + "/1984360_-19.39_-40.06_"+ year[i] + ".csv"
    csvdata = pd.read_csv(filename, skiprows=[0,1])
    #csvdata = pd.read_csv("./Solcast/-19.407482_-40.045201_Solcast_PT15M.csv")
    #ghi = csvdata['GHI'].values
    ghi = ghi + [csvdata['GHI'][i] for i in range(len(csvdata['GHI']))]
    #dni = csvdata['DNI'].values
    dni = dni + [csvdata['DNI'][i] for i in range(len(csvdata['DNI']))]
    #dhi = csvdata['DHI'].values
    dhi = dhi + [csvdata['DHI'][i] for i in range(len(csvdata['DHI']))]
    #dry_bulb_temp = csvdata['Temperature'].values
    Albedo = Albedo + [csvdata['Surface Albedo'][i] for i in range(len(csvdata['Surface Albedo']))]
    dry_bulb_temp = dry_bulb_temp + [csvdata['Temperature'][i] for i in range(len(csvdata['Temperature']))]
    #dew_point_temp = csvdata['Dew Point'].values
    dew_point_temp = dew_point_temp + [csvdata['Dew Point'][i] for i in range(len(csvdata['Dew Point']))]
    #Wind_Speed = csvdata['Wind Speed'].values
    Wind_Speed = Wind_Speed + [csvdata['Wind Speed'][i] for i in range(len(csvdata['Wind Speed']))]
    Wind_Dir = Wind_Dir + [csvdata['Wind Direction'][i] for i in range(len(csvdata['Wind Direction']))]
    R_H = R_H + [csvdata['Relative Humidity'][i] for i in range(len(csvdata['Relative Humidity']))] 
    Pressure = Pressure + [csvdata['Pressure'][i] for i in range(len(csvdata['Pressure']))] 
# ghi = csvdata['Ghi'].values
# dni = csvdata['Dni'].values
# dhi = csvdata['Dhi'].values
# dry_bulb_temp = csvdata['AirTemp'].values
# dew_point_temp = csvdata['DewpointTemp'].values
# Wind_Speed = csvdata['WindSpeed10m'].values
# Loop para gerar os dados de cada hora do dia para 2 anos (granularidade de 30 min)

for i in range(0,len(ghi),1):
    
    
    # Geração dos valores aleatórios para irradiância e temperatura
    etr = 0
    etrn = 0

    # Formatação dos valores para o padrão do arquivo TMY3
    # if current_time.hour == 0 and current_time.minute == 0:
    #     date_str = (current_time - timedelta(days=1)).strftime("%m/%d/%Y")
    #     time_str = "{0}:{1}".format("24","00")
    # else:
    date_str = current_time.strftime("%m/%d/%Y")
    time_str = current_time.strftime("%H:%M")
        
    #date_str = (current_time - 3*time_interval).strftime("%m/%d/%Y")
    #time_str = (current_time - 3*time_interval).strftime("%H:%M")
    etr_str = "{:.2f}".format(etr)
    etrn_str = "{:.2f}".format(etrn)
    ghi_str = str(ghi[i])
    dni_str = str(dni[i])
    dhi_str = str(dhi[i])
    Albedo_str = str(Albedo[i])
    dry_bulb_temp_str = (str(dry_bulb_temp[i]))
    dew_point_temp_str = (str(dew_point_temp[i]))
    Wind_Speed_str = (str(Wind_Speed[i]))
    Wind_Dir_str = (str(Wind_Dir[i]))
    R_H_str = (str(R_H[i]))
    Pressure_str = (str(Pressure[i]))
    # Adição dos dados na lista de dados
    data.append([date_str, time_str, etr_str, etrn_str, ghi_str, "1", "10", dni_str,"1", "10", dhi_str, "1", "18", "50", "1", "0", "0", "1", "0", "0", "1", "0", "0", "1", "20", "10", "A", "7", "10", "A", "7", dry_bulb_temp_str, "A", "7", dew_point_temp_str, "A", "7", R_H_str, "A", "7", Pressure_str, "A", "7", Wind_Dir_str, "A", "8", Wind_Speed_str, "B", "7", "370", "A", "7", "2.0", "E", "8", "0.000", "F", "8", "0.00", "?", "0", Albedo_str, "1", "D", "00", "00", "C", "8"])
    if(current_time.month == 2 and current_time.day == 28 and current_time.hour == 23 and current_time.minute == 00):
        if(calendar.isleap(current_time.year)):
            current_time = datetime(current_time.year, 2, 29, 23, 00, 0, 0)
    current_time = current_time + time_interval           
#Escrita dos dados no arquivo TMY3
with open("./ES_Linhares.tmy3", "w+") as file:
    # Escrita do cabeçalho
    file.write(", ".join(header1) + "\n")
    file.write(",".join(header2) + "\n")
    # Escrita dos dados
    for row in data:
        file.write(",".join(row) + "\n")
            
print("Arquivo gerado com sucesso!")


