# Folder NSRDB

Neste folder se encontram os dados da base [NSRDB](https://nsrdb.nrel.gov/data-viewer) (National Solar Radiation Database) mantida pelo NREL.


1. Na ferramenta [NSRDB viewer](https://nsrdb.nrel.gov/data-viewer),  insira ou navegue pelo mapa até a coodenada desejada.
2. Selecione o *Dataset* disponível para a localidade (*Datasets Available at Location*), p.ex., **USA & Americas (30, 60min / 4km / 1998-2020)**.
3. Selecionar os atribuitos desejados (*Select Attributes*), p.ex., selecione todos (*select all*) e remova atributos sobre *radiação UV*.
4. Selecionar o período (*Select Year*), p.ex.,  para montar base de dados podem ser selecionados todo o período disponível (*Select all*).
5. Selecionar o intervalo de tempo (*Select Interval*), 30 ou 60 minutos.
6. Clique em *View Code* e copie o código gerado. Este código *python* automativa as requisições e gera os *links* para *download* de cada arquivo e os envia para o *e-mail* fornecido no código. Para realizar o *download* é necessário criar uma conta gratuita no [NREL](https://developer.nrel.gov/signup/) e criar uma **API key**, que deve ser informada no código python.

