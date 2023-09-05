# PVforecast
### Projeto PIBIC Edital 03/2023 - Picti, Ifes Campus Linhares
Orientador: Prof. Alysson Machado - alysson.machado@ifes.edu.br

Orientado: Joicy Nunes Bastos

Título: **Previsão de geração de energia solar fotovoltaica utilizando dados meteorológicos e aprendizagem de máquina.**

Plano de trabalho (12727): **Desenvolvimento do algoritmo de previsão de geração de energia solar fotovoltaica utilizando dados meteorológicos e aprendizagem de máquina.**

## Objetivos 
Investigar, analisar e experimentar a utilização da tecnologia aprendizado de máquinapara a previsão de geração solar fotovoltaica para um horizonte de 24 horas à frente. 
Ao final do projeto obter de forma qualitativa e quantitativa a qualidade do sistema de previsão de geração utilizando alguns métodos de aprendizagem distintos. Analisar a relevância dos dados metereológicos na previsão.

## Requisitos
É projeto é desenvolvido e testado em ambiente Linux, para usuários Windows é altamente recomendado o uso do [WSL](https://docs.microsoft.com/en-us/windows/wsl/install).

## Instalação
Primeiramente em um diretório local de preferência faça o clone do repositório:
```shell
cd <path>
git clone https://github.com/AlyssonM/PVforecast.git
```
Dentro do repositório local criar um ambiente virtual python, em seguida o ative e realize a instalação das dependências:
```shell
cd PVforecast
python -m venv ./.venv
source activate .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
## VS Code
É recomendado o uso do [VS code](https://code.visualstudio.com/download) integrado com a [extensão do WSL](https://learn.microsoft.com/pt-br/windows/wsl/tutorials/wsl-vscode) 

## Revisão Bibliográfica
1. SILVA, L. T. **Aprendizado de máquina aplicado na Previsão da Geração de Energia Elétrica de uma Usina Solar Fotovoltaica**. Trabalho de Conclusão de Curso - Universidade Federal do Ceará. Fortaleza, p. 60. 2022. Disponível em: https://repositorio.ufc.br/handle/riufc/65867.
2. CUNHA, B. A. **Previsão intra-diária de geração fotovoltaica usando redes neurais recorrentes do tipo LSTM e dados históricos de energia**. Tese (Doutorado em Engenharia Elétrica) -  Universidade Estadual Paulista Júlio de Mesquita Filho. Bauru, p. 80. 2021. Disponível em: https://repositorio.unesp.br/handle/11449/215639.
3. LAHOUAR, A.; MEJRI, A.; BEN HADJ SLAMA, J. **Importance based selection method for day-ahead photovoltaic power forecast using random forests.** International Conference on Green Energy and Conversion Systems, GECS 2017, 2017. https://doi.org/10.1109/GECS.2017.8066171.
4. THAKER, J.; HÖLLER, R. **Evaluation of High Resolution WRF Solar**. Energies, vol. 16, no. 8, p. 1–13, 2023. https://doi.org/10.3390/en16083518.
