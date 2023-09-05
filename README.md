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
