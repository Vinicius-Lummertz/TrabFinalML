# 🤖 Projeto de Machine Learning: Previsão de Partidas do Brasileirão

**Integrantes:**
*Vinicius Lummertz
*Victor Galato
*...

---

## 1. Etapa 1: Tema e Dataset

### Tema
O projeto se enquadra na área de **Esportes (Análise Peditiva de Futebol)**.

### Justificativa de Relevância
O futebol é o esporte mais popular no Brasil, movimentando paixões e indústrias (mídia, apostas esportivas, análise tática). A capacidade de prever resultados de partidas usando dados históricos é um desafio clássico e complexo em Ciência de Dados. Este projeto visa aplicar técnicas de Machine Learning para criar um modelo preditivo, explorando os fatores estatísticos que influenciam o resultado de um jogo no Campeonato Brasileiro.

### Dataset
O conjunto de dados utilizado será o **"Campeonato Brasileiro de Futebol"**, um dataset público e abrangente disponível na plataforma Kaggle.

* **Fonte:** [Kaggle - Campeonato Brasileiro de Futebol](https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol/data)
* **Descrição:** O dataset é composto por 4 arquivos CSV, contendo informações detalhadas sobre partidas (placar, técnicos, formações), estatísticas (chutes, posse, faltas), gols (autor, minuto) e cartões (atleta, minuto) de diversas temporadas do Brasileirão.

---

## 2. Etapa 2: Formulação do Problema

### Objetivo
Desenvolver um modelo de Machine Learning capaz de prever o resultado final de uma partida do Campeonato Brasileiro, dadas as informações pré-jogo dos times (mandante e visitante) e o contexto da partida (rodada, arena, etc.).

### Tipo de Aprendizado
O problema será abordado como **Aprendizado Supervisionado**.

### Técnica de Modelagem
Será utilizada a técnica de **Classificação Multiclasse**. O modelo deverá prever uma das três classes possíveis para cada partida:

1.  `Vitoria_Mandante`
2.  `Vitoria_Visitante`
3.  `Empate`

---

## 3. Estrutura do Projeto (Em Desenvolvimento)

* `/data`: Armazena os datasets brutos (ignorado pelo .gitignore).
* `/models`: Armazena os modelos treinados (ex: `.joblib`) (ignorado pelo .gitignore).
* `/notebooks`: Contém o notebook final (`AnaliseBrasileirao.ipynb`) para exploração e apresentação.
* `/src`: Contém os scripts Python modularizados para carga, pré-processamento, treino e deploy.
* `requirements.txt`: Lista de dependências do projeto.
* `main.py`: Ponto de entrada da API (FastAPI) para o deploy no Render.