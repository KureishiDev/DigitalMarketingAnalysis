
Análise de Dados de Campanha de Marketing

Este repositório contém um exemplo de análise de dados de uma campanha de marketing, utilizando Python e bibliotecas populares como Pandas, Matplotlib, Seaborn e Scikit-learn. O objetivo é demonstrar como realizar uma análise exploratória de dados (EDA), visualizações, modelagem preditiva e segmentação de público.

Requisitos
Para executar o código, você precisará das seguintes bibliotecas Python:

pandas

matplotlib

seaborn

scikit-learn

Você pode instalar as dependências usando o seguinte comando:

bash
Copy
pip install pandas matplotlib seaborn scikit-learn
Estrutura do Código
O código está dividido em cinco seções principais:

Coleta e Limpeza de Dados

Análise Exploratória de Dados (EDA)

Visualização de Dados

Modelagem Preditiva (Opcional)

Segmentação de Público (Opcional)

1. Coleta e Limpeza de Dados
Nesta seção, os dados da campanha são carregados a partir de um arquivo CSV (dados_campanha.csv). Certifique-se de substituir este arquivo pelo seu conjunto de dados real.

python
Copy
import pandas as pd

# Substitua pelo seu arquivo de dados de campanha real
dados_campanha = pd.read_csv('dados_campanha.csv')
2. Análise Exploratória de Dados (EDA)
Aqui, são realizadas análises descritivas e visualizações iniciais para entender a distribuição dos dados, como a distribuição do Custo por Aquisição (CPA) e as conversões por plataforma.

python
Copy
import matplotlib.pyplot as plt
import seaborn as sns

# Estatísticas descritivas
print(dados_campanha.describe())

# Distribuição do CPA
plt.figure(figsize=(10, 6))
sns.histplot(dados_campanha['cpa'], kde=True)
plt.title('Distribuição do Custo por Aquisição (CPA)')
plt.show()

# Conversões por plataforma
conversões_plataforma = dados_campanha.groupby('plataforma')['conversões'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
conversões_plataforma.plot(kind='bar')
plt.title('Conversões por Plataforma')
plt.show()

# Conversões ao longo do tempo
dados_campanha['data'] = pd.to_datetime(dados_campanha['data'])
conversões_tempo = dados_campanha.groupby('data')['conversões'].sum()
plt.figure(figsize=(15, 6))
conversões_tempo.plot(kind='line')
plt.title('Conversões ao Longo do Tempo')
plt.show()
3. Visualização de Dados
Esta seção inclui visualizações adicionais, como scatter plots e boxplots, para explorar relações entre variáveis, como orçamento e conversões, e a distribuição do CPA por plataforma.

python
Copy
# Scatter plot da relação entre orçamento e conversões
plt.figure(figsize=(10, 6))
sns.scatterplot(x='orcamento', y='conversões', data=dados_campanha)
plt.title('Relação entre Orçamento e Conversões')
plt.show()

# Boxplot do CPA por plataforma
plt.figure(figsize=(12, 6))
sns.boxplot(x='plataforma', y='cpa', data=dados_campanha)
plt.title('Custo por Aquisição (CPA) por Plataforma')
plt.xticks(rotation=45, ha='right')
plt.show()
4. Modelagem Preditiva (Opcional)
Aqui, é demonstrado como construir um modelo de regressão linear simples para prever conversões com base em features como orçamento e cliques.

python
Copy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Preparação dos dados
features = ['orcamento', 'cliques']  # Exemplo de features
target = 'conversões'

dados_modelo = dados_campanha.dropna(subset=features + [target])

X_train, X_test, y_train, y_test = train_test_split(dados_modelo[features], dados_modelo[target], test_size=0.2, random_state=42)

# Treinamento do modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Avaliação do modelo
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio (MSE): {mse}')
5. Segmentação de Público (Opcional)
Nesta seção, é utilizado o algoritmo K-means para segmentar o público com base em dados demográficos, como idade e gênero.

python
Copy
from sklearn.cluster import KMeans

# Supondo que você tenha dados demográficos (idade, gênero)
dados_segmentacao = dados_campanha[['idade', 'genero']].dropna() # Exemplo
kmeans = KMeans(n_clusters=3, random_state=42)
dados_segmentacao['cluster'] = kmeans.fit_predict(dados_segmentacao)

# Visualização dos clusters
sns.scatterplot(x='idade', y='genero', hue='cluster', data=dados_segmentacao)
plt.title('Segmentação de Público (K-means)')
plt.show()
Como Usar
Clone este repositório.

Instale as dependências necessárias.

Substitua o arquivo dados_campanha.csv pelo seu conjunto de dados.

Execute o código no seu ambiente Python.

Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para melhorar este projeto.

Licença
Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

