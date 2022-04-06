# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:06:31 2022

@author: pedro
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from yellowbrick.regressor import ResidualsPlot

#%%

# Importação e verificação de dados
base = pd.read_csv('cars.csv')
sns.histplot(base)
plt.tight_layout()

#%%

# Limpeza e visualização dos dados de forma geral
base = base.drop(['Unnamed: 0'], axis = 1)
sns.histplot(base)
plt.tight_layout()

#%%

# Definindo variáveis
x = base.iloc[:, 1].values  # Distância = variável independente
y = base.iloc[:, 0].values  # Velocidade = Variável dependente

#%%

# Cálculo de correlação entre X e Y:
correlacao = np.corrcoef(x,y)

#%%

# Formato da matriz com uma coluna a mais
x = x.reshape(-1, 1)

# Criação do modelo e treinamento (fit indica que o treinamento deve ser executado)
modelo = LinearRegression()
modelo.fit(x, y)

#%%

# Visualização dos coeficientes
# Intercept = onde x = 0 - intersecção com y
mod_interc = modelo.intercept_

# Coef = coeficiente de inclinação da reta
mod_incli = modelo.coef_

#%%

# Geração do gráfico com os pontos reais e as previsões
plt.scatter(x, y)
plt.plot(x, modelo.predict(x), color = 'red')

#%%

# Previsão da distância "22 pés" usando a formula manual
# Interceptação * inclinação * valor de dist
# Qual velocidade levou 22 pés para parar?

prev_parada_22 = mod_interc + mod_incli * 22

#%%

# Previsão usando a função do sklearn
prev_parada_22skl = modelo.predict([[22]])

#%%

# Gráfico para visualizar os resíduos

vis_residuos = ResidualsPlot(modelo, hist = True, qqplot = False)
vis_residuos.fit(x, y)
vis_residuos.poof()
plt.tight_layout()
