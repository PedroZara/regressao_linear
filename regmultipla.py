import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

#%%

df = pd.read_csv('mt_cars.csv')

#%%

# Excluir coluna com nomes
df = df.drop(['Unnamed: 0'], axis = 1)

#%%

# Criaçã]o de x e y: variável independente e dependente
x = df.iloc[:, 2].values  # Coluna disp
y = df.iloc[:, 0].values  # Coluna mpg
correlacao = np.corrcoef(x,y)
plt.scatter(x, y)


#%%

# Mudança no formato de x para formato de matrix (necessário par aversões mais recentes do sklearn)
x = x.reshape(-1, 1)

#%%

# Criação de modelo, treinamento, visualização dos coeficientes e score do modelo
modelo = LinearRegression()
modelo.fit(x, y)

#%%

# Interceptação
m_i = modelo.intercept_

#%%

# Inclinação
m_c = modelo.coef_

#%%

# Score
m_s = modelo.score(x, y)

#%%

# Gerando previsões
m_prev = modelo.predict(x)

#%%

# Ciração do modelo, utilizando a biblioteca statsmodel, podemos ver r ajustado do r2
m_ajustado = sm.ols(formula = 'mpg ~ disp', data = df)
m_treinado = m_ajustado.fit()
m_trein1 = m_treinado.summary()

#%%

# Visualização dos resultados
plt.scatter(x, y)
plt.plot(x, m_prev, color = 'red')

#%%

# Previsão para somente um valor
mprev_200 = modelo.predict([[200]])

#%%

# Criação de novas variáveis e novo modelo para comparação com o anterior, 3 variáveis dependentes para prever mpg
# cyl - disp - hp
x1 = df.iloc[:, 1:4].values
y1 = df.iloc[:, 0].values

#%%

modelo2 = LinearRegression()
modelo2.fit(x1, y1)

# R^2
m2_s = modelo2.score(x1, y1)

#%%

# Criação do modelo ajustado com mais atributos (regressão linear múltipla)
m_ajustado2 = sm.ols(formula = 'mpg ~ cyl + disp + hp', data = df)
m_treinado2 = m_ajustado2.fit()
m_trein2 = m_treinado2.summary()

#%%

# Previsão de um novo registro com um número para cada uma das 3 variáveis utilizadas no ultimo modelo
novo_reg = np.array([4, 200, 100])
novo_reg = novo_reg.reshape(1, -1)
mod2_novo = modelo2.predict(novo_reg)
