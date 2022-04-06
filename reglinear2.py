import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from yellowbrick.regressor import ResidualsPlot

#%%

Taxa_anual = input('Digite aqui a taxa anual: ')


#%%

# importando e verificando os dados
df = pd.read_csv('slr12.csv', sep=';')
sns.histplot(df)
plt.tight_layout()

#%%

# Definindo variáveis
x = df.iloc[:, 0].values  # Preço anual = variável independente
y = df.iloc[:, 1].values  # Custo inicial da franquia = Variável dependente

sns.scatterplot(x, y)
plt.tight_layout()
                
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

# Gráfico para visualizar os resíduos

vis_residuos = ResidualsPlot(modelo, hist = True, qqplot = False)
vis_residuos.fit(x, y)
vis_residuos.poof()
plt.tight_layout()

#%%

# Previsão usando a função do sklearn
prev = modelo.predict([[Taxa_anual]])
print(prev)



