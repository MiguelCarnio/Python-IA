import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams['figure.figsize']=(10,8)
def ativacao_sigmoid(x: float) -> float:
    return (1/(1+np.e**(-x)))

idade = np.array([0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9])

valor = np.array([((x-0.3)**2 - 0.45*x + 0.3)*2 + np.random.randn()/60 for x in idade])

plt.scatter(idade, valor, s=100)
plt.xlabel('Idade')
plt.ylabel('Valor')
plt.title('Valor do seguro em função da idade (valores normalizados)', fontdict={'size': 12})
plt.show()
