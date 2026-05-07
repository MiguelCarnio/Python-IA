# imports
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (10,8)

# função do neurônio artificial
def E2(x, w, b=0):
    return 1 / (1 + 2.7**(-(x*w + b)))

# dados
b=-1
w=0.5
x1=np.arange(0, 20.1, 1)
Y1=x1*b-1
x2 = Y1+10
Y2 = E2(x2, w, b)

# gráfico
plt.scatter(x1, Y1)
plt.scatter(x2*5, Y2*5)
plt.xlabel('x', fontsize=10)
plt.ylabel('Saída do neurônio', fontsize=10)

plt.show()





