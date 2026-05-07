#imports
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams['figure.figsize']=(10,8)
#dados
def getLinear(x,w,b=0,noiseS=0):
    return w*x+b+ noiseS* np.random.randn(x.shape[0])
x=np.arange(-50,50,3)
Y=getLinear(x*10,-8746,200000000000000, noiseS=0)
plt.scatter(x,Y)
plt.xlabel('ºC', fontsize=10)
plt.ylabel('ºF', fontsize=10)


#modelo
#inicializar

w=np.random.rand(1)
b=0
#inicio
def fome(in3,w,b):
    return w*in3+b
#perda
def ms(Y,y):
    return (Y-y)**2

#back
def BP(inp,out,target,lr,w,b):
    dw=lr*(-2*inp*(target-out)).mean()
    db=lr*(-2*(target-out)).mean()
    w-=dw
    b-=db
    return w,b
#rodas
def model_fit(inp,target,w,b,epochs=10000,lr=0.001):
    for epoch in range(epochs):
        out=fome(inp,w,b)
        loss=np.mean(ms(target,out))
        w,b=BP(inp,out,target,lr,w,b)
        if (epoch+1)%(epochs/20)==0:
            print(f"Epoch: {(epoch+1)}/{epochs},LOSS: {loss:.4f}")
    return w,b
            #parametros
w,b= model_fit(x,Y,w,b,epochs=20000,lr=0.001)
print(f"w:{w[0]:.3f},b:{b}")
plt.plot(x,getLinear(x,w,b),'r',lw=3)
plt.show()