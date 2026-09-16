#  Algoritmo de Regressão Linear em Python Do Zero (`Python-IA`)

![Python](https://img.shields.io/badge/Python-3873A9?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=Matplotlib&logoColor=black)

Implementação *from scratch* (do zero) de um modelo de **Regressão Linear** e otimização por **Gradiente Descendente** usando pura matemática de matrizes com **NumPy** e visualização em **Matplotlib** — sem frameworks de alto nível como PyTorch, TensorFlow ou Scikit-Learn!

---

##  Visão Geral

Este projeto demonstra o funcionamento básico e a intuição matemática por trás do treinamento de modelos aprendidos supervisionados:
1. **Modelagem de Dados**: Relação linear entre variáveis ($y = w \cdot x + b$).
2. **Função de Custo/Perda**: Mensuração do erro através do Erro Quadrático Médio (MSE).
3. **Otimização**: Atualização dos parâmetros via *Backpropagation* manual recalculando os gradientes da perda.
4. **Visualização**: Plotagem em ambiente *dark mode* para comparar os dados reais e a reta ajustada.

---

##  A Matemática por Trás do Código

### 1. Modelo Linear
$$\hat{y} = w \cdot x + b$$

### 2. Função de Custo (Mean Squared Error - MSE)
$$L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

### 3. Backpropagation (Derivadas Parciais dos Gradientes)
Para atualizar os pesos $w$ e o bias $b$, calculamos as derivadas parciais do erro $L$ em relação a cada um dos parâmetros:

$$\frac{\partial L}{\partial w} = -\frac{2}{N} \sum x_i \cdot (y_i - \hat{y}_i)$$

$$\frac{\partial L}{\partial b} = -\frac{2}{N} \sum (y_i - \hat{y}_i)$$

### 4. Atualização dos Parâmetros (Regra da Cadeia)
$$w \leftarrow w - \alpha \cdot \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \alpha \cdot \frac{\partial L}{\partial b}$$

*(onde $\alpha$ é a taxa de aprendizado ou Learning Rate)*

---

##  Estrutura das Funções no Código

| Função | Descrição |
| :--- | :--- |
| `getLinear()` | Gera uma massa de dados sintética com padrão linear e suporte a ruído estatístico. |
| `fome()` | Executa o *forward pass* do modelo ($\hat{y} = w \cdot x + b$). |
| `ms()` | Calcula o Erro Quadrático Médio entre a saída prevista e a meta real. |
| `BP()` | Realiza o *Backpropagation*, calculando os derivadas parciais e atualizando $w$ e $b$. |
| `model_fit()` | Loop principal de treinamento que executa as épocas e exibe a redução do *Loss*. |

---

##  Requisitos e Instalação

### Pré-requisitos
Certifique-se de ter o Python 3 instalado no seu ambiente.

### 1. Clonar o Repositório
```bash
git clone [https://github.com/MiguelCarnio/Python-IA.git](https://github.com/MiguelCarnio/Python-IA.git)
cd Python-IA
