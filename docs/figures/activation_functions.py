from matplotlib.pyplot import show, subplots, tight_layout, grid, savefig
from numpy import arange, heaviside, tanh, sum as np_sum, exp as np_exp
from math import exp


def linear_function(x):
    return x


def binary_step(x):
    return heaviside(x, 1)


def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+exp(-item)))
    return a

def relu(x):
    x1=[]
    for i in x:
        if i<0:
            x1.append(0)
        else:
            x1.append(i)

    return x1


def softmax(x):
    return np_exp(x) / np_sum(np_exp(x), axis=0)

custom_fontdict  = {'fontsize': 11, 'fontweight': 'medium'}

fig, ax = subplots(2, 3, figsize=(9, 6))
x = arange(-10., 10., 0.2)

# Binary step
ax[0, 0].plot(x, binary_step(x))
ax[0, 0].set_title('Funkcja progowa \nunipolarna', fontdict=custom_fontdict)

# Funkcja liniowa
ax[0, 1].plot(x, linear_function(x))
ax[0, 1].set_title('Funkcja liniowa', fontdict=custom_fontdict)

# Tanh
ax[0, 2].plot(x, tanh(x))
ax[0, 2].set_title('Tangens hiperboliczny \n(tanh)', fontdict=custom_fontdict)

# Funkcja sigmoidalna
ax[1, 0].plot(x, sigmoid(x))
ax[1, 0].set_title('Sigmoidalna funkcja \nunipolarna (Sigmoid)', fontdict=custom_fontdict)

# Relu
ax[1, 1].plot(x, relu(x))
ax[1, 1].set_title('RELU', fontdict=custom_fontdict)

# Relu
ax[1, 2].plot(x, softmax(x))
ax[1, 2].set_title('Znormalizowana funkcja \nwykładnicza (Softmax)', fontdict=custom_fontdict)

tight_layout()
savefig('activation_functions.jpg', dpi=150)
show()
