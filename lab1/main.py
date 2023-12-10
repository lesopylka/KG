import matplotlib.pyplot as plt
import numpy as np

def f(x,y,a):
    return  (x**2 + y**2) ** 2 - (a**2) * (x**2 - y**2)

def doit(a):
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5

    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X,Y = np.meshgrid(x, y)

    plt.contour(X, Y, f(X, Y, a), levels=[0], colors='black')
    plt.axis('equal')
    plt.show()
