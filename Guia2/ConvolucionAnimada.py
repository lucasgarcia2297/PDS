import numpy as np
import matplotlib.pyplot as plt

# Definición de las señales
x = np.array([1, 2, 2])
h = np.array([2, 1, 0.5])
y = np.zeros(len(x) + len(h) - 1)

# Configuración de la figura
plt.figure(figsize=(10, 8))
plt.subplots_adjust(hspace=0.5)

# Función para mostrar el estado actual
def plot_step(i, j=None):
    plt.clf()  # Limpiar la figura
    
    # Señal x[n]
    plt.subplot(3, 1, 1)
    markerline, stemlines, baseline = plt.stem(x, basefmt='k', linefmt='b-', markerfmt='bo')
    plt.title(f"Señal $x[n]$ (Posición actual: i={i})")
    if i < len(x):
        markerline.set_markerfacecolor('b')
        markerline.set_markeredgecolor('b')
        markerline._y[i] = x[i]  # Resaltar elemento actual
    
    # Señal h[n] (desplazada)
    plt.subplot(3, 1, 2)
    shifted_h = np.zeros(len(y))
    if i < len(x):
        shifted_h[i:i+len(h)] = h
    markerline, stemlines, baseline = plt.stem(shifted_h, basefmt='k', linefmt='r-', markerfmt='ro')
    plt.title(f"Señal $h[n]$ desplazada (Posición: i={i})")
    if j is not None and j < len(h):
        markerline.set_markerfacecolor('r')
        markerline.set_markeredgecolor('r')
        markerline._y[i+j] = h[j]  # Resaltar elemento actual
    
    # Resultado parcial y[n]
    plt.subplot(3, 1, 3)
    markerline, stemlines, baseline = plt.stem(y, basefmt='k', linefmt='m-', markerfmt='mo')
    plt.title("Resultado parcial $y[n]$")
    if i < len(x) and j is not None:
        if (i+j) < len(y):
            markerline.set_markerfacecolor('m')
            markerline.set_markeredgecolor('m')
            markerline._y[i+j] = y[i+j]  # Resaltar elemento actual
    
    # Texto explicativo
    if j is None:
        plt.suptitle(f"Preparado para comenzar. Presiona Enter para iniciar...", y=1.02)
    elif i < len(x):
        plt.suptitle(f"Calculando: y[{i+j}] += x[{i}]×h[{j}] = {x[i]}×{h[j]} = {x[i]*h[j]}", y=1.02)
    else:
        plt.suptitle("¡Convolución completa! Presiona Enter para salir...", y=1.02)
    
    plt.draw()
    plt.pause(0.1)

# Mostrar estado inicial
plot_step(0, None)
plt.waitforbuttonpress()

# Proceso de convolución paso a paso
for i in range(len(x)):
    plot_step(i, None)
    plt.waitforbuttonpress()
    
    for j in range(len(h)):
        if (i + j) < len(y):
            y[i + j] += x[i] * h[j]
            plot_step(i, j)
            plt.waitforbuttonpress()

# Mostrar resultado final
plot_step(len(x), None)
plt.waitforbuttonpress()
plt.close()

print("Resultado final de la convolución:", y)