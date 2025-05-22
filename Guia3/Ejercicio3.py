import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importar Axes3D para gráficos 3D
# PROCESAMIENTO DIGITAL DE SEÑALES [2024]
# GUIA 03 - EJERCICIO 3
# %{
#   Ejercicio 3: (?) Calcule el error cuadrático total de aproximaci´on en el ejemplo con funciones de Legendre bajo las siguientes condiciones:
#   1. con los coeficientes calculados en el ejemplo, 
#   2. con peque˜nas variaciones en torno a estos coeficentes ?, construyendo una gr´afica en 3D con la variaci´on en los coeficientes en x, y y el error cuadr´atico total en z,
#   3. con m´as coeficientes ?, para comprobar c´ omo se reduce el error cuadr´atico total al aumentar los coeficientes.}
# %}

def cuadrada(tinicial, tfinal, fs, fm, phi, A):
  t = np.arange(tinicial, tfinal, 1/fm)  # Vector de tiempo
  y = A * np.sign(np.sin(2 * np.pi * fs * t + phi))  # Señal cuadrada
  return t, y

tinicial = -1.0  # Tiempo inicial
tfinal = 1.0  # Tiempo final
fs = 0.5 # Frecuencia de la señal
fm = 100  # Frecuencia de muestreo 
phi = 0.0  # Fase inicial
A = 1.0  # Amplitud de la señal

[t,y] = cuadrada(tinicial, tfinal, fs, fm, phi, A)  # Generar señal cuadrada
#===============================INCISO 1========================================
##
### FUNCIONES DE LEGENDRE: (pág 62 - Libro de cátedra Señales)
### Funcion de Legendre de phi0 a phi3 evaluadas en t
##%phi_n = [phi0  ,   phi1,   phi2,   phi3];

phi_3 = np.array([np.ones(len(t))*1/np.sqrt(2),
                  np.sqrt(3/2)*t, 
                  np.sqrt(5/2)*((3/2)*t**2- 1/2),
                  np.sqrt(7/2)*((5/2)*t**3 - (3/2)*t)])

### Coeficientes alpha del ejemplo:
alphas3 = np.array([0, np.sqrt(3/2), 0, -np.sqrt(7/32)]) # Coeficientes del ejemplo

y_aprox3 = phi_3.T @ alphas3  # Aproximación de la señal cuadrada

### GRÁFICA
plt.figure(figsize=(10, 6))
plt.stem(t, y,'k-', label='Señal Cuadrada', basefmt=' ', use_line_collection=True)
plt.grid()
plt.stem(t, y_aprox3, 'r-',label='Aproximación con 4 coeficientes', basefmt=' ', use_line_collection=True)
plt.title('Aproximación de la señal cuadrada con funciones de Legendre')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend() 
plt.show()

# Error cuadrático total
error_cuadratico_total3 = np.sum((y - y_aprox3)**2)  # Error cuadrático total
print(f"Error cuadrático total con 4 coeficientes: {error_cuadratico_total3:.4f}")

#================================================================================
#==================================INCISO 2 =====================================
# Variaciones pequeñas en los coeficientes α1 (índice 1) y α3 (índice 3)
variaciones = np.arange(-10, 10 + 0.5, 0.5)  # Paso fino

# Crear malla para las variaciones
X, Y = np.meshgrid(variaciones, variaciones)

# Inicializar matriz de errores
error_cuadratico_medioB = np.zeros_like(X, dtype=float)

# Calcular ECM para cada combinación
for i in range(len(variaciones)):
    for j in range(len(variaciones)):
        alphasB = alphas3.copy()
        alphasB[1] = X[i, j]  # Variación en α1
        alphasB[3] = Y[i, j]  # Variación en α3
        y_aproxB = phi_3.T @ alphasB
        error_cuadratico_medioB[i, j] = np.mean((y - y_aproxB) ** 2)  # ECM

# Gráfico 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, error_cuadratico_medioB, cmap='viridis')

ax.set_title('ECM según variaciones en coeficientes α₁ y α₃')
ax.set_xlabel('Variación en α₁ (coef. índice 1)')
ax.set_ylabel('Variación en α₃ (coef. índice 3)')
ax.set_zlabel('Error Cuadrático Medio')

## marcar el mínimo en la gráfica 3D
min_index = np.unravel_index(np.argmin(error_cuadratico_medioB), error_cuadratico_medioB.shape)
min_alpha1_variation = X[min_index]
min_alpha3_variation = Y[min_index]
min_error = error_cuadratico_medioB[min_index]
print(f"🔍 Mínimo ECM encontrado:"
      f"\n - Variación en α₁ (índice 1): {min_alpha1_variation:.4f} es igual a {alphas3[1]:.4f}"
      f"\n - Variación en α₃ (índice 3): {min_alpha3_variation:.4f} es igual a { alphas3[3]:.4f}"
      f"\n - Variación en α₃ (índice 3): {min_alpha3_variation:.4f}"
      f"\n - Error cuadrático medio mínimo: {min_error:.6f}")

ax.scatter(X[min_index], Y[min_index], min_error, color='r', s=100, label='Mínimo ECM')
ax.legend()
fig.colorbar(surf, label='ECM')
plt.show()

print("Rango de variaciones aplicadas:")
print(f"α₁ (coef. índice 1): {variaciones[0]} a {variaciones[-1]}")
print(f"α₃ (coef. índice 3): {variaciones[0]} a {variaciones[-1]}")

# Buscar el índice del mínimo ECM
min_index = np.unravel_index(np.argmin(error_cuadratico_medioB), error_cuadratico_medioB.shape)

# Obtener las variaciones correspondientes
min_alpha1_variation = X[min_index]
min_alpha3_variation = Y[min_index]
min_error = error_cuadratico_medioB[min_index]

print(f"\n🔍 Mínimo ECM encontrado:")
print(f" - Variación en α₁ (índice 1): {min_alpha1_variation:.4f}")
print(f" - Variación en α₃ (índice 3): {min_alpha3_variation:.4f}")
print(f" - Error cuadrático medio mínimo: {min_error:.6f}")

#================================================================================

#================================INCISO 3========================================
# Funciones de Legendre ortonormales hasta orden n
def legendre_ortonormales(n, t):
    from numpy.polynomial.legendre import legval
    phi = []
    for k in range(n+1):
        coef = np.zeros(k+1)
        coef[-1] = 1  # P_k
        Pk = legval(t, coef)
        norm = np.sqrt((2*k + 1)/2)
        phi_k = norm * Pk
        phi.append(phi_k)
    return np.array(phi)

# Parámetros
tinicial = -1.0
tfinal = 1.0
fs = 0.5
fm = 100
phi = 0.0
A = 1.0

t, y = cuadrada(tinicial, tfinal, fs, fm, phi, A)
dt = 1/fm
errores = []
ordenes = range(4,11,2)


plt.figure(figsize=(10, 6))
# Aproximaciones sucesivas
import matplotlib.pyplot as plt
import numpy as np

colores = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple']

for i, n in enumerate(ordenes):
    phi_n = legendre_ortonormales(n, t)
    alphas_n = np.array([np.sum(y * phi_k) * dt for phi_k in phi_n])
    y_aprox_n = phi_n.T @ alphas_n

    error_n = np.sum((y - y_aprox_n)**2)
    errores.append(error_n)

    color = colores[i % len(colores)] 
    plt.stem(t, y_aprox_n, linefmt=color + '-', markerfmt=color + 'o', basefmt=' ', 
             label=f'Aproximación con {n} coef.', use_line_collection=True)

plt.stem(t, y,'k-', label='Señal Cuadrada', basefmt=' ', use_line_collection=True)
plt.grid()
plt.title('Aproximación de la señal cuadrada con funciones de Legendre')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend() 
plt.show()

# Graficar evolución del error
plt.figure(figsize=(10, 6))
plt.plot(ordenes, errores, 'o-', color='crimson')
plt.title('Error cuadrático total vs cantidad de coeficientes α')
plt.xlabel('Cantidad de coeficientes (orden de la base + 1)')
plt.ylabel('Error cuadrático total')
plt.grid(True)
plt.show()
#================================================================================