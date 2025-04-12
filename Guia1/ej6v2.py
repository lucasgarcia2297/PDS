import numpy as np

def senoidal(tiempo_inicial, tiempo_final, frecuencia_senoidal, frecuencia_muestreo, fase):
    Ts = 1 / frecuencia_muestreo  # Período de muestreo
    tiempo = np.arange(tiempo_inicial, tiempo_final, Ts)  # Instantes de muestreo
    y = np.sin(2 * np.pi * frecuencia_senoidal * tiempo + fase)  # Señal muestreada
    return tiempo, y


# Funciones de interpolación
def sinc(fs,arg):
    t = 2*np.pi*fs*arg
    x = 0
    if t == 0:
        x = 1
    else:
        x = np.sin(t) / t
    return x

# Una versión mas intuitiva, mas parecida a la del pdf
def interpolacion(t0, y0, factorSobremuestreo = 4):
    # Se asumen valores equiespaciados

    # Período original
    T = t0[1] - t0[0]

    # Período nuevo
    T_i = T / factorSobremuestreo
    
    
    # Tamaño del tiempo original
    N = len(t0)

    # Tamaño del tiempo nuevo
    M = N * factorSobremuestreo
    

    # Inicialización de las señales interpoladas
    t1 = np.linspace(t0[0], t0[-1] + T, M, endpoint=False)
    y1 = np.zeros(M)

    # Interpolación
    for n in range(N):
        for m in range(M):
            arg = ((m*T_i)/T - n)

            # Interpolador sinc
            y1[m] += y0[n] * sinc(arg, 0.5)

    return t1, y1
