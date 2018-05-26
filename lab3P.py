# -*- coding: utf-8 -*-
"""
Created on Sun May  8 06:28:03 2016

@author: Francisco
"""

import numpy as np
from numpy import sinc, linspace, cos, arange, convolve
import matplotlib.pyplot as plt
from math import pi
import scipy.signal as signal
from scipy.fftpack import fft
from numpy.fft import fftshift
import random


#==============================================================================
# Qué hace la función?: Genera un pulso coseno alzado
# Parámetros de entrada: x (vector tiempo), alfa, y periodo
# Parámetros de salida: Vector con correspondiente al pulso requerido
#==============================================================================
def Prc(x, alfa,T):
    xx = x/T
    alfaCuadrado = alfa**2
    tiempoCuadrado = x**2
    divisor = 1-(4*alfaCuadrado*tiempoCuadrado)/(T**2)
    dividendo = sinc(xx)*cos(alfa*pi*xx)
    salida = dividendo/divisor
    
    return salida
    
#==============================================================================
# Qué hace la función?: Grafica el cualquier pulso
# Parámetros de entrada: Vector con la señal (EJE Y) y vector con el tiempo (EJE X), titulo del grafico, nombre eje x e y
# Parámetros de salida: NINGUNO
#==============================================================================    
def graficar_pulso(señal,x,titulo,nombre_eje_x,nombre_eje_y):
    plt.plot(x,señal,"-")
    plt.title(titulo)
    plt.xlabel(nombre_eje_x)
    plt.ylabel(nombre_eje_y)
    plt.show()
    
#==============================================================================
# Qué hace la función?: Grafica el cualquier pulso
# Parámetros de entrada: Vector con la señal (EJE Y), titulo del grafico, nombre eje x e y
# Parámetros de salida: NINGUNO
#==============================================================================    
def graficar_conv(señal,titulo,nombre_eje_x,nombre_eje_y):
    plt.plot(señal,"-")
    plt.title(titulo)
    plt.xlabel(nombre_eje_x)
    plt.ylabel(nombre_eje_y)
    plt.show()
    
    
#==============================================================================
# Qué hace la función?: Grafica el cualquier pulso
# Parámetros de entrada: Vector con la señal (EJE Y), titulo del grafico, nombre eje x e y
# Parámetros de salida: NINGUNO
#==============================================================================    
def graficar_diagrama_ojo(señal,titulo,nombre_eje_x,nombre_eje_y):
    cont = 0
    while cont < len(señal):
        var_aux = señal[cont:cont+16]
        if len(var_aux) > 10:
            if var_aux[8] > 0.9 or var_aux[8] < -0.9:
                plt.plot(var_aux,"-")
        cont = cont +16
        

    plt.title(titulo)
    plt.xlabel(nombre_eje_x)
    plt.ylabel(nombre_eje_y)
    plt.show()
    

    
#==============================================================================
# Qué hace la función?: Genera la transformada de fourier de una señal dada
# Parámetros de entrada: Vector con la señal a transformar y el largo de la señal
# Parámetros de salida: Vector con la amplitud transformada y un vector con las frecuencias
#==============================================================================    
def obtener_transformada_fourier(señal,largo_señal):
    transFour =  fftshift(abs(fft(señal,largo_señal)))#eje Y
    return 2*transFour/10
    

#==============================================================================
# Qué hace la función?: 
# Parámetros de entrada: 
# Parámetros de salida: 
#==============================================================================    
def obtener_n_bits_random(numero_bits):
    lista = []
    opciones = [-1,1]
    cont = 0
    while cont < numero_bits:
        var=random.choice(opciones)
        lista.append(var)
        cont = cont +1
    
    return lista
    
#==============================================================================
# Qué hace la función?: 
# Parámetros de entrada: 
# Parámetros de salida: 
#==============================================================================    
def obtener_vector(lista_bits):
    lista = []
    for i in lista_bits:
        cont = 0
        while cont < 16:
            lista.append(0)
            cont+=1
        lista.append(i)

    return lista
  






    


## PARTE 1 

# 1.1 graficar pulsos (sinc y Prc) con respecto al tiempo
# A CONTINUACIÓN SE GENERA EL PULSO SINC
x = linspace(-10,10,100) # genero vector de tiempo
T = 1 #defino un periodo
xx = x/T #normalizo el tiempo
resulSinc = sinc(xx) # genero vector de sinc con tiempo normalizado
## A CONTINUACIÓN PLOTEO EL PULSO SINC
graficar_pulso(resulSinc,x, "Pulso función 'Sinc()' respecto tiempo", "Tiempo [s]", "Amplitud [dB]")

# A CONTINUACIÓN SE GENERA EL PULSO Pcr
alfa = 0.22
prc = Prc(x,alfa,T)
## A CONTINUACIÓN PLOTEO EL PULSO Prc
graficar_pulso(prc,x, "Pulso función 'Prc()' respecto tiempo", "Tiempo [s]", "Amplitud [dB]")

## 1.2 graficar pulsos (sinc y Prc) con respecto a la frecuencia
# A continuación aplico la transformada de fourier al pulso sinc
ySinc = obtener_transformada_fourier(resulSinc,len(resulSinc))
ySinc2 = fftshift(ySinc)
graficar_pulso(ySinc,x, "Pulso función 'Sinc()' respecto frecuencia", "Frecuencia [Hz]", "Amplitud [dB]")

# A continuación aplico la transformada de fourier al pulso Prc
yPrc = obtener_transformada_fourier(prc,len(prc))
graficar_pulso(yPrc,x, "Pulso función 'Prc()' respecto frecuencia", "Frecuencia [Hz]", "Amplitud [dB]")

# 1.3 variar alfo y comparar

prc1 = Prc(x,0.01,T)
## A CONTINUACIÓN PLOTEO EL PULSO Prc
graficar_pulso(prc1,x, "Pulso función 'Prc()' respecto tiempo alfa = 0.01", "Tiempo [s]", "Amplitud [dB]")
yPrc1 = obtener_transformada_fourier(prc1,len(prc1))
graficar_pulso(yPrc1,x, "Pulso función 'Prc()' respecto frecuencia alfa = 0.01", "Frecuencia [Hz]", "Amplitud [dB]")

prc22 = Prc(x,0.2,T)
## A CONTINUACIÓN PLOTEO EL PULSO Prc
graficar_pulso(prc22,x, "Pulso función 'Prc()' respecto tiempo alfa = 0.2", "Tiempo [s]", "Amplitud [dB]")
yPrc22 = obtener_transformada_fourier(prc22,len(prc22))
graficar_pulso(yPrc22,x, "Pulso función 'Prc()' respecto frecuencia alfa = 0.2", "Frecuencia [Hz]", "Amplitud [dB]")

prc2 = Prc(x,0.5,T)
## A CONTINUACIÓN PLOTEO EL PULSO Prc
graficar_pulso(prc2,x, "Pulso función 'Prc()' respecto tiempo alfa = 0.5", "Tiempo [s]", "Amplitud [dB]")
yPrc2 = obtener_transformada_fourier(prc2,len(prc2))
graficar_pulso(yPrc2,x, "Pulso función 'Prc()' respecto frecuencia alfa = 0.5", "Frecuencia [Hz]", "Amplitud [dB]")

prc3 = Prc(x,0.95,T)
## A CONTINUACIÓN PLOTEO EL PULSO Prc
graficar_pulso(prc3,x, "Pulso función 'Prc()' respecto tiempo alfa = 0.95", "Tiempo [s]", "Amplitud [dB]")
yPrc3 = obtener_transformada_fourier(prc3,len(prc3))
graficar_pulso(yPrc3,x, "Pulso función 'Prc()' respecto frecuencia alfa = 0.95", "Frecuencia [Hz]", "Amplitud [dB]")


## PARTE 2
# 2.1 Mostrar señal resultante de enviar 10 bits
# A continuacion  obtengo los bits random (1 y -1)
bits1 = obtener_n_bits_random(10)
#genero la señal
señal_res = obtener_vector(bits1)
#convoluciono la señal con la función sinc y la funcion prc
aaa = convolve(resulSinc,señal_res)
aaa1 = convolve(prc, señal_res)

#grafico ambas convoluciones
graficar_conv(aaa, "Convolucion 10 bits con Sinc()", "Tiempo [s]", "Amplitud [dB]")
graficar_conv(aaa1, "Convolucion 10 bits con Prc()", "Tiempo [s]", "Amplitud [dB]")

## 2.2 simular  transmision con 10 elevado a 4 bits
# A continuacion  obtengo los bits random (1 y -1)
bits2 = obtener_n_bits_random(10**4)
#genero la señal
señal_res2 = obtener_vector(bits2)
#convoluciono la señal con la función sinc y la funcion prc
aaa2 = convolve(resulSinc,señal_res2)
aaa12 = convolve(prc, señal_res2)


#grafico ambas convoluciones
graficar_diagrama_ojo(aaa2, "Diagrama de ojo convolucion 10^4 bits con Sinc()", "Tiempo [s]", "Amplitud [dB]")
graficar_diagrama_ojo(aaa12, "Diagrama de ojo convolucion 10^4 bits con Prc()", "Tiempo [s]", "Amplitud [dB]")


## 2.3 agregar ruido y graficar diagrama de ojo
# a continuacion genero el ruido y lo agrego a cada vector
noise = np.random.normal(0, 0.1, len(aaa2))
aaa3 = aaa2+noise
aaa13 = aaa12+noise

#grafico ambas convoluciones
graficar_diagrama_ojo(aaa3, "Diagrama de ojo convolucion 10^4 bits con Sinc() con ruido", "Tiempo [s]", "Amplitud [dB]")
graficar_diagrama_ojo(aaa13, "Diagrama de ojo convolucion 10^4 bits con Prc() con ruido", "Tiempo [s]", "Amplitud [dB]")

    
    


