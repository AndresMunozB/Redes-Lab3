# -*- coding: utf-8 -*-
"""
Created on Sat May  5 18:06:02 2018
Integrantes:
    Diego Mellis - 18.663.454-3
    Andrés Muñoz - 19.646.487-5
"""

from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
from numpy import arange, linspace
from pylab import savefig
from scipy.fftpack import fft, ifft
import warnings 
warnings.filterwarnings('ignore')

#==============================================================================
# Función: En base a los datos que entrega beacon.wav se obtiene			   
# los datos de la señal, la cantidad de datos que esta tiene, y el tiempo que  
# que dura el audio.														   
# Parámetros de entrada: Matriz con los datos de la amplitud del audio.
# Parámetros de salida: Vector con la señal a trabajar, el largo de la señal y 
# un vector con los tiempos de la señal.
#==============================================================================
def getDatos(info,rate):
	#Datos del audio.
	senal = info[:,0]
	#print(signal)
	#Largo de todos los datos.
	len_signal = len(senal)
	#Transformado a float.
	len_signal = float(len_signal)
	#Duración del audio.
	time = len_signal/float(rate)
	#print(time)
	#Eje x para el gráfico, de 0 a la duración del audio.
	t = linspace(0, time, len_signal)
	#print(t)
	return senal, len_signal, t

#=============================================================================
# Función: Grafica los datos del audio en función del tiempo.
# Parámetros de entrada: El arreglo con los datos del tiempo y los datos de la
# señal.
# Parámetros de salida: Ninguno, pero se muestra un gráfico por pantalla.
#=============================================================================
def graphTime(t, senal,text,figura):
	plt.plot(t,senal)
	plt.title(text)
	plt.xlabel("Tiempo [s]")
	plt.ylabel("Amplitud [dB]")
	savefig(figura)
	plt.show()

#=============================================================================
# Función: Función que se encarga de obtener la transformada de Fourier de los
# datos de la señal.
# Parámetros de entrada: Un arreglo con los datos de la señal, y el largo de 
# este arreglo.
# Parámetros de salida: Dos arreglos, uno con los valores del eje x y otro con
# los valores del eje y.
#=============================================================================
def fourierTransformation(senal, len_signal):
	fourierT = fft(senal)
	fourierNorm = fourierT/len_signal
	xfourier = np.fft.fftfreq(len(fourierNorm),1/rate)
	return xfourier, fourierNorm

#===============================================================================
# Función: Grafica la transformada de Fourier, usando los arreglos de la función
# anterior.
# Parámetros de entrada: arreglo con los valores del eje x y arreglo con los 
# valores del eje y.
# Parámetros de salida: Ninguno, se muestra un gráfico por pantalla.
#===============================================================================
def graphTransformation(xfourier,yfourier,text,figura):
    
	plt.title(text)
	plt.xlabel("Frecuencia [Hz]")
	plt.ylabel("Amplitud [dB]")
	plt.plot(xfourier,abs(yfourier))
	savefig(figura)
	plt.show()

def modulacion_AM(señal,largo_señal,x,rate,K):
    KK = K/100
    fm=rate/2 #se requiere la mitad de la frecuencia dada por la funcion "read" para obtener la frecuencia de muestreo
    fc=200000 #se requiere una frecuencia portadora minimo 4 veces mayor que la frecuencia de muestreo
    wc=2*pi*fc
    
    tiempo = float(largo_señal)/float(rate)#genero el tiempo del audio 
    #x2 = arange(0,tiempo,(1.0/float(2*fc)))#genero un vector de 0 hasta tiempo con intervalos del porte de la frecuencia
    x2 = linspace(0,tiempo,fc*tiempo)
    señal2=np.interp(x2, x,señal)  
    portadoraX=np.linspace(0,len(x2)/rate, num=len(x2))

    
    señal_portadora = KK*cos(wc*portadoraX)
    
    señal_modulada=señal2*señal_portadora
    
    
    
    
    return señal_modulada,x2,señal2,señal_portadora

def graphModulationAM(señal2,señal_portadora,señal_modulada):
	plt.subplot(311)
    plt.title("Señal del Audio")
    plt.plot(x2[:800],señal2[:800],linewidth=0.4)
    plt.subplot(312)
    plt.title("Señal Portadora")
    plt.plot(portadoraX[:600],señal_portadora[:600],linewidth=0.4)
    plt.subplot(313)
    plt.title("Modulación AM "+str(K)+" %")
    plt.plot(x2[:800],señal_modulada[:800],linewidth=0.4)
    plt.show()