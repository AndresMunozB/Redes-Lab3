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
from scipy.fftpack import fft, ifft, fftshift
import warnings 
from scipy import integrate
from math import pi
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
	senal = info
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

def getData(nameFile):
	rate_signal,y_signal=read(nameFile)
	len_signal = len(y_signal)
	time_signal =  len_signal /float(rate_signal)
	x_signal = linspace(0, time_signal, len_signal)
	return y_signal, x_signal, rate_signal, time_signal, len_signal


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
	#savefig(figura)
	plt.show()

#=============================================================================
# Función: Función que se encarga de obtener la transformada de Fourier de los
# datos de la señal.
# Parámetros de entrada: Un arreglo con los datos de la señal, y el largo de 
# este arreglo.
# Parámetros de salida: Dos arreglos, uno con los valores del eje x y otro con
# los valores del eje y.
#=============================================================================
def fourierTransformation(senal, len_signal,rate):
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

   


def graphModulationFM(t2,señal_interpolada,portadoraX,señal_portadora,señal_modulada,K):
	plt.subplot(311)
	plt.title("Señal del Audio")
	plt.plot(t2[:200],señal_interpolada[:200],linewidth=0.4)
	plt.subplot(312)
	plt.title("Señal Portadora")
	plt.plot(portadoraX[:200],señal_portadora[:200],linewidth=0.4)
	plt.subplot(313)
	plt.title("Modulación FM "+str(K)+" %")
	plt.plot(t2[:200],señal_modulada[:200],linewidth=0.4)
	plt.show()

    
    
    
#==============================================================================
# Qué hace la función?: Realiza la modulación de una señal en frecuencia
# Parámetros de entrada: señal, largo señal, vector del eje x de la señal, frecuencia moduladora
# Parámetros de salida: NINGUNO
#==============================================================================    
def modulationFM(y_signal,x_signal,len_signal,time_signal,rate_signal,K):
    
	KK = K/100
	fm=rate_signal/2 #se requiere la mitad de la frecuencia dada por la funcion "read" para obtener la frecuencia de muestreo
	fc=30000 #se requiere una frecuencia portadora minimo 4 veces mayor que la frecuencia de muestreo
	wc=2*pi
	rate_signal_2 = fc * time_signal
	x_signal_2 = linspace(0,time_signal,fc*time_signal)
	y_signal_2 = np.interp(x_signal_2, x_signal,y_signal)

	A=1

	x_portadora = linspace(0,fc,fc*time_signal)
	y_portadora = np.cos(wc * x_portadora)
	
	integral= integrate.cumtrapz(y_signal_2,x_signal_2, initial=0)
	w= fm * x_signal_2
	y_modulada= A*(np.cos(w*pi+KK*integral*pi))
	return x_signal_2,y_signal_2,x_portadora,y_portadora,y_modulada, rate_signal_2
#==============================================================================
# Qué hace la función?: Realiza la modulación de una señal en amplitud
# Parámetros de entrada: señal, largo señal, vector del eje x de la señal, frecuencia moduladora
# Parámetros de salida: NINGUNO
#==============================================================================    
def modulacionAM(y_signal,x_signal,len_signal,time_signal,K):
	KK = K/100
	fc=30000 # Frecuencia en la que se modulará la señal
	rate_signal_2 = len_signal*10/time_signal
	x_signal_2 = linspace(0,time_signal,len_signal*10)
	y_signal_2 = np.interp(x_signal_2, x_signal,y_signal)  
	

	wc=2*np.pi*fc
	x_portadora =x_signal_2
	y_portadora = KK*np.cos(wc * x_portadora)
	y_modulada =y_signal_2*y_portadora

	return x_signal_2,y_signal_2,x_portadora,y_portadora,y_modulada, rate_signal_2

def graphModulationAM(x2,señal2,portadoraX,señal_portadora,señal_modulada,K):
	plt.subplot(311)
	plt.title("Señal del Audio",fontsize = 10)
	plt.plot(x2[:600],señal2[:600],linewidth=0.4)
	plt.subplot(312)
	plt.title("Señal Portadora",fontsize = 10)
	plt.plot(portadoraX[:600],señal_portadora[:600],linewidth=0.4)
	plt.subplot(313)
	plt.title("Modulación AM "+str(K)+" %",fontsize = 10)
	plt.plot(x2[:600],señal_modulada[:600],linewidth=0.4)
	plt.show()

def appLowFilter(senal, rate):
    nyquist = rate/2
    lowcut = 10000
    lowcut2 = lowcut/nyquist
    numtaps = 10001
    filteredLow = signal.firwin(numtaps,cutoff = lowcut2, window = 'hamming' )
    filtered = lfilter(filteredLow,1.0,senal)
    len_signal = len(senal)#obtengo el largo de la señal
    len_signal = float(len_signal)
    time = float(len_signal)/float(rate)#genero el tiempo total del audio
    x = arange(0,time,1.0/float(rate))
    return x,filtered

def demoduladorAM(x_modulada,y_modulada,largo_señal_modulada,rate_signal_2):
    #y_demodulada=y_modulada/y_portadora
    #x_demodulada = arange(0,len(y_modulada)/rate_signal_2,1.0/rate_signal_2)
    #x_demodulada, y_demodulada = appLowFilter(y_demodulada,rate_signal_2)
    #return x_demodulada, y_demodulada
    fc=30000 #se requiere una frecuencia portadora minimo 4 veces mayor que la frecuencia de muestreo
    wc=2*pi*fc
    x_demodulada=np.linspace(0,len(x_modulada)/(rate_signal_2), num=len(x_modulada))
    y_portadora = np.cos(wc*x_demodulada)
    y_desmodulada=y_modulada/y_portadora 
    return x_demodulada,y_desmodulada

#Obtencion de datos del audio
y_signal, x_signal, rate_signal, time_signal, len_signal = getData("handel.wav")

#Modulacion AM

x_signal_2,y_signal_2,x_portadora,y_portadora,y_modulada, rate_signal_2 = modulacionAM(y_signal,x_signal,len_signal,time_signal,15)
#graphModulationAM(x_signal_2,y_signal_2,x_portadora,y_portadora,y_modulada,15)

x_fourier, y_fourier = fourierTransformation(y_signal_2, len(y_signal_2),rate_signal_2)
graphTransformation(x_fourier,y_fourier,"Transformada Original","figura")

x_fourier, y_fourier = fourierTransformation(y_modulada, len(y_modulada),rate_signal_2)
graphTransformation(x_fourier,y_fourier,"Transformada Modulación AM","figura")

""""
x_signal_2,y_signal_2,x_portadora,y_portadora,y_modulada, rate_signal_2 = modulationFM(y_signal,x_signal,len_signal,time_signal,rate_signal,100)
graphModulationAM(x_signal_2,y_signal_2,x_portadora,y_portadora,y_modulada,100)

x_fourier, y_fourier = fourierTransformation(y_signal_2, len(y_signal_2),rate_signal_2)
graphTransformation(x_fourier,y_fourier,"real","figura")

x_fourier, y_fourier = fourierTransformation(y_modulada, len(y_modulada),rate_signal_2)
graphTransformation(x_fourier,y_fourier,"real","figura")

"""
x_demodulada,y_demodulada = demoduladorAM(x_signal_2,y_modulada,len(y_modulada),rate_signal_2)
print(len(x_demodulada))
print(len(y_demodulada))
x_fourier, y_fourier = fourierTransformation(y_demodulada, len(y_demodulada),rate_signal_2)
graphTransformation(x_fourier,y_fourier,"Transformada Demodulada AM","figura")
graphTime(x_demodulada,y_demodulada,"Audio Demodulacion","hola")
graphTime(x_signal,y_signal,"Audio Original","hola")





"""t2,señal2,portadoraX,señal_portadora,señal_modulada = modulationFM(señal,largo_señal,t,rate,100)
graphModulationFM(t2,señal2,portadoraX,señal_portadora,señal_modulada,100)

xfourier, fourierNorm = fourierTransformation(señal2, len(señal2),20000*4)
graphTransformation(xfourier,fourierNorm,"real","figura")

xfourier, fourierNorm = fourierTransformation(señal_modulada, len(señal_modulada),20000*4)
graphTransformation(xfourier,fourierNorm,"modulada","figura")"""