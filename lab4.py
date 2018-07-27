# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:11:37 2016

@author: Francisco
"""


from scipy.signal import butter, lfilter
from scipy import integrate
from numpy import sinc, linspace, cos, arange, convolve,sin
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
from numpy import arange, linspace
from scipy.fftpack import fft, ifft
from math import pi




#==============================================================================
# Qué hace la función?: Trabajo con los datos de audio, obtengo la cantidad de datos y determino el tiempo
# Parámetros de entrada: Matriz con los datos de la amplitud del audio
# Parámetros de salida: Vector con la señal a trabajar, el largo de la señal y un vector con los tiempos de la señal
#==============================================================================
def obtener_datos_1(info,rate):
    señal = info
    largo_señal = len(señal)#obtengo el largo de la señal
    largo_señal = float(largo_señal)
    tiempo = float(largo_señal)/float(rate)#genero el tiempo total del audio
    x = linspace(0,tiempo,largo_señal)#genero un vector de 0 hasta tiempo con intervalos del porte de la frecuencia
    return señal,largo_señal,x
    
    
    

    
    
#==============================================================================
# Qué hace la función?: Realiza la modulación de una señal en frecuencia
# Parámetros de entrada: señal, largo señal, vector del eje x de la señal, frecuencia moduladora
# Parámetros de salida: NINGUNO
#==============================================================================    
def modulacion_FM(señal,largo_señal,x,rate,K):
    
    KK = K/100
    fm=rate/2 #se requiere la mitad de la frecuencia dada por la funcion "read" para obtener la frecuencia de muestreo
    fc=5*fm #se requiere una frecuencia portadora minimo 4 veces mayor que la frecuencia de muestreo
    
    print (fc)
    wc=2*pi
    
    tiempo = float(largo_señal)/float(rate)#genero el tiempo del audio 
    #x2 = arange(0,tiempo,(1.0/float(2*fc)))#genero un vector de 0 hasta tiempo con intervalos del porte de la frecuencia
    x2 = linspace(0,tiempo,fc*tiempo)
    señal2=np.interp(x2, x,señal)
    
    
    A=1
    
    portadoraX=np.linspace(0,fc, fc*tiempo)
    
    señal_portadora = cos(wc*portadoraX)
    
    
    integral= integrate.cumtrapz(señal2,x2, initial=0)
    w= fm * x2
    señal_modulada=A*(np.cos(w*pi+KK*integral*pi))#por que se agrega un pi?
    
    plt.subplot(311)
    plt.title("Señal del Audio")
    plt.plot(x2[:200],señal2[:200],linewidth=0.4)
    plt.subplot(312)
    plt.title("Señal Portadora")
    plt.plot(portadoraX[:600],señal_portadora[:600],linewidth=0.4)
    plt.subplot(313)
    plt.title("Modulación FM "+str(K)+" %")
    plt.plot(x2[:200],señal_modulada[:200],linewidth=0.4)
    plt.show()
    return señal_modulada,x2
    
    
    
#==============================================================================
# Qué hace la función?: Realiza la modulación de una señal en amplitud
# Parámetros de entrada: señal, largo señal, vector del eje x de la señal, frecuencia moduladora
# Parámetros de salida: NINGUNO
#==============================================================================    
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
    print("len(x2)/rate: ",len(x2)/rate)
    print ("len(x2): ", len(x2))
    señal_portadora = KK*cos(wc*portadoraX)
    
    señal_modulada=señal2*señal_portadora
    
    
    
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
    return señal_modulada,x2



#==============================================================================
# Qué hace la función?: Genera la transformada de fourier de una señal dada
# Parámetros de entrada: Vector con la señal a transformar y el largo de la señal
# Parámetros de salida: Vector con la amplitud transformada y un vector con las frecuencias
#==============================================================================
def obtener_transformada_fourier(señal,largo_señal):
    transFour = fft(señal,largo_señal)#eje Y
    transFourN = transFour/largo_señal#eje y normalizado
    
    aux = linspace(0.0,1.0,largo_señal/2+1)#obtengo las frecuencias
    xfourier = rate/2*aux#genero las frecuencias dentro del espectro real
    yfourier = transFourN[0.0:largo_señal/2+1]#genero la parte necesaria para graficar de la transformada
    return xfourier,yfourier
    
#==============================================================================
# Qué hace la función?: Grafica la transformada de fourier de una función
# Parámetros de entrada: -vector de amplitudes (EJE Y) y vector con frecuencias (EJE X)
# Parámetros de salida: NINGUNO
#==============================================================================    
def graficar_transformada_1(xfourier,yfourier,tipo,por):
    plt.plot(xfourier,abs(yfourier),linewidth=0.4)
    plt.title("Espectro frecuencia "+tipo+" al "+str(por)+" %")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.show()


#==============================================================================
# Qué hace la función?: Genera la transformada de fourier de una señal dada
# Parámetros de entrada: Vector con la señal a transformar y el largo de la señal
# Parámetros de salida: Vector con la amplitud transformada y un vector con las frecuencias
#==============================================================================   
def graficar_transformada_2(xfourier,yfourier):
    plt.plot(xfourier,abs(yfourier),linewidth=0.4)
    plt.title("Espectro frecuencia señal original")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.show()
    

#==============================================================================
# Qué hace la función?: desmodula una señal por amplitud
# Parámetros de entrada: 
# Parámetros de salida: señal demodulada
#==============================================================================    
def desmodulador_AM(señal,largo_señal,x,rate):
    fc=200000 #se requiere una frecuencia portadora minimo 4 veces mayor que la frecuencia de muestreo
    wc=2*pi*fc
    
    portadoraX=np.linspace(0,len(x)/(rate), num=len(x))
    
    señal_portadora = cos(wc*portadoraX)
    
    señal_desmodulada=señal/señal_portadora
    
    tiempo = float(len(señal_desmodulada))/float(rate*24)#genero el tiempo del audio
    x2 = linspace(0,tiempo,rate*24*tiempo)
     
    return x2,señal_desmodulada/2
    

#==============================================================================
# Qué hace la función?: Grafica el la amplitud del audio con respecto al tiempo
# Parámetros de entrada: Vector con la señal (EJE Y) y vector con el tiempo (EJE X)
# Parámetros de salida: NINGUNO
#==============================================================================   
def graficar_audio_respecto_tiempo(señal,x,tipo):
    plt.plot(x,señal,linewidth=0.4)
    plt.title("Audio con respecto al tiempo "+tipo)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")
    plt.xlim(0,9.5)
    plt.show()
    
    
#==============================================================================
# Qué hace la función?: Grafica el la amplitud del audio con respecto al tiempo
# Parámetros de entrada: Vector con la señal (EJE Y) y vector con el tiempo (EJE X)
# Parámetros de salida: NINGUNO
#==============================================================================   
def graficar_audio_respecto_tiempo2(señal,x,tipo,porcentaje):
    plt.plot(x,señal,linewidth=0.4)
    plt.title("Audio con respecto al tiempo "+tipo+" "+porcentaje +"%")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")
    plt.xlim(0,9.5)
    plt.show()



#==============================================================================
# inicio del codigo a ejecutar
# PARA GENERAR LOS GRAFICOS ES NECESARIO DESCOMENTAR LINEAS DE CODIGO!!!!!
#==============================================================================

# PUNTO 1, IMPORTAR LA SEÑAL DE AUDIO
rate,info=read("handel.wav")
#rate = frecuencia de muestreo: numero de veces por segundo en que cambia el nivel de una señal digital
#       tantas muestras por segundo

print ()


# obtener datos:
señal,largo_señal,x = obtener_datos_1(info,rate)

sFM15,xFM15 = modulacion_FM(señal,largo_señal,x,rate,15)
sAM15,xAM15 = modulacion_AM(señal,largo_señal,x,rate,15)
'''sFM100,xFM100 = modulacion_FM(señal,largo_señal,x,rate,100)
sAM100,xAM100 = modulacion_AM(señal,largo_señal,x,rate,100)
sFM125,xFM125 = modulacion_FM(señal,largo_señal,x,rate,125)
sAM125,xAM125 = modulacion_AM(señal,largo_señal,x,rate,125)

señal2, x2 = obtener_transformada_fourier(señal,len(señal))
xFour_FM15, yFour_FM15 = obtener_transformada_fourier(sFM15,len(sFM15))
xFour_AM15, yFour_AM15 = obtener_transformada_fourier(sAM15,len(sAM15))
xFour_FM100, yFour_FM100 = obtener_transformada_fourier(sFM100,len(sFM100))
xFour_AM100, yFour_AM100 = obtener_transformada_fourier(sAM100,len(sAM100))
xFour_FM125, yFour_FM125 = obtener_transformada_fourier(sFM125,len(sFM125))
xFour_AM125, yFour_AM125 = obtener_transformada_fourier(sAM125,len(sAM125))



graficar_transformada_2(señal2,x2)
graficar_transformada_1(xFour_FM15,yFour_FM15,"FM",15)
graficar_transformada_1(xFour_AM15,yFour_AM15,"AM",15)
graficar_transformada_1(xFour_FM100,yFour_FM100,"FM",100)
graficar_transformada_1(xFour_AM100,yFour_AM100,"AM",100)
graficar_transformada_1(xFour_FM125,yFour_FM125,"FM",125)
graficar_transformada_1(xFour_AM125,yFour_AM125,"AM",125)

x3,señal3 = desmodulador_AM(sAM15,len(sAM15),xAM15,rate)
x4,señal4 = desmodulador_AM(sAM100,len(sAM100),xAM100,rate)
x5,señal5 = desmodulador_AM(sAM125,len(sAM125),xAM125,rate)

graficar_audio_respecto_tiempo(señal,x,"sin modular")
graficar_audio_respecto_tiempo2(señal3,x3,"desmodulado","15")
graficar_audio_respecto_tiempo2(señal4,x4,"desmodulado","100")
graficar_audio_respecto_tiempo2(señal5,x5,"desmodulado","125")'''








