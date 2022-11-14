# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 01:18:44 2022
Un sistema de predicción para el diagnóstico de diabetes. utilizar k vecinos más cercanos.
Partir el conjunto de datos, en entrenamiento y prueba, de manera automática
@author: sick_
"""
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from src.main.DbConnect.FilesConnect import datosDiabetes

def DiabetesHoldpred():
    print(" Programa  KnnCvHoldout. que utiliza train_test_split ".title().center(85, "#"))
    """ Paso 1: Cargar los datos y procesarlos """
    data= datosDiabetes
    # Definimos los atributos(features) y la variable objetivo(target)
    X=data[["glucose","insulin","sspg"]]  # atributos
    y=data["class"]                       # respuesta
    
    """Paso 1.1. Definimos los conjuntos de prueba y entrenamiento.
    Nota: train_test_split(Atributos,target, train_size= proporción de 
    los datos que vamos a emplear para entrenar, random_state=Semilla que 
    permite reproducir los resultados). En este ejemplo, hemos seleccionado
    el 80% de los datos para entrenar. """
    try: 
        trainSize= float(input("Introdusca el % del tamaño del train(ex. 0.8): "))
        vecinos= int(input("Introduzca el numero de vecinos(ex. 5): "))
        numProcesador= int(input("Introdusca el numero de procesadores a utlizar del pc: "))
        X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=trainSize)
        
        """ Paso 2: Entrenar el modelo. (K vecinos más cercanos con 5 vecinos) 
        n_jobs=-1 : Todos los procesadores disponibles de la PC."""
        Kvecinos=KNeighborsClassifier(n_neighbors=vecinos,n_jobs=-numProcesador).fit(X_train,y_train)
        
        """ Paso 3: Hacer predicciones.
        En el conjunto de entrenamiento"""
        predicciones=Kvecinos.predict(X_train)
        metrica= metrics.accuracy_score(y_train,predicciones)
        print(f"Exactitud en el conjunto de entrenamiento: {metrica:.4f}")
        
        """ En el conjunto de prueba """
        predicciones=Kvecinos.predict(X_test)
        metrica=metrics.accuracy_score(y_test,predicciones)
        print(f"Predicciones en el conjunto de prueba:{metrica:.4f} ")
    except Exception as e:
        print(e)
