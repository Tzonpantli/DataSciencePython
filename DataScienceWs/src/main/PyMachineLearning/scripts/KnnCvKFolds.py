# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 01:18:13 2022

Un sistema de predicción para el diagnóstico de diabetes.
utilizar k vecinos más cercanos. Empleamos la validación cruzada con k repeticiones (k- folds)

@author: sick_
"""
from src.main.DbConnect.FilesConnect import datosDiabetes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
"""Para calcular la metrica de desempeño en cada iteración/capas
Las dos últimas librerias se pueden incorporar en una sola:
from sklearn.model_selection import KFold, cross_val_score    """

def DiabetesKFold():
    print(" Programa  KnnCvKFolds. (la validación cruzada con k- folds) ".title().center(85, "#"))
    """Paso 1: Cargar los datos y procesarlos"""
    data= datosDiabetes
    """ Definimos los atributos(features) y la variable objetivo(target) """
    X=data[["glucose","insulin","sspg"]]  # atributos
    y=data["class"]                       # respuesta
    """"Definir el modelo. (K vecinos más cercanos con 5 vecinos) 
    n_jobs=-1 : Todos los procesadores disponibles de la PC. """
    try: 
        vecinos= int(input("Introduzca el numero de vecinos(ex. 5): "))
        numProcesador= int(input("Introdusca el numero de procesadores a utlizar del pc: "))
        numSplits= int(input("Escoge el numero de de splits(ex. 10): "))
        Kvecinos=KNeighborsClassifier(n_neighbors=vecinos,n_jobs=-numProcesador)
        
        """Medir el desempeño predictivo del modelo con validación
        cruzada  k-repetida / k-folds.
        
        Definir el tipo de validación"""
        kfolds=KFold(n_splits=numSplits,shuffle=True)
        """ calcular la metrica de desempeño en cada una de las capas de prueba """
        scores=cross_val_score(Kvecinos,X,y,scoring="accuracy",cv=kfolds,n_jobs=-numProcesador) 
        """ Promedio de exactitud del modelo """
        print(f"Exactitud promedio del modelo: {scores.mean():.4f}")
    except Exception as e:
        print(e)