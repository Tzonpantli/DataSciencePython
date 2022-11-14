# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 01:16:56 2022
Un sistema de predicción para el diagnóstico de diabetes.
utilizar k vecinos más cercanos.
Empleamos la validación cruzada con k repeticiones (k- folds) repetidas
@author: sick_
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score    
from src.main.DbConnect.FilesConnect import datosDiabetes

def DiabetesFoldrepeated():
    print(" Programa  KnnCvKFoldsRepeated. (k- folds) repetidas ".title().center(85, "#"))
    """ Paso 1: Cargar los datos y procesarlos """
    data= datosDiabetes
    """ Definimos los atributos(features) y la variable objetivo(target) """
    X=data[["glucose","insulin","sspg"]]  # atributos
    y=data["class"]                       # respuesta
    
    """Definir el modelo. (K vecinos más cercanos con 5 vecinos) 
    n_jobs=-1 : Todos los procesadores disponibles de la PC."""
    try:
        vecinos= int(input("Introduzca el numero de vecinos(ex. 5): "))
        numProcesador= int(input("Introdusca el numero de procesadores a utlizar del pc: "))
        numSplits= int(input("Escoge el numero de splits(ex. 10): "))
        numRepeats= int(input("Escoge el numero de repeticiones(ex. 50): "))
        Kvecinos=KNeighborsClassifier(n_neighbors=vecinos,n_jobs=-numProcesador)
        
        """Medir el desempeño predictivo del modelo con validación
        cruzada  k-iteracions / k-folds repetida.
        
        Definir el tipo de validación """
        kfolds_repeated=RepeatedKFold(n_splits=numSplits,n_repeats=numRepeats)
        """calcular la metrica de desempeño en cada una de las capas de prueba"""
        scores=cross_val_score(Kvecinos,X,y,scoring="accuracy",cv=kfolds_repeated,n_jobs=-numProcesador) 
        """Promedio de exactitud del modelo"""
        print(f"Exactitud promedio del modelo: {scores.mean():.4f}")
    except Exception as e:
        print(e)