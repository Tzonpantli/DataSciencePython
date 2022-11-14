# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 23:58:49 2022

@author: sick_
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import plot_confusion_matrix 
from sklearn import metrics     
from src.main.DbConnect.FilesConnect import datosEstatura

""" Global variables"""
try:
    datos= datosEstatura
except Exception as e :
    print(e,"Algo ha ocurrido al abrir el archivo")
"""
def vecinos002():
    print(" Programa  KnnNeighbors001 ".title().center(85, "#"))
    try:
        plt.title("Dispersión Estatura-Peso")
        plt.xlabel("Estatura(pulgadas)")
        plt.ylabel("Peso(libras)")
        plt.scatter(x=datos["height"], y=datos["weight"], c=datos["gender"])
        plt.savefig("C:/Users/sick_/Music/eclipsePy/src/main/files/SaveImages/Knn002/DispersionKnn002.png")
        plt.show()
        print("La grafica de dispersion se genero exitosamente \n")
    except:
        print("No se ha podido generar la grafica de la matriz de confusion \n")
        
    X=datos[["height", "weight"]]  # Atributos
    y=datos["gender"]            # gender
    try:
        neighbors= int (input("Seleccione el numero de vecinos Knn: \n"))
        Knn=KNeighborsClassifier(n_neighbors=neighbors).fit(X,y)
        # Predicciones... 
        y_pred=Knn.predict(X)
        print("\nlos datos y_pred son: \n",y_pred,"\n")
        plot_confusion_matrix(Knn,X,y)
        plt.title("Matriz de confusion")
        plt.savefig("C:/Users/sick_/Music/eclipsePy/src/main/files/SaveImages/Knn002/ConfusionKnn002.png")
        plt.show()
        print("Se ha generado la grafica de matriz de confusion exitosamente \n")
    except:
        print("No se ha podido generar la grafica de la matriz de confusion, los vecinos deben ser nuemeros enteros \n")
"""
def EntrenamientoPrueba():
    print(" Programa  KnnNeighbors002 de train y test ".title().center(85, "#"))
    """ Dividir al conjunto de datos en datos de entrenamiento y datos de prueba
    Usualmente seleccionamos entre el 70% - 90%
    Seleccionamos una muestra aleatoria de los 70 de los registros que tenemos """
    try:
        semilla= int (input("Seleccione la semilla porfavor(ex. 1234): "))
        np.random.seed(semilla)
        entrenar= float(input("Seleccione el % de datos para entrenar (expamle: 0.3): "))
        test= float(input("Seleccione el % de datos para Test (expamle: 0.7): "))
        sorteo=np.random.choice(2,70,p=[entrenar,test]) # 1: seleccionamos el registro para entrenar 
                                                   # y 0: para conjunto de prueba
        print("-----------------------------------------------------------------------")
        print("El resultado del sorteo es: ",sorteo)
        print("la suma del sorteo es: ",sum(sorteo)) 
        # Construimos ahora el conjunto de datos de entrenamiento y de prueba 
        data_train=datos[sorteo==1]
        data_test=datos[sorteo==0]
        # Seleccionamos los atributos y la variable objetivo del conjunto de entrenamiento
        X_train=data_train[["height", "weight"]]  # Atributos
        y_train=data_train["gender"]    # Variable objetivo / target
        # Seleccionamos los atributos y la variable objetivo del conjunto de prueba
        X_test=data_test[["height", "weight"]]  # Atributos
        y_test=data_test["gender"]  #Variable objetivo/ target
        # Vamos a entrenar al algoritmo, en este caso el Knn con k=3
        Kvecinos=KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train)
        # Veamos qué tan bien predice el modelo en el conjunto de entrenamiento
        predicciones=Kvecinos.predict(X_train)
        plot_confusion_matrix(Kvecinos,X_train,y_train)
        print("-----------------------------------------------------------------------")
        print("\nSe ha creado la grafica dematriz de confusion de prueba Train")
        plt.title("Matriz de confusion entrenada Train")
        plt.savefig("C:/Users/sick_/Music/eclipsePy/src/main/files/SaveImages/Knn002/ConfusionEntrenamientoKnn002.png")
        plt.show()
        accurencyTrain= metrics.accuracy_score(y_train,predicciones)
        print("EL accurency score Train es: ",accurencyTrain)
        """ Medición optimista del desempeño del algoritmo k vecinos más cercanos. (88.5% 
         de exactitud, el procentaje de los casos que predice correctamente)
         La matriz de confusión me ayuda a identificar que casos le cuesta más trabajo
         al método clasificar. Dicho de otro modo, en dónde se equivoca más el
         algoritmo. Le cuesta más trabajo predecir a las mujeres.
         Para tener una medida real del desempeño predictivo del algortimo, necesitamos
         probar al algortimo en datos que no ha "visto" (conjunto de prueba) """
        predicciones=Kvecinos.predict(X_test)
        plot_confusion_matrix(Kvecinos,X_test,y_test)
        plt.title("Matriz de confusion entrenada Test")
        print("-----------------------------------------------------------------------")
        print("\nSe ha creado la grafica dematriz de confusion de prueba Test")
        plt.savefig("C:/Users/sick_/Music/eclipsePy/src/main/files/SaveImages/Knn002/ConfusionEntreTestKnn002.png")
        plt.show()
        accurancyTest= metrics.accuracy_score(y_test,predicciones)
        print("EL accurency score Test es: ",accurancyTest,"\n")
    except Exception as e:
        print(e)











