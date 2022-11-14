# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:33:22 2022
Primer codigo, lo ultimo son random choice que no esta terminado por lo cual 
no sirve para el programa
@author: sick_
"""

# Importar las librerias
import matplotlib.pyplot as plt   # Libreria de manejo de gráficos
from sklearn.neighbors import KNeighborsClassifier # Para implementar k vecinos mas cercanos
from sklearn.metrics import plot_confusion_matrix # Para hacer una maatris de confusion
from src.main.DbConnect.FilesConnect import datosEstatura


def Vecinos01():
    print(" Programa  KnnNeighbors001 ".title().center(85, "#"))
    datos= datosEstatura
    try:
        plt.title("Dispersión Estatura-Peso")
        plt.xlabel("Estatura(pulgadas)")
        plt.ylabel("Peso(libras)")
        plt.scatter(x=datos["height"], y=datos["weight"], c=datos["gender"])
        plt.savefig("C:/Users/sick_/Music/eclipsePy/src/main/files/SaveImages/Knn001/DispersionKnn001.png")
        plt.show()
        print("La grafica de dispersion se genero exitosamente \n")
    except:
        print("No se ha podido generar la grafica")
    """ Dimensión de los datos: cuántos datos y covariables tiene la base de datos.
    N=len(datos)
    N
    # Identificar los atributos/covariables y la variable objetivo/target """
    X=datos[["height", "weight"]]  # Atributos
    y=datos["gender"]            # gender
    """ Queremos construir una función f
    f(X) = y 
    Entrenar al algoritmo. Memorizar los datos. """
    neighbors= int (input("Seleccione el numero de vecinos Knn: \n"))
    Knn=KNeighborsClassifier(n_neighbors=neighbors).fit(X,y)
    # Predicciones... 
    y_pred=Knn.predict(X)
    print("\nlos datos y_pred son: \n",y_pred,"\n")
    try:
        plot_confusion_matrix(Knn,X,y)
        plt.savefig("C:/Users/sick_/Music/eclipsePy/src/main/files/SaveImages/Knn001/Confusion001.png")
        plt.show()
        print("Se ha generado la grafica de matriz de confusion exitosamente \n")
    except:
        print("No se ha podido generar la grafica de la matriz de confusion \n")
        
