# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:28:43 2022

@author: sick_
"""
from src.main.PyMachineLearning.scripts.KnnNeighbors001 import Vecinos01
from src.main.PyMachineLearning.scripts.KnnNeighbors002 import EntrenamientoPrueba
from src.main.PyMachineLearning.scripts.KnnCvHoldout import DiabetesHoldpred
from src.main.PyMachineLearning.scripts.KnnCvKFolds import DiabetesKFold
from src.main.PyMachineLearning.scripts.KnnCvKFoldsRepeated import DiabetesFoldrepeated

def mainmenu1():
    print("")
    print(" Bienvenido al menu ".title().center(85, "#"))
    print(" Escoge el programa que quieras usar ".title().center(85, "#"))
    menu=[["1. Programa KnnNeighbors001 Dispersion "],["2. Programa KnnNeighbors002 Train and Test"], \
          ["3. KnnCvHoldout (train_test_split)"],["4. KnnCvKFolds (validación cruzada)"],\
          ["5. KnnCvFoldsRepeated (k- folds) repetidas"],["6. Cerrar agenda"]]
    for i in range(5):
        print(menu[i])
        print("")
    opcion=int(input("Introduzca la opción: "))
    print("")
    if opcion==1:
        Vecinos01()
    elif opcion==2:   
        EntrenamientoPrueba()
    elif opcion==3:
        DiabetesHoldpred()
    elif opcion==4:
        DiabetesKFold()
    elif opcion==5:   
        DiabetesFoldrepeated()
    else :
        exit()
    mainmenu1()

