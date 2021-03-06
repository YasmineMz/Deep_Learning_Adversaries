# -*- coding: utf-8 -*-
##########################################################
#Ensemble des fonctions nécessaires à la suite de l'étude:
##########################################################
def adversarial_examples_linear_model(X,Y,model,epsilon):
  x_adversarial=np.zeros(X.shape)
  noise=np.sign(model.coef_)
  for i in range(X.shape[0]):
    x_adversarial[i]=X[i]+epsilon*noise[Y[i]].T
  return x_adversarial, noise
  
def adversarial_examples_linear_model2(X,Y,model,epsilon):
  x_adversarial_noise=np.zeros(X.shape)
  noise=np.sign(model.coef_)
  for x in range(X.shape[0]):
    minus=1
    if Y[x]==7:
      minus=-1
    x_adversarial_noise[x]=X[x]+minus*epsilon*noise
  return x_adversarial_noise, noise


def affichage_data(x_test,x_adversarial,nb):
  data_clean = x_test[0].reshape(28,28)
  data_adversarial = x_adversarial[0].reshape(28,28)
  for n in range(1,nb):
      image_clean= np.array(x_test[n])    
      image_adversarial= np.array(x_adversarial[n])
      image_clean = image_clean.reshape(28,28)
      image_adversarial = image_adversarial.reshape(28,28)

      data_clean = np.concatenate((data_clean, image_clean),axis=1)
      data_adversarial=np.concatenate((data_adversarial, image_adversarial),axis=1)

  plt.imshow(data_clean, cmap='gray')
  plt.show()
  plt.imshow(data_adversarial, cmap='gray')
  plt.show()
  return  

def affichage_bruit(noise):
  nb=noise.shape[0]  
  adversarial_noise = noise[0].reshape(28,28)
  for n in range(1,nb):
    image_noise= np.array(noise[n])    
    image_noise=image_noise.reshape(28,28)
    adversarial_noise=np.concatenate((adversarial_noise, image_noise),axis=1)
  plt.imshow(adversarial_noise, cmap='gray')
  plt.show()
  return

#Permet de sélectionner deux classes parmi les 10 de MNIST
def generate2ClassMNISTdata(class1,class2):
  index_train=np.concatenate((np.where(y_train==class1),np.where(y_train==class2)),axis=1)
  index_test=np.concatenate((np.where(y_test==class1),np.where(y_test==class2)),axis=1)

  X_train=x_train[index_train].reshape(x_train[index_train].shape[1],784)
  X_test=x_test[index_test].reshape(x_test[index_test].shape[1],784)
  Y_train=y_train[index_train].reshape(y_train[index_train].shape[1])
  Y_test=y_test[index_test].reshape(y_test[index_test].shape[1])
  return X_train,Y_train,X_test,Y_test

#######################
#PROGRAMME PRINCIPAL
#######################
import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd

#Premier test sur l'ensemble des données MNIST en utilisant la régression logistique directement disponible sur SKlearn

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)

logisticRegr = LogisticRegression(max_iter=100,solver = 'lbfgs')
logisticRegr.fit(x_train, y_train)
y_pred_prob=logisticRegr.predict_proba(x_test)
weight=logisticRegr.coef_
score = logisticRegr.score(x_test, y_test)
print("Score régression logistique simple sur MNIST:", score)

#Génération des adversarials examples:
######################################

x_test1=x_test.reshape(10000,784)
x_adversarial_test1, noise1=adversarial_examples_linear_model(x_test1,y_test,logisticRegr,-10)

#Score de prédiction sur les adversarials examples: 
###################################################

score = logisticRegr.score(x_adversarial_test1, y_test)
print("Score adversarial: ", score)

#Affichage data/bruit:
######################

affichage_data(x_test1,x_adversarial_test1,4)
affichage_bruit(noise1)


#On choisit de prendre seulement les classes 3 et 7 pour avoir comparé certains résultats avec ceux de l'article et pour accélerer la rapidité des calculs

print("Modèle linéaire en gardant les classes 3 et 7 MNIST: ")

X_train,Y_train, X_test, Y_test=generate2ClassMNISTdata(3,7)
logisticRegr2 = LogisticRegression(max_iter=100,solver = 'lbfgs')
logisticRegr2.fit(X_train, Y_train)
score = logisticRegr2.score(X_test, Y_test)
print("Score régression logistique simple sur MNIST:", score)

x_adversarial_test2=np.zeros((10000,784))
x_adversarial_test2, noise2=adversarial_examples_linear_model2(X_test,Y_test,logisticRegr2,5)

score2 = logisticRegr2.score(x_adversarial_test2, Y_test)
print("Score adversarial: ", score2)

#Affichage data/bruit:
######################

affichage_data(X_test,x_adversarial_test2,4)
affichage_bruit(noise2)

#Adversarial training:
######################

print("Adversarial training sur la régression logistique")

x_adversarial_train3,noise3=adversarial_examples_linear_model2(X_train,Y_train,logisticRegr2,5)
X_adversarial_train3=np.concatenate((x_adversarial_train3,X_train))
X_adversarial_test3=np.concatenate((x_adversarial_test2,X_test))
Y_adversarial_train3=np.concatenate((Y_train,Y_train))
Y_adversarial_test3=np.concatenate((Y_test,Y_test))

#On teste notre solveur entraîné des adversarial examples
logisticRegr3 = LogisticRegression(max_iter=100,solver = 'lbfgs')
logisticRegr3.fit(X_adversarial_train3, Y_adversarial_train3)
score1 = logisticRegr3.score(X_adversarial_test3, Y_adversarial_test3)
score2 = logisticRegr3.score(X_test,Y_test)

print("Score prediction toutes données confondues", score1)
print("Score prediction avec les données de base MNIST", score2,"\n\n")

#On essaye de générer d'autres adversarial examples à partir du nouveau modèle
X_adversarial_test4,noise4=adversarial_examples_linear_model2(X_adversarial_test3,Y_adversarial_test3,logisticRegr3,5)
score = logisticRegr3.score(X_adversarial_test4, Y_adversarial_test3)
print("Score adversarial sur nouveau modèle:",score)

#Affichage des exemples cleans et adversarials examples et du bruit associé
affichage_data(X_adversarial_test3,X_adversarial_test4,4)
affichage_bruit(noise4)

print("Les deux noises sont-ils égaux? ",np.array_equal(np.sign(logisticRegr2.coef_),np.sign(logisticRegr3.coef_)))
