import keras
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.engine import Model
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    nb_classes = 10

###########################################################################
# Encodage one-hot vector
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
###########################################################################
# Réseau
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


###########################################################################
# Historique de l'apprentissage du réseau
    Enregistrement = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=1)
###########################################################################
# Visualisation de l'enregistrement
    loss_train = Enregistrement.history['loss']
    loss_valid = Enregistrement.history['val_loss']
    metric_train = Enregistrement.history['accuracy']
    metric_valid = Enregistrement.history['val_accuracy']
    plt.plot(loss_train,"b:o", label = "loss_train")
    plt.plot(loss_valid,"r:o", label = "loss_valid")
    plt.title("Loss over training epochs")
    plt.legend()
    plt.show()
###########################################################################
# Test des performances
    loss_test = model.evaluate(x_test, y_test)
    print("\nTEST LOSS AND ACCURACY = ", loss_test)

###########################################################################
 # Cas adversarial
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    i=1 # Choix de l'image considérée
    x_tf = tf.convert_to_tensor(x_train[:i,:,:,:])
    x = x_train[:i,:,:,:]
    y = tf.convert_to_tensor(y_train[i,])
    print(y)

    with tf.GradientTape() as grad:
        grad.watch(x_tf)
        pred, = model.predict(x_tf)
        print(tf.convert_to_tensor(pred))
        loss = loss_object(y, tf.convert_to_tensor(pred))
        print(loss)

    # Gradient de la loss relativement à l'entrée.
    gradient = grad.gradient(loss, x_tf)
    print(gradient)
    signed_grad = tf.sign(gradient)
    # Choix du epsilon
    eps = 0.1
    perturbation = eps*signed_grad
