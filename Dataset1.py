import keras
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    train_data, test_data = cifar10.load_data()
    nb_classes = 10
    x_train = tf.convert_to_tensor(train_data[0])
    y_train = tf.convert_to_tensor(train_data[1])
    x_test = tf.convert_to_tensor(test_data[0])
    y_test = tf.convert_to_tensor(test_data[1])

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
    Enregistrement = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=30)
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

###########################################################################
# Initialisation
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    i=6 # Choix de l'image considérée

    # Entrée
    x = test_data[0][i:i+1,:,:,:]
    # Conversion en tenseur
    x_t = tf.convert_to_tensor(x, dtype=tf.float32)

    # Sortie
    y_temp = test_data[1][i,]
    y = np.zeros(10)
    y[y_temp[0]]=1
    # Conversion en tenseur
    y_t = tf.convert_to_tensor(y, dtype=tf.float32)

    #print(x_t)
    #print(y_t)

###########################################################################
# Calcul de la perturbation

    with tf.GradientTape() as grad:
        grad.watch(x_t)
        pred, = model(x_t)
        # print(tf.convert_to_tensor(pred))
        loss = loss_object(y_t, tf.convert_to_tensor(pred))
        # print(loss)
    
    # Gradient de la loss relativement à l'entrée.
    g = grad.gradient(loss,x_t)
    # print(g)
    signed_grad = tf.sign(g)
    
    # Choix du epsilon
    eps = 0.4
    perturbation = eps*signed_grad
    # print(perturbation)

###########################################################################
# Calcul de l'adversarial example

    adv = x_t + perturbation
    # print(adv)

    # Prédiction obtenue à l'aide du modèle
    pred_adv = model(adv)

###########################################################################
# Traitement des résultats
    
    classe = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Label de l'image normale
    label = classe[np.argmax(y_t)]
    print("Label de l'image normale: ",label)

    # Label prédit de l'image normale
    label_pred = classe[np.argmax(pred)]
    print("Label prédit de l'image normale: ", label_pred)
    confiance = np.amax(pred)*100
    print("Confiance: ", confiance)

    # Label de l'image perturbée
    label_adv = classe[np.argmax(pred_adv)]
    print("Label de l'image perturbée: ", label_adv)
    confiance_adv = np.amax(pred_adv) * 100
    print("Confiance: ", confiance_adv)

###########################################################################
# Graphique

plt.imshow(x[0,:,:,:])
plt.show()
adv_int = tf.cast(adv,tf.int32)
plt.imshow(adv_int[0,:,:,:])
plt.show()
plt.imshow(perturbation[0,:,:,:])
plt.show()
    

