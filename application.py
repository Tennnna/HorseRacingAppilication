from __future__ import print_function

import pandas as pd
from pandas import Series,DataFrame

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam



wine_data_set = pd.read_csv("winequality-red.csv",sep=";",header=0)


x = DataFrame(wine_data_set.drop("quality",axis=1))


y = DataFrame(wine_data_set["quality"])


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.05)



x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)



model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(11,)))
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu', input_shape=(11,)))
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu', input_shape=(11,)))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.summary()
print("\n")


model.compile(loss='mean_squared_error',optimizer=RMSprop(),metrics=['accuracy'])



history = model.fit(x_train, y_train,batch_size=200,epochs=1000,verbose=1,validation_data=(x_test, y_test))


score = model.evaluate(x_test,y_test,verbose=1)
print("\n")
print("Test loss:",score[0])
print("Test accuracy:",score[1])


sample = [7.9, 0.35, 0.46, 5, 0.078, 15, 37, 0.9973, 3.35, 0.86, 12.8]
print("\n")
print("sampledata")

print(sample)


sample = np.array(sample)
predict = model.predict_classes(sample.reshape(1,-1),batch_size=1,verbose=0)

print("\n")
print("Predicted value")
print(predict)
print("\n")





def plot_history(history):



    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()


plot_history(history)
