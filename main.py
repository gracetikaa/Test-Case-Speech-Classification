from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
import os
import numpy as np
import pandas as pd
import soundfile as sf


def readfile(path):
    dataset = []
    label = []
    k = 1
    for folder in os.listdir(path):
        dir = os.path.join(path)
        directory = dir+"/"+folder
        for root,dirs,files in os.walk(directory):
            for i in files:
                if i.endswith(".flac"):
                    file = os.path.abspath(directory+"/"+i)
                    data, samplerate = sf.read(file)
                    dataset.append(data)
                    label.append(k)
        k+=1
    # print(dataset[0].shape)
    df = pd.DataFrame({'data':dataset,'label':label})
    return df

data = readfile('dataset')
a = data.loc[:,'data']
b = data.loc[:,'label']

(x_train, x_test, y_train, y_test) = train_test_split(a, b, test_size = 0.25, shuffle = True)
x_train = [x_train,x_train]
x_test = [x_test,x_test]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=x_train[0].shape))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
