import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.callbacks import TensorBoard
from time import time

batch_size = 32
num_classes = 10
epochs = 100
model_name = 'keras_cifar10.h5'

#Load CIFAR10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float')
X_train /= 255
X_test = X_test.astype('float')
X_test /= 255
print(X_train.shape[0] + X_test.shape[0], "images loaded...")

#convert y to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#Optimizer
rms = keras.optimizers.rmsprop(lr=.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

tb = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=10, write_images=True)

#es = EarlyStopping(monitor='val_loss', min_delta=.0001, patience=3)

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test,y_test), shuffle=True, callbacks=[tb])

model.save(model_name)

scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
