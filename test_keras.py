from keras.models import Sequential
from keras.layers import Dense
import numpy as np

model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=1))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


def build_data(n_samples):
    x = np.random.random(n_samples)
    y = np.zeros((x.shape[0], 10))
    idx = np.array(np.mod(x * 10, 10), dtype=int)
    y[np.arange(n_samples), idx] = 1
    return x, y


x_train, y_train = build_data(10000)
x_test, y_test = build_data(10000)

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=40, batch_size=32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size=128)
print(loss_and_metrics)
