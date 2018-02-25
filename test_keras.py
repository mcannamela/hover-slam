from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers, optimizers
import numpy as np
from bokeh import plotting as plt
import os

model = Sequential()
reg = 1e-5
model.add(Dense(units=32, activation='relu', input_shape=(1,),
                kernel_regularizer=regularizers.l2(reg),
                activity_regularizer=regularizers.l2(reg)
                ))
model.add(Dense(units=32, activation='relu',
                kernel_regularizer=regularizers.l2(reg),
                activity_regularizer=regularizers.l2(reg)
                ))
model.add(Dense(units=10, activation='softmax'))

opt = optimizers.SGD(lr=0.2, decay=1e-3, momentum=1e-5, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


def build_data(n_samples):
    x = np.random.random(n_samples)
    y = np.zeros((x.shape[0], 10))
    idx = np.array(np.mod(x * 10, 10), dtype=int)
    y[np.arange(n_samples), idx] = 1
    return x, y


x_train, y_train = build_data(10000)
x_test, y_test = build_data(10000)

bound = np.arange(9) * .1 + .1
delta_minus = (bound - x_train[:, None])
delta_plus = (x_train[:, None] - bound)
delta_plus[delta_plus < 0] = 1.0
delta_minus[delta_minus < 0] = 1.0
gaps = np.min(delta_plus, axis=0) + np.min(delta_minus, axis=0)


# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=60, batch_size=32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(loss_and_metrics)
print (gaps, 1.0-np.sum(gaps))

x_mesh = np.linspace(0, 1, 100000)
classes = np.argmax(model.predict(x_mesh, batch_size=128), axis=1)
x = np.concatenate([[0], x_mesh[np.flatnonzero(np.diff(classes) > 0)], [1]])
print(np.diff(x))

f = plt.figure()
f.segment(x, x * 0, x, x * 0 + 1)
f.circle(x, x * 0 + 1)
plt.output_file(os.path.join("resources", "bokeh_tmp", "test_keras.html"))
plt.show(f)
