from keras import models, layers
import numpy as np


def build_mlp_generator(input_shape, hidden_units, output_shape, hidden_layers=3):
    model = models.Sequential()

    model.add(layers.Dense(
        units=hidden_units,
                    activation='relu',
                    input_shape=input_shape,
                    ))
    for i in range(hidden_layers):
        model.add(layers.Dense(
            units=hidden_units,
            activation='relu',
                        ))

    model.add(layers.Dense(units=np.prod(output_shape), activation='tanh'))
    model.add(layers.Reshape(output_shape))

    return model
