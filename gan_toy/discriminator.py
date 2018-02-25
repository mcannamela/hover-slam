from keras import models, layers


def build_mlp_discriminator(input_shape, hidden_units, hidden_layers=3, dropout=.2):
    model = models.Sequential()

    model.add(layers.Dense(
        units=hidden_units,
                    activation='relu',
                    input_shape=input_shape,
                    ))
    model.add(layers.Dropout(dropout))

    for i in range(hidden_layers):
        model.add(layers.Dense(
            units=hidden_units,
            activation='relu',
                        ))

        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(units=1, activation='sigmoid'))

    return model


# def build_discriminator(input_shape):
#     model = models.Sequential()
#
#     model.add(layers.Conv2D(
#         input_shape=input_shape,
#         filters=32,
#         kernelsize=5,
#         strides=1
#     ))
#
#     model.add(layers.Dense(
#         units=hidden_units,
#         activation='relu',
#     ))
#
#     model.add(layers.Conv2D(
#         input_shape=input_shape,
#         filters=32,
#         kernelsize=5,
#         strides=1
#     ))
#
#
#     for i in range(hidden_layers):
#
#
#     model.add(layers.Dense(units=np.prod(output_shape), activation='sigmoid'))
#     model.add(layers.Reshape(output_shape))
#
#     return model