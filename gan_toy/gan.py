from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input


def build_gan(generator, discriminator, noise_dim):

    input = Input(shape=noise_dim, name="noise_input")
    sample = generator(input)
    output = discriminator(sample)

    gan = Model(inputs=[input],
                    outputs=[output],
                    name="gan")

    return gan