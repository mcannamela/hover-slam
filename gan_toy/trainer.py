import numpy as np

from gan_toy.discriminator import build_mlp_discriminator
from gan_toy.generator import build_mlp_generator


class Trainer(object):

    def __init__(self,
                 image_shape,
                 n_noise_dims,
                 n_d_units,
                 n_g_units,
                 n_hidden_d_layers,
                 n_hidden_g_layers
                 ):
        self.n_noise_dims = n_noise_dims
        self.discriminator = build_mlp_discriminator(
            image_shape,
            n_d_units,
            hidden_layers=n_hidden_d_layers
        )

        self.generator = build_mlp_generator(
            n_noise_dims,
            n_g_units,
            image_shape,
            hidden_layers=n_hidden_g_layers
        )

    def train(self, circle_generator, circle_generator_session, n_epochs=10, samples_per_epoch=10000):

        for epoch in range(n_epochs):
            real = circle_generator_session.run(circle_generator)
            fake = self.generator.predict(self.sample_noise(samples_per_epoch))
            self._update_discriminator(real, fake)
            self._update_generator(samples_per_epoch)

    def sample_noise(self, n_samples):
        return np.random.random((n_samples, self.n_noise_dims))

    def _update_discriminator(self, real, fake):
        raise NotImplementedError()

    def _update_generator(self, samples_per_epoch):
        raise NotImplementedError()







