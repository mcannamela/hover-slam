import numpy as np

from gan_toy.discriminator import build_mlp_discriminator
from gan_toy.gan import build_gan
from gan_toy.generator import build_mlp_generator
from tensorflow.python.keras import optimizers


def weighted_mean_loss(y_true, y_pred):
    # wasserstein gan maximizes the difference between the expectation of the
    # discriminator function values
    # wrt the real and generated distributions
    # here y_pred is expected to be all ones or all negative ones
    # keras does the mean over the batch for us, so we just need to multiply
    return y_true*y_pred


class Trainer(object):

    def __init__(self,
                 image_shape,
                 n_noise_dims,
                 n_d_units,
                 n_g_units,
                 n_hidden_d_layers,
                 n_hidden_g_layers,
                 clamp_lower=-.1,
                 clamp_upper=.1,
                 n_critic=5
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

        self.gan = build_gan(self.generator, self.discriminator, self.n_noise_dims)

        self.clamp_lower = clamp_lower
        self.clamp_upper = clamp_upper
        self.n_critic = n_critic

    def train(self, circle_generator, circle_generator_session, n_epochs=10, batch_size=32, samples_per_epoch=10000):
        self._compile_models()

        for epoch in range(n_epochs):
            for i in range(samples_per_epoch//batch_size):
                for j in range(self.n_critic):
                    real = circle_generator_session.run(circle_generator)
                    fake = self.generator.predict(self.sample_noise(batch_size))
                    self._update_discriminator(real, fake)
                self._update_generator(batch_size)

    def _compile_models(self):
        self._compile_generator()
        self._compile_discriminator()
        self._compile_gan()

    def _compile_generator(self):
        gan_optimizer = optimizers.RMSprop()
        self.generator.compile(loss='mse', optimizer=gan_optimizer)
        return gan_optimizer

    def _compile_discriminator(self):
        discriminator_optimizer = None
        self.discriminator.compile(loss=weighted_mean_loss, optimizer=discriminator_optimizer)

    def _compile_gan(self):
        gan_optimizer = optimizers.RMSprop
        self.discriminator.trainable = False
        self.gan.compile(loss=weighted_mean_loss, optimizer=gan_optimizer)
        self.discriminator.trainable = True

    def sample_noise(self, n_samples):
        return np.random.random((n_samples, self.n_noise_dims))

    def _update_discriminator(self, real, fake):
        self._enforce_discriminator_lipschitz()
        self._maximize_discriminator_wrt_real(real)
        self._minimize_discriminator_wrt_to_fake(fake)

    def _enforce_discriminator_lipschitz(self):
        for layer in self.discriminator.layers:
            weights = layer.get_weights()
            clipped_weights = [np.clip(w, self.clamp_lower, self.clamp_upper) for w in weights]
            layer.set_weights(clipped_weights)

    def _minimize_discriminator_wrt_to_fake(self, fake):
        self.discriminator.train_on_batch(fake, np.ones(fake.shape[0]))

    def _maximize_discriminator_wrt_real(self, real):
        self.discriminator.train_on_batch(real, -np.ones(real.shape[0]))

    def _update_generator(self, samples_per_epoch):
        self.discriminator.trainable = False
        self.gan.train_on_batch(self.sample_noise(samples_per_epoch), -np.ones(samples_per_epoch))
        self.discriminator.trainable = True
