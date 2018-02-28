import unittest
import tensorflow as tf
import numpy as np

from gan_toy.circle_generator import build_circle_generator
import bokeh.plotting as plt
import os


class TestTrueGenerator(tf.test.TestCase, unittest.TestCase):

    _MAKE_PLOTS = False

    def setUp(self):
        tf.test.TestCase.setUp(self)

        self.exp_image = 1.2*np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])

        with self.test_session():
            self.x = tf.linspace(-1.0, 1.0, 10)
            self.y = tf.linspace(-2.0, 2.0, 11)
            self.h = tf.constant(-.11)
            self.k = tf.constant(.4)
            self.r = tf.constant(.46)
            self.a = tf.constant(1.2)

    def test_generator(self):
        with self.test_session():
            generator = build_circle_generator(self.x, self.y, self.h, self.k, self.r, self.a)
            im = generator.eval()
            self.assertAllClose(self.exp_image, im)

            if self._MAKE_PLOTS:
                plt.output_file(os.path.join(self.get_output_dir(), "single_image.html"))
                p = plt.figure(title="h={}, k={}".format(self.h, self.k), x_axis_label='x', y_axis_label='y',
                               x_range=(0, 1),
                               y_range=(0, 1))
                p.image([im], 0, 0, 1, 1)
                plt.show(p)

    def test_random(self):
        with self.test_session():
            h = tf.random_uniform(shape=[1], minval=-.8, maxval=.8)
            k = tf.random_uniform(shape=[1], minval=-1.8, maxval=1.8)
            r = tf.random_uniform(shape=[1], minval=.12, maxval=.7)
            a = tf.random_uniform(shape=[1], minval=.5, maxval=1.5)

            generator = build_circle_generator(self.x, self.y, h, k, r, a)

            if self._MAKE_PLOTS:
                ims = np.concatenate([np.concatenate([generator.eval() for i in range(10)], axis=1) for j in range(10)],
                                     axis=0)
                plt.output_file(os.path.join(self.get_output_dir(), "random_images.html"))
                p = plt.figure(title="random circles one at a time", x_axis_label='x', y_axis_label='y', x_range=(0, 1),
                               y_range=(0, 1))
                p.image([ims], 0, 0, 1, 1)
                plt.show(p)

    def test_random_batch(self):
        with self.test_session():
            shape = [64, 1, 1]
            h = tf.random_uniform(shape=shape, minval=-.8, maxval=.8)
            k = tf.random_uniform(shape=shape, minval=-1.8, maxval=1.8)
            r = tf.random_uniform(shape=shape, minval=.12, maxval=.7)
            a = tf.random_uniform(shape=shape, minval=.5, maxval=1.5)

            generator = build_circle_generator(self.x, self.y, h, k, r, a)
            im_batch = generator.eval()

            if self._MAKE_PLOTS:
                ims = np.zeros((8 * 11, 8 * 10))
                for i in range(8):
                    for j in range(8):
                        try:
                            ims[i * 11:(i + 1) * 11, j * 10:(j + 1) * 10] = im_batch[i * 8 + j]
                        except:
                            raise
                plt.output_file(os.path.join(self.get_output_dir(),"random_batch_images.html"))

                p = plt.figure(title="random batch of circles", x_axis_label='x', y_axis_label='y', x_range=(0, 1),
                               y_range=(0, 1))

                p.image([ims], 0, 0, 1, 1)

                plt.show(p)

    def get_output_dir(self):
        return os.path.join(*[os.path.pardir] * 2 + ['resources', 'bokeh_tmp'])

