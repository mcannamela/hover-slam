import unittest
import tensorflow as tf
import numpy as np

from gan_toy.true_generator import build_true_generator
import bokeh.plotting as plt

class TestTrueGenerator(tf.test.TestCase, unittest.TestCase):
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
            generator = build_true_generator(self.x, self.y, self.h, self.k, self.r, self.a)
            im = generator.eval()
            self.assertAllClose(self.exp_image, im)

    def test_random(self):
        with self.test_session():
            h = tf.random_uniform(shape=[1], minval=-.8, maxval=.8)
            k = tf.random_uniform(shape=[1], minval=-1.8, maxval=1.8)
            r = tf.random_uniform(shape=[1], minval=.12, maxval=.7)
            a = tf.random_uniform(shape=[1], minval=.5, maxval=1.5)

            generator = build_true_generator(self.x, self.y, h, k, r, a)
            ims = np.concatenate([np.concatenate([generator.eval() for i in range(10)], axis=1) for j in range(10)], axis=0)

            if True:
                # output to static HTML file
                plt.output_file("random_images.html")

                # create a new plot with a title and axis labels
                p = plt.figure(title="simple line example", x_axis_label='x', y_axis_label='y', x_range=(0, 1),
                               y_range=(0, 1))

                # add a line renderer with legend and line thickness
                # p.line(x, y, legend="Temp.", line_width=2)
                p.image([ims], 0, 0, 1, 1)

                # show the results
                plt.show(p)


