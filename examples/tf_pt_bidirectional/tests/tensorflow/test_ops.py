import tensorflow as tf


def test_relu_numerical_boundary():
    values = tf.constant([-1.0, 0.0, 2.0], dtype=tf.float32)
    actual = tf.nn.relu(values)
    expected = tf.constant([0.0, 0.0, 2.0], dtype=tf.float32)
    tf.debugging.assert_near(actual, expected)


def test_reduce_sum_numerical():
    values = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    actual = tf.math.reduce_sum(values, axis=1)
    expected = tf.constant([3.0, 7.0], dtype=tf.float32)
    tf.debugging.assert_near(actual, expected)
