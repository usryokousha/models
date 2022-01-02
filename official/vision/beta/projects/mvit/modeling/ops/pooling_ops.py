import tensorflow as tf
from tensorflow.keras import layers

class FixedPadding(layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self, kernel_size, data_format='channels_last', **kwargs):
    super(FixedPadding, self).__init__(**kwargs)
    self.kernel_size = kernel_size
    self.data_format = data_format

  def call(self, inputs, training):
    kernel_size = self.kernel_size
    data_format = self.data_format
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
      padded_inputs = tf.pad(
          inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
      padded_inputs = tf.pad(
          inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

    return padded_inputs


class Conv2D(layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               filters,
               kernel_size,
               strides,
               data_format='channels_last',
               **kwargs):
    super(Conv2D, self).__init__(**kwargs)
    if strides > 1:
      self.fixed_padding = FixedPadding(kernel_size, data_format=data_format)
    else:
      self.fixed_padding = None
    self.conv2d = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        data_format=data_format)

  def call(self, inputs, training):
    if self.fixed_padding:
      inputs = self.fixed_padding(inputs, training=training)
    return self.conv2d(inputs, training=training)