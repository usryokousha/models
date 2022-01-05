import tensorflow as tf
from attention import MultiscaleAttention


class MultiscaleAttentionTest(tf.test.TestCase):
    def test_multiscale_attention(self):
        tensor = tf.keras.Input([56 ** 2, 96], batch_size=8)
        attn = MultiscaleAttention()
        x = attn(tensor, tf.TensorShape([1, 56, 56]))
        self.assertAllEqual(tensor.shape.as_list(), x[0].shape.as_list())
