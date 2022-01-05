import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend

from typing import Callable, Optional, Tuple, List

from official.vision.beta.modeling.layers.nn_layers import StochasticDepth


class RelativePositionEmbedding(layers.Layer):
    def __init__(self, initializer: object, **kwargs) -> None:
        super().__init__(**kwargs)
        self._initializer = initializer

    def build(self, input_shape):
        batch, time, height, width, channels = input_shape

        self.pos_height = self.add_weight(
            name="height_embedding",
            shape=[1, height, channels],
            initializer=self._initializer
        )
        self.pos_width = self.add_weight(
            name="width_embedding",
            shape=[1, width, channels],
            initializer=self._initializer
        )
        self.pos_temporal = self.add_weight(
            name="temporal_embedding",
            shape=[1, time, channels],
            initializer=self._initializer
        )

        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        batch, time, height, width, channels = inputs.shape
        length = time * height * width
        height_embed = tf.tile(tf.repeat(
            self.pos_height,
            width,
            axis=1
        ), [1, time, 1])
        width_embed = tf.tile(
            self.pos_width,
            [1, height * time, 1]
        )
        temporal_embed = tf.repeat(
            self.pos_temporal,
            height * width,
            axis=1
        )

        pos_embed = temporal_embed + height_embed + width_embed

        return tf.matmul(tf.reshape(inputs, [-1, length, channels]),
                         tf.transpose(pos_embed, perm=[0, 2, 1]))

    def get_input_shape_at(self, node_index):
        return super().get_input_shape_at(node_index)

    @property
    def patch_embed_shape(self):
        return self._patch_embed_shape


def _attention_pooling(
        tensor: tf.Tensor,
        pooling: Callable,
        patch_shape: tf.TensorShape,
        norm: Optional[Callable] = None
):
    if pooling is None:
        return tensor, patch_shape

    input_rank = tensor.shape.rank
    if input_rank == 3:
        tensor = tf.expand_dims(tensor, axis=1)

    batch, num_heads, length, channels = tensor.shape
    frames, height, width = patch_shape

    tensor = tf.reshape(tensor,
                        [batch * num_heads, frames, height, width, channels])

    if backend.image_data_format() == 'channels_first':
        tensor = tf.transpose(tensor, perm=[0, 4, 1, 2, 3])

    tensor = pooling(tensor)

    if backend.image_data_format() == 'channels_first':
        output_patch_shape = tensor.shape[2:]
        length_pooled = np.prod(output_patch_shape)
        tensor = tensor.reshape(tensor,
                                [batch, num_heads, channels, length_pooled])
        tensor = tf.transpose(tensor, perm=[0, 1, 3, 2])
    else:
        output_patch_shape = tensor.shape[1:-2]
        length_pooled = np.prod(output_patch_shape)
        tensor = tensor.reshape(tensor,
                                [batch, num_heads, length_pooled, channels])

    if norm is not None:
        tensor = norm(tensor)

    if input_rank == 3:
        tensor = tf.squeeze(tensor, axis=1)

    return tensor, output_patch_shape


class MultiscaleAttention(layers.Layer):
    def __init__(
            self,
            num_heads: int = 8,
            qkv_bias: bool = False,
            dropout_rate: float = 0.0,
            kernel_q: Tuple[int] = (1, 1, 1),
            kernel_kv: Tuple[int] = (1, 1, 1),
            stride_q: Tuple[int] = (1, 1, 1),
            stride_kv: Tuple[int] = (1, 1, 1),
            pool_mode: str = 'conv',
            embed_initializer="zeros",
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            **kwargs):
        super().__init__(**kwargs)
        assert pool_mode in ["conv", "avg", "max"], "Selection is not valid."
        self._pool_mode = pool_mode
        self._kernel_q = kernel_q
        self._kernel_kv = kernel_kv
        self._stride_q = stride_q
        self._stride_kv = stride_kv
        self._qkv_bias = qkv_bias
        self._dropout_rate = dropout_rate
        self._num_heads = num_heads
        self._embed_initializer = tf.keras.initializers.get(
            embed_initializer)
        self._kernel_initializer = tf.keras.initializers.get(
            kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(
            kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        _, _, embed_dim = input_shape

        common_kwargs = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)

        head_dim = input_shape[-1] // self._num_heads
        self._scale = head_dim ** -0.5

        self.proj_q = layers.Dense(
            embed_dim, use_bias=self._qkv_bias, **common_kwargs)
        self.proj_k = layers.Dense(
            embed_dim, use_bias=self._qkv_bias, **common_kwargs)
        self.proj_v = layers.Dense(
            embed_dim, use_bias=self._qkv_bias, **common_kwargs)
        self.proj_output = layers.Dense(
            embed_dim, use_bias=True, **common_kwargs)

        if self._dropout_rate > 0.0:
            self.proj_drop = layers.Dropout(self._dropout_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if (
                self._kernel_q is not None
                and np.prod(self._kernel_q) == 1
                and np.prod(self._stride_q) == 1
        ):
            self._kernel_q = None
        if (
                self._kernel_kv is not None
                and np.prod(self._kernel_kv) == 1
                and np.prod(self._stride_kv) == 1
        ):
            self._kernel_kv = None

        if self._pool_mode in ["avg", "max"]:
            pool_op = layers.MaxPooling3D if self._pool_mode == "max" else layers.AveragePooling3D
            self.pool_q = pool_op(self._kernel_q,
                                  strides=self._stride_q,
                                  padding='VALID') \
                if self._kernel_q is not None \
                else None
            self.pool_k = pool_op(self._kernel_kv,
                                  self._stride_kv,
                                  padding='VALID') \
                if self._kernel_kv is not None \
                else None
            self.pool_v = pool_op(self._kernel_kv,
                                  strides=self._stride_kv,
                                  padding='VALID') \
                if self._kernel_kv is not None \
                else None
        elif self._pool_mode == 'conv':
            self.pool_q = (
                layers.Conv3D(
                    head_dim,
                    kernel_size=self._kernel_q,
                    strides=self._stride_q,
                    padding='VALID',
                    groups=head_dim,
                    use_bias=False,
                    **common_kwargs
                )
                if self._kernel_q is not None
                else None
            )

            self.pool_k = (
                layers.Conv3D(
                    head_dim,
                    kernel_size=self._kernel_kv,
                    strides=self._stride_kv,
                    padding='VALID',
                    groups=head_dim,
                    use_bias=False,
                    **common_kwargs
                )
                if self._kernel_kv is not None
                else None
            )

            self.pool_v = (
                layers.Conv3D(
                    head_dim,
                    kernel_size=self._kernel_kv,
                    strides=self._stride_kv,
                    padding='VALID',
                    groups=head_dim,
                    use_bias=False,
                    **common_kwargs
                )
                if self._kernel_kv is not None
                else None
            )

        else:
            raise NotImplementedError(f"Unsupported model {self._pool_mode}")

        self.pos_embed = RelativePositionEmbedding(
            initializer=self._embed_initializer
        )
        return super().build(input_shape)

    def call(self, inputs, patch_shape, *args, **kwargs):
        _, _, channels = inputs.shape

        q = self.proj_q(inputs)
        k = self.proj_k(inputs)
        v = self.proj_v(inputs)

        q_pool, q_shape = _attention_pooling(q, self.pool_q, patch_shape)
        k_pool, _ = _attention_pooling(k, self.pool_k, patch_shape)
        v_pool, _ = _attention_pooling(v, self.pool_v, patch_shape)

        # Reshape is a hack to pass the patch shape information
        q_pos = tf.reshape(q_pool, [-1, *patch_shape, channels])
        pos_rel_embed = self.pos_embed(q_pos)
        attn = tf.matmul(q_pool, tf.transpose(k_pool, [0, 2, 1]))
        attn = (attn + pos_rel_embed) * self._scale
        attn = tf.nn.softmax(attn, axis=-1)

        x = tf.matmul(attn, v_pool)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x, [-1, q_pool.shape[1], channels])
        x = x + q_pool

        x = self.proj_output(x)

        if self._dropout_rate > 0.0:
            x = self.proj_drop(x)

        return x, q_shape


class MLP(layers.Layer):
    def __init__(
            self,
            hidden_dim: Optional[int] = None,
            output_dim: Optional[int] = None,
            activation_layer: Callable = tf.nn.gelu,
            dropout_rate: float = 0.0,
            kernel_initializer: object = 'zeros',
            bias_initializer: object = 'zeros',
            kernel_regularizer: Optional[object] = None,
            bias_regularizer: Optional[object] = None,
            **kwargs):
        super().__init__(**kwargs)

        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._activation = activation_layer
        self._dropout_rate = dropout_rate
        self._kernel_initializer = tf.keras.initializers.get(
            kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regulazer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        common_kwargs = {
            'kernel_initializer': self._kernel_initializer,
            'bias_initializer': self._bias_initializer,
            'kernel_regularizer': self._kernel_regulazer,
            'bias_regularizer': self._bias_initializer
        }
        self.fc1 = layers.Dense(self._hidden_dim or input_shape[1],
                                **common_kwargs
                                )
        self.fc2 = layers.Dense(self._output_dim or input_shape[1],
                                **common_kwargs)
        self.dropout = layers.Dropout(self._dropout_rate)
        return super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = self.fc1(inputs)
        x = self._activation(x)
        if self._dropout_rate > 0.0:
            x = self.dropout(x)
        x = self.fc2(x)
        if self._dropout_rate > 0.0:
            x = self.dropout(x)
        return x


class MultiscaleBlock(layers.Layer):
    def __init__(self,
                 output_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = False,
                 dropout_rate: float = 0.0,
                 droppath_rate: float = 0.0,
                 activation_layer: Callable = tf.nn.gelu,
                 norm_epsilon: float = 1e-6,
                 kernel_q: tuple = (1, 1, 1),
                 kernel_kv: tuple = (1, 1, 1),
                 stride_q: tuple = (1, 1, 1),
                 stride_kv: tuple = (1, 1, 1),
                 pool_mode: str = 'conv',
                 pool_first: bool = False,
                 embed_initializer: object = 'zeros',
                 kernel_initializer: object = 'zeros',
                 bias_initializer: object = 'zeros',
                 kernel_regularizer: Optional[object] = None,
                 bias_regularizer: Optional[object] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._output_dim = output_dim
        self._num_heads = num_heads
        self._mlp_ratio = mlp_ratio
        self._qkv_bias = qkv_bias
        self._dropout_rate = dropout_rate
        self._droppath_rate = droppath_rate
        self._activation = activation_layer
        self._norm_epsilon = norm_epsilon
        self._kernel_q = kernel_q
        self._kernel_kv = kernel_kv
        self._stride_q = stride_q
        self._stride_kv = stride_kv
        self._pool_mode = pool_mode
        self._pool_first = pool_first
        self._embed_initializer = tf.keras.initializers.get(embed_initializer)
        self._kernel_initializer = tf.keras.initializers.get(
            kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regulizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):

        kernel_skip = [s + 1 if s > 1 else s for s in self._stride_q]
        stride_skip = self._stride_q

        mlp_hidden_dim = int(input_shape[1] * self._mlp_ratio)

        common_kwargs = {
            'kernel_initializer': self._kernel_initializer,
            'bias_initializer': self._bias_initializer,
            'kernel_regularizer': self._kernel_regulizer,
            'bias_regularizer': self._bias_initializer
        }

        if self._stochastic_depth_drop_rate:
            self.droppath = StochasticDepth(self._droppath_rate)
        else:
            self.droppath = lambda x, *args, **kwargs: tf.identity(x)

        self.norm1 = layers.LayerNormalization(self._norm_epsilon)
        self.norm2 = layers.LayerNormalization(self._norm_epsilon)

        self.mlp = MLP(
            mlp_hidden_dim,
            self._output_dim,
            self._activation,
            self._dropout_rate,
            **common_kwargs
        )

        if input_shape[1] != self._output_dim:
            self.proj = layers.Dense(
                self._output_dim,
                **common_kwargs
            )

        self.pool_skip = layers.MaxPooling3D(
            kernel_skip,
            stride_skip,
            padding='same'
        )

        self.attn = MultiscaleAttention(
            patch_embed_shape=self._patch_embed_shape,
            num_heads=self._num_heads,
            qkv_bias=self._qkv_bias,
            dropout_rate=self._dropout_rate,
            kernel_q=self._kernel_q,
            kernel_kv=self._kernel_kv,
            stride_q=self._stride_q,
            stride_kv=self._stride_kv,
            norm_epsilon=self._norm_epsilon,
            pool_mode=self._pool_mode,
            pool_first=self._pool_first,
            embed_initializer=self._embed_initializer,
            **common_kwargs
        )

        return super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        tensor, patch_shape = inputs
        x_block, new_patch_shape = self.attn(self.norm1(tensor), patch_shape)
        x_res, _ = _attention_pooling(
            tensor, self.pool_skip, patch_shape
        )

        x = x_res + self.droppath(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)

        if inputs.shape[1] != self._output_dim:
            self.proj(x_mlp)
        x = x + self.droppath(x_mlp)
        return x, new_patch_shape
