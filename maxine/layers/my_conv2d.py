import tensorflow as tf
from tensorflow import keras

class My_Conv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, name=None, **kwargs):
        super(My_Conv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
    
    def build(self, input_shape):

        self.in_channels = input_shape[-1]

        # kernel shape: [filter_height, filter_width, in_channels, out_channels]
        self.kernel = self.add_weight(
            shape=(self.kernel_size, self.kernel_size, self.in_channels, self.filters),
            initializer="random_normal",
            trainable=True,
        )
    
    @tf.function(jit_compile=True)
    def op(self, kernel, patches):
        mul = tf.multiply(kernel, patches)
        return tf.math.reduce_sum(mul, axis=-2)
    
    #@tf.function(jit_compile=True)
    def call(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.strides, self.strides, 1],
            rates=[1, 1, 1, 1],
            padding=self.padding
        )

        # flatten kernel: [filter_height * filter_width * in_channels, output_channels]
        kernel_reshaped = tf.reshape(self.kernel, shape=(self.kernel_size*self.kernel_size*self.in_channels, self.filters))
        # patches output dimension: [batch, out_height, out_width, filter_height * filter_width * in_channels] -> Add 1 dimension to broadcast
        patches_expanded = tf.expand_dims(patches, axis=-1)

        """mul = tf.multiply(kernel_reshaped, patches_expanded)
        my_conv2d = tf.math.reduce_sum(mul, axis=-2)"""

        return self.op(kernel_reshaped, patches_expanded)