import tensorflow as tf
from tensorflow import keras

class TropicalConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, beta, name=None, **kwargs):
        super(TropicalConv2D, self).__init__()
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.beta = beta

        if isinstance(kernel_size, tuple):
            if len(kernel_size) != 2:
                raise TypeError("Specify (height, width) or height for squared kernel. Only integer values")
            self.kernel_height = kernel_size[0]
            self.kernel_width = kernel_size[1]
        elif isinstance(kernel_size, int):
            self.kernel_height = kernel_size
            self.kernel_width = kernel_size
        else:
            raise TypeError("Specify (height, width) or height for squared kernel. Only integer values")

    
    def build(self, input_shape):

        self.in_channels = input_shape[-1]

        # kernel shape: [filter_height, filter_width, in_channels, out_channels]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_height*self.kernel_width*self.in_channels, self.filters),
            initializer="glorot_uniform",
            trainable=True,
            dtype = self.dtype,
        )
    
    @tf.function(jit_compile=True)
    def op(self, kernel, patches):
        patches = tf.expand_dims(patches, axis=-1, name="expand_patches")
        output = tf.add(kernel, patches)
        #output = tf.scalar_mul(self.beta, output)
        #output = tf.math.reduce_logsumexp(output, axis=-2)
        output = tf.math.reduce_max(output, axis=-2)
        return tf.math.scalar_mul(1/self.beta, output)
        #patches = tf.expand_dims(patches, axis=-1, name="expand_patches")
        #output = tf.raw_ops.Mul(x=kernel, y=patches, name="mult_kernel")
        #return tf.math.reduce_sum(output, axis=-2, name="reducing_sum")
    
    @tf.function(jit_compile=False)
    def get_patches(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.kernel_height, self.kernel_width, 1],
            strides=[1, self.strides, self.strides, 1],
            rates=[1, 1, 1, 1],
            padding=self.padding,
            name="patches"
        )
        # patches output dimension: [batch, out_height, out_width, filter_height * filter_width * in_channels] -> Add 1 dimension to broadcast
        return patches
    
    #@tf.function(jit_compile=True)
    def call(self, inputs):
        patches = self.get_patches(inputs)

        # flatten kernel: [filter_height * filter_width * in_channels, output_channels]
        #kernel_reshaped = tf.reshape(self.kernel, shape=(self.kernel_height*self.kernel_width*self.in_channels, self.filters), name="reshape_kernel")
        #if kernel_reshaped.dtype != patches.dtype:
        #    kernel_reshaped = tf.cast(kernel_reshaped, patches.dtype)
        # patches output dimension: [batch, out_height, out_width, filter_height * filter_width * in_channels] -> Add 1 dimension to broadcast
        #patches_expanded = tf.expand_dims(patches, axis=-1)

        """mul = tf.multiply(kernel_reshaped, patches_expanded)
        my_conv2d = tf.math.reduce_sum(mul, axis=-2)"""

        return self.op(self.kernel, patches)