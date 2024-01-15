import tensorflow as tf
from tensorflow import keras

#@tf.function(jit_compile=True)
def softmax_op( inputs, weights, beta, M):
    sum = tf.add(inputs, weights)
    sum = tf.scalar_mul(beta, sum)
    hv = tf.reduce_logsumexp(sum[..., :M], axis=-2)/beta
    hw = tf.reduce_logsumexp(-sum[..., M:], axis=-2)/beta
    return (hv, hw)

class SoftMaxMinPool(keras.layers.Layer):
    def __init__(self, units=128, M=None, activation='linear', beta=1, **kwargs):
        super(SoftMaxMinPool, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.beta = tf.constant(beta, dtype="float16")

        if M is None:
            self.M = units//2 #int div
        else:
            if M > units:
                raise Exception("M is greater than the number of neurons")
            self.M = M
        
    def build(self, input_shape):
        
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="weights_MNN",
        )
    
    def call(self, inputs):
        inputs_expanded = tf.expand_dims(inputs, axis=2)
        w_expanded = tf.expand_dims(self.w, axis=0)
        _M = tf.constant(self.M)
        _beta = self.beta
        hv, hw = softmax_op(inputs_expanded, w_expanded, _beta, _M)

        return self.activation(tf.concat([hv,hw], axis=1))
    
        """hv = tf.reduce_logsumexp(sum[..., :self.M], axis=-2)
        hv = tf.scalar_mul(1/self.beta, hv)

        sum = tf.negative(sum)
        hw = tf.reduce_logsumexp(sum[..., :self.M], axis=-2)
        hw = tf.scalar_mul(-1/self.beta, hw)"""

        """sum = tf.add(inputs_expanded, w_expanded)
        sum = tf.scalar_mul(self.beta, sum)
        hv = tf.reduce_logsumexp(sum[..., :self.M], axis=-2)/self.beta
        hw = tf.reduce_logsumexp(-sum[..., self.M:], axis=-2)/self.beta"""