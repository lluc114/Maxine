import tensorflow as tf
from tensorflow import keras

class SoftMaxMin(keras.layers.Layer):
    def __init__(self, units=128, max_units=None, activation='linear', beta=15, **kwargs):
        super(SoftMaxMin, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.activation_type = activation
        self.beta_arg = beta 

        self.beta = tf.constant(beta, dtype=self.compute_dtype)

        if max_units is None:
            self.M = units//2 #int div
        else:
            if max_units > units:
                raise Exception("M is greater than the number of neurons")
            self.M = max_units
        
    def build(self, input_shape):
        
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="weights_MNN",
            dtype = self.dtype,
        )

    @tf.function(jit_compile=True)
    def softmax_op(self, inputs, weights, beta, M):
        sum = tf.add(inputs, weights, name="sum_inputs_weights")
        sum = tf.scalar_mul(beta, sum, name="multiply_beta")
        hv = tf.reduce_logsumexp(sum[..., :M], axis=-2, name="hv")/beta
        hw = -tf.reduce_logsumexp(-sum[..., M:], axis=-2, name="hw")/beta
        return (hv, hw)
    
    def call(self, inputs):
        inputs_expanded = tf.expand_dims(inputs, axis=2)
        w_expanded = tf.expand_dims(self.w, axis=0)
        _M = tf.constant(self.M)
        _beta = self.beta
        hv, hw = self.softmax_op(inputs_expanded, w_expanded, _beta, _M)

        return self.activation(tf.concat([hv,hw], axis=1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "max_units": self.M,
            "activation": self.activation_type,
            "beta": self.beta_arg
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)