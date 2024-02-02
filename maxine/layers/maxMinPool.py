import tensorflow as tf
from tensorflow import keras

class MaxMinPool(keras.layers.Layer):
    def __init__(self, units=128, M=None, activation='linear', **kwargs):
        super(MaxMinPool, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

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
        )

    def call(self, inputs):
        inputs_expanded = tf.expand_dims(inputs, axis=2)
        w_expanded = tf.expand_dims(self.w, axis=0)
        sum = tf.add(inputs_expanded, w_expanded)
        hv = tf.reduce_max(sum[..., :self.M], axis=-2)
        hw = tf.reduce_min(sum[..., self.M:], axis=-2)
        return self.activation(tf.concat([hv,hw], axis=1))
    
