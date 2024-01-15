import tensorflow as tf
from tensorflow import keras

class SoftMaxMin_Pruned(keras.layers.Layer): 
    def __init__(self, p_masc, p_masc_for_sum, units=200, max_units=None, activation='linear', beta=15, **kwargs):
        super(SoftMaxMin_Pruned, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.beta = tf.constant(beta, dtype=self.dtype)
        self.p_masc=p_masc
        self.p_masc_for_sum=p_masc_for_sum
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
        sum = tf.add(inputs, weights)
        sum = tf.scalar_mul(beta, sum)
        hv = tf.reduce_logsumexp(sum[..., :M], axis=-2)/beta
        hw = tf.reduce_logsumexp(-sum[..., M:], axis=-2)/beta
        return (hv, hw)
    
    def call(self, inputs):
        inputs_expanded = tf.expand_dims(inputs, axis=2)
        weigths=tf.math.add(self.w,self.p_masc_for_sum) # eliminar fila para func normal
        w_expanded = tf.expand_dims(weigths, axis=0) # cambiar weights por self.w para funcionamiento normal
        _M = tf.constant(self.M)
        _beta = self.beta
        hv, hw = self.softmax_op(inputs_expanded, w_expanded, _beta, _M)
        return self.activation(tf.concat([hv,hw], axis=1))
    
class SoftMaxMin_Pruned_v02(keras.layers.Layer):
    def __init__(self, p_weigths, units=200, max_units=None, activation='linear', beta=15, **kwargs):
        super(SoftMaxMin_Pruned_v02, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.beta = tf.constant(beta, dtype=self.dtype)
        self.p_weigths=p_weigths

        if max_units is None:
            self.M = units//2 #int div
        else:
            if max_units > units:
                raise Exception("M is greater than the number of neurons")
            self.M = max_units
        
    def build(self, input_shape):
        
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="ones",
            trainable=True,
            name="weights_MNN",
            dtype = self.dtype,
        )

    @tf.function(jit_compile=True)
    def softmax_op(self, inputs, weights, beta, M):

        sum = tf.add(inputs, weights)
        sum = tf.scalar_mul(beta, sum)
        hv = tf.reduce_logsumexp(sum[..., :M], axis=-2)/beta
        hw = tf.reduce_logsumexp(-sum[..., M:], axis=-2)/beta

        return (hv, hw)
    
    def call(self, inputs):

        inputs_expanded = tf.expand_dims(inputs, axis=2)
        weigths=tf.math.multiply(self.w, self.p_weigths) 
        w_expanded = tf.expand_dims(weigths, axis=0) 
        
        _M = tf.constant(self.M)
        _beta = self.beta
        hv, hw = self.softmax_op(inputs_expanded, w_expanded, _beta, _M)

        return self.activation(tf.concat([hv,hw], axis=1))

