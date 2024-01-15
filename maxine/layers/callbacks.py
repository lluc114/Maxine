import tensorflow as tf
from tensorflow import keras
import numpy as np


# Creación de una clase callback que permite modificar el parámetro interno de la B de distintas formas
class BvarCallback(tf.keras.callbacks.Callback):
    """

    Callback function used to vary the B parameter, depending on the epoch or batch step, if keras fit training method is used.
    The name of the softMaxMin layers must contain 'soft_max_min' and the name of the convolutional layers must contain 'soft_tropical_conv2d'.

    Args:
    -----
    model : tf.keras.models.Model
        The model compiled. 
    B_var_mode : np.array
    batch_size : int
    epochs : int
    layer_type : str
        'norm' for softMaxMin layers and 'conv' for convolutional layers. 
    beta_range : list [int, int]
        The range in which beta varies lienally.
    """
    def __init__ (self, model, B_var_mode, batch_size, epochs, layer_type, beta_range=[5, 100]):
        super().__init__()
        self.beta_range=beta_range
        self.best_weights=None
        self.model=model
        self.batch_size=batch_size
        self.epochs=epochs
        self.B_var_mode=B_var_mode

        if B_var_mode == "epoch":
            self.beta_step = beta_range[0] + (beta_range[1] - beta_range[0])/epochs
        elif B_var_mode == "batch_step":
            self.beta_step = beta_range[0] + (beta_range[1] - beta_range[0])/batch_size
        else:
            raise Exception("B_var_mode parameter must be specified, batch_step or epochs!")
        
        self.morf_layers = []
        for ind, layer in enumerate(self.model.layers):
            if layer_type=="norm":
                if layer.name.find("soft_max_min") != -1:
                    self.morf_layers.append(ind)
            elif layer_type=="conv":
                if layer.name.find("soft_tropical_conv2d") != -1:
                    self.morf_layers.append(ind)
            else:
                raise Exception("There is no morphological layer type defined. Please define morphological layer type for pruning, 'conv' or 'norm'!")

    def on_train_begin(self, logs=None):
        for ind in self.morf_layers:
            self.model.layers[ind].beta=self.beta_range[0]

    def on_epoch_begin(self, epoch, logs=None):
        if self.B_var_mode=='epoch':
            for ind in self.morf_layers:
                beta_layer = float(tf.keras.backend.get_value(self.model.layers[ind].beta))
                layer_name = self.model.layers[ind].name
                print("Epoch: " + str(epoch)+ ' | The B value of layer ' + layer_name + ' is ' + str(beta_layer))
            print('\n')
        else:
            pass
    def on_epoch_end(self, epoch, logs=None):
        if self.B_var_mode=='epoch':
            for ind in self.morf_layers:
                # Set the new b value
                beta_layer_1 = float(tf.keras.backend.get_value(self.model.layers[ind].beta))
                self.model.layers[ind].beta= beta_layer_1 + self.beta_step
        else:
            pass

    def on_train_batch_begin(self, epoch, batch, logs=None):
        if self.B_var_mode=='batch_step':
            for ind in self.morf_layers:
                beta_layer = float(tf.keras.backend.get_value(self.model.layers[ind].beta))
                layer_name = self.model.layers[ind].name
                print("Epoch: " + str(batch)+ ' | The B value of layer ' + layer_name + ' is ' + str(beta_layer))
            print('\n')
        else:
            pass

    def on_train_batch_end(self, batch, logs=None):
        if self.B_var_mode=='batch_step':
            for ind in self.morf_layers:
                # Set the new b value
                beta_layer_1 = float(tf.keras.backend.get_value(self.model.layers[ind].beta))
                self.model.layers[ind].beta= beta_layer_1 + self.beta_step
        else:
            pass


#Se define una capa para realizar el pruning de una capa morflogica
class PruningCallback(tf.keras.callbacks.Callback):
    """

    Callback function used to vary the B parameter, depending on the epoch or batch step, if keras fit training method is used.
    The name of the softMaxMin layers must contain 'soft_max_min' and the name of the convolutional layers must contain 'soft_tropical_conv2d'.

    Args:
    -----
    model : tf.keras.models.Model
        The model compiled. 
    B_var_mode : np.array
    batch_size : int
    epochs : int
    layer_type : str
        'norm' for softMaxMin layers and 'conv' for convolutional layers. 
    beta_range : list [int, int]
        The range in which beta varies lienally.
    """
    def __init__ (self, model, layer_type, weight_threshold, gradient_threshold):
        super().__init__()
        self.model = model
        self.layer_type = layer_type
        self.weight_threshold = weight_threshold
        self.gradient_threshold = gradient_threshold
        self.layer_names=[layer.name for layer in self.model.layers]
        self.weight_mask = []
        self.morf_layers = []
        for ind, layer in enumerate(self.model.layers):
            if self.layer_type=="norm":
                if layer.name.find("soft_max_min") != -1:
                    self.morf_layers.append(ind)
            elif self.layer_type=="conv":
                if layer.name.find("soft_tropical_conv2d") != -1:
                    self.morf_layers.append(ind)
            else:
                raise Exception("There is no morphological layer type defined. Please define morphological layer type for pruning, 'conv' or 'norm'!")

        self.weight_mask = [None]*len(self.morf_layers)
    
    def on_train_batch_begin(self, batch, logs=None):
        for i, ind in  enumerate(self.morf_layers):
            layer = self.model.layers[ind]
            self.weight_mask[i] = layer.get_weights()

    def on_train_batch_end(self, batch, logs=None):
        if batch == 0:
            return
        for i, ind in  enumerate(self.morf_layers):
            morf_layer = self.model.layers[ind]
            weight_mask = self.weight_mask[i]
            if len(weight_mask)>0:
                w_mask = [abs(ele) for ele in weight_mask]
                w_weight = [abs(ele)for ele in morf_layer.get_weights()]
                w_gradient = np.subtract(np.array(w_mask), np.array(w_weight))
                grad_condition = w_gradient<=self.gradient_threshold
                weight_condition = np.array(w_weight)<=self.weight_threshold
                p_cond=np.invert(grad_condition*weight_condition) # los true se pasan a false, para multiplicar por 0 cuando se cumple la condición
                p_matrix=p_cond.astype('uint8')
                p_weights=morf_layer.get_weights()*p_matrix
                # Setting the layer weights to the pruned values
                morf_layer.set_weights(p_weights)
            
    def on_train_end(self, epoch, logs=None):
        total_weights = 0
        activ_weights = 0
        for ind in self.morf_layers:
            morf_layer=self.model.layers[ind]
            total_weights = total_weights + np.array(morf_layer.get_weights()).size
            activ_weights = activ_weights + np.count_nonzero(np.array(morf_layer.get_weights()))
        print('Total weights: ' + str(total_weights))
        print("Pruned weights: "+ str(total_weights-activ_weights))
        print('Parameter reduction: '+ str(int(((total_weights-activ_weights)*100)/total_weights))+'%')


# A diferencia del pruning normal, en este caso se calcula la media del gradiente de variación de los pesos entre un batch y el siguiente, y si el valor del gradiente es menor a la media, se comprueba la condición del threshold
class PruningCallback_v02(tf.keras.callbacks.Callback):
    """

    Callback function used to vary the B parameter, depending on the epoch or batch step, if keras fit training method is used.
    The name of the softMaxMin layers must contain 'soft_max_min' and the name of the convolutional layers must contain 'soft_tropical_conv2d'.

    Args:
    -----
    model : tf.keras.models.Model
        The model compiled. 
    B_var_mode : np.array
    batch_size : int
    epochs : int
    layer_type : str
        'norm' for softMaxMin layers and 'conv' for convolutional layers. 
    beta_range : list [int, int]
        The range in which beta varies lienally.
    """
     
    def __init__ (self, model, layer_type, weight_threshold, gradient_threshold, input_shape):
        super().__init__()
        self.model = model
        self.layer_type = layer_type
        self.weight_threshold = weight_threshold
        self.gradient_threshold = gradient_threshold
        self.input_shape=input_shape
        self.layer_names=[layer.name for layer in self.model.layers]
        self.weight_mask = []
        self.morf_layers = []
        for ind, layer in enumerate(self.model.layers):
            if self.layer_type=="norm":
                if layer.name.find("soft_max_min") != -1:
                    self.morf_layers.append(ind)
            elif self.layer_type=="conv":
                if layer.name.find("soft_tropical_conv2d") != -1:
                    self.morf_layers.append(ind)
            else:
                raise Exception("There is no morphological layer type defined. Please define morphological layer type for pruning, 'conv' or 'norm'!")

        self.weight_mask = [None]*len(self.morf_layers)
    
    def on_train_batch_begin(self, batch, logs=None):
        for i, ind in  enumerate(self.morf_layers):
            layer = self.model.layers[ind]
            self.weight_mask[i] = layer.get_weights()

    def on_train_batch_end(self, batch, logs=None):
        if batch == 0:
            return
        for i, ind in  enumerate(self.morf_layers):
            morf_layer = self.model.layers[ind]
            weight_mask = self.weight_mask[i]
            if len(weight_mask)>0:
                w_mask = [abs(ele) for ele in weight_mask]
                w_weight = [abs(ele)for ele in morf_layer.get_weights()]
                w_gradient = np.subtract(np.array(w_mask), np.array(w_weight))
                # Dynamical gradient threshold
                w_grad_in=w_gradient[0,:,:]
                thresh=abs(sum(sum(w_grad_in)))/(len(w_gradient[0][0])*self.input_shape)
                #----------------------------------------
                grad_condition = w_gradient<=thresh
                weight_condition = np.array(w_weight)<=self.weight_threshold
                p_cond=np.invert(grad_condition*weight_condition) # los true se pasan a false, para multiplicar por 0 cuando se cumple la condición
                p_matrix=p_cond.astype('uint8')
                p_weights=morf_layer.get_weights()*p_matrix
                # Setting the layer weights to the pruned values
                morf_layer.set_weights(p_weights)
            
    def on_train_end(self, epoch, logs=None):
        total_weights = 0
        activ_weights = 0
        for ind in self.morf_layers:
            morf_layer=self.model.layers[ind]
            total_weights = total_weights + np.array(morf_layer.get_weights()).size
            activ_weights = activ_weights + np.count_nonzero(np.array(morf_layer.get_weights()))
        print('Total weights: ' + str(total_weights))
        print("Pruned weights: "+ str(total_weights-activ_weights))
        print('Parameter reduction: '+ str(int(((total_weights-activ_weights)*100)/total_weights))+'%')


class PruningCallback_MLP(tf.keras.callbacks.Callback):
    def __init__ (self, model,layer_type, input_shape, p_masc,hp_masc):
        super().__init__()
        self.model = model
        self.p_masc = p_masc
        self.layer_type = layer_type
        self.hp_masc=hp_masc
        self.input_shape=input_shape

    def on_train_batch_begin(self, batch, logs=None):
        layer = self.model.layers[0]
        w=tf.convert_to_tensor(layer.get_weights()[0])
        p_w=tf.math.multiply(w,self.p_masc)
        layer.set_weights([p_w])

        layer = self.model.layers[1]
        w=tf.convert_to_tensor(layer.get_weights()[0])
        p_w=tf.math.multiply(w,self.hp_masc[0])
        layer.set_weights([p_w])
        

class PruningCallback_MLP_2(tf.keras.callbacks.Callback):
    def __init__ (self, dense_layers, hp_masc_list):
        super().__init__()
        self.dense_layers = dense_layers
        self.hp_masc_list = hp_masc_list

    def on_train_batch_begin(self, batch, logs=None):
        for layer, masc in zip(self.dense_layers, self.hp_masc_list):
            weights = layer.get_weights()
            new_weights = np.multiply(weights[0], masc)
            weights[0] = new_weights
            layer.set_weights(weights)