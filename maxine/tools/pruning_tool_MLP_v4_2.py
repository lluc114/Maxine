import tensorflow as tf
from tensorflow import keras
from keras import backend
from keras import regularizers
import numpy as np
import maxine.layers
from maxine.layers.callbacks import PruningCallback_MLP_2
from maxine.tools import pruning_tool_v4_2

@keras.saving.register_keras_serializable(package="PruningConstraint")
class PruningConstraint(tf.keras.constraints.Constraint):
    """Constrains weight tensors to keep zero values from pruning mascs during training."""

    def __init__(self, masc):
        self.masc = masc

    def __call__(self, w):
        return tf.multiply(w, self.masc)
    
    def get_config(self):
        return {'masc': self.masc}

@keras.saving.register_keras_serializable(package="Weights")
class Weights(tf.keras.initializers.Initializer):
    def __init__(self, weights):
      self.weights = weights

    def __call__(self, shape, dtype=None):
    #   if shape != self.weights.shape: 
        #   raise ValueError("Shape mismatch")
      return tf.convert_to_tensor(self.weights, dtype=dtype)
    
    def get_config(self):  # To support serialization
      return {'weights': self.weights}
    
# @keras.saving.register_keras_serializable(package="PruningConstraint")
# class Bias(tf.keras.initializers.Initializer):
#     def __init__(self, bias):
#       self.bias = bias

#     def __call__(self, shape, dtype=None):
#       if shape != self.bias.shape: 
#           raise ValueError("Shape mismatch")
#       return tf.convert_to_tensor(self.bias, dtype=dtype)
    
#     def get_config(self):  # To support serialization
#       return {'bias': self.bias}
    

def pruning_MLP(model, x_train, y_train, x_val, y_val, x_test, y_test, batch_size, morph_epochs=70, mlp_epochs=100, mlp_epochs_refit=50, p=20, pList=[], ommit_layers=[], optimizer_config=None, extreme=False, preLoad=False, activLists=[], activCounts=[]):
    """
        Prune MLP model with Tropical Pruning. Entire model is retrained after pruning mascs are applied.
    """
    # PRUNING TOOL MLP FUNCTION
    #--------------------------
    # GET DENSE LAYERS AND CONFIG
    #--------------------------
    dense_layers = []
    dense_layers_config = []
    n_dense_parameters = 0
    for n_layer, layer in enumerate(model.layers):
        layer_type = str(type(layer))
        if (layer_type.find("Dense") != -1) & (n_layer not in ommit_layers):
            dense_layers.append(layer)
            n_dense_parameters += layer.count_params()
            _config = layer.get_config()
            dense_layers_config.append({'n_layer':n_layer, 'units':_config['units'], 'dtype':layer.compute_dtype, 'use_bias':layer.use_bias, 'activation':_config['activation']})

    # %%
    # GET INPUT DATA OF THE MODEL

    model_input = model.input
    layers_count = len(dense_layers)

    # Obtain the input of the dense layer
    n_layer_output = dense_layers_config[0]['n_layer']
    if n_layer_output > 0: # 1st Dense layer is not the first layer of the model
        layer_output = model.layers[n_layer_output - 1].output # Input of the morphological layer
        output_func = backend.function([model_input], [layer_output]) 
        train_data = output_func(x_train)[0]
        val_data = output_func(x_val)[0]
        test_data = output_func(x_test)[0]

    else: # NO FLATTEN LAYER OR INPUT LAYER
        train_data = x_train
        val_data = x_val
        test_data = x_test
    
    # GET MODEL LAYERS
    model_layers = model.layers

    train_data_morph = train_data
    val_data_morph = val_data
    test_data_morph = test_data
    

    # %%
    #------------------------------------------------------------------------------------------------------------------------------ 

    # FOR EVERY DENSE LAYER, CREATE AN EQUIVALENT MORPHOLOGICAL LAYER, TRAIN, PRUNE, AND SAVE OUTPUT OF MORPH LAYER AND PRUNING MASC 

    #------------------------------------------------------------------------------------------------------------------------------ 

    hp_masc_list = []
    ep_masc_list = []
    _activLists = [None]
    _activCounts = [None]

    pruned_params = 0

    for i, (layer, layer_config) in enumerate(zip(dense_layers, dense_layers_config)):
        #--------------------------
        # MORPHOLOGICAL MODEL
        #--------------------------
        print("Training morphological model...")
        # CREATE MORPH LAYER WITH SAME NUMBER OF NEURONS max_neurons=units/2 default
        morph_model = tf.keras.Sequential([
            maxine.layers.SoftMaxMin(units=layer_config['units']),
            tf.keras.layers.Dense(10, activation='softmax', use_bias=False)
        ])

        morph_model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
        
        morph_model.build(input_shape=train_data_morph.shape)
        morph_model.summary()
        
        morph_model.fit(train_data_morph, y_train, validation_data=(val_data_morph, y_val), epochs=morph_epochs, verbose=2)
       
        if len(pList) > 0:
            _p = pList[i]
        else:
            _p = p

        if preLoad:
            _activLists[0] = activLists[i]
            _activCounts[0] = activCounts[i]
        print("Pruning morphological model...")
        _, hp_masc, ep_masc = pruning_tool_v4_2.pruning_morphological(morph_model, train_data_morph, y_train, val_data_morph, y_val, test_data_morph, y_test, p=_p, batch_size=batch_size, extreme=extreme, optimizer='adam', metrics=['accuracy'], finetuning=False, preLoad=preLoad, activLists=_activLists, activCounts=_activCounts)
        hp_masc_list.append(hp_masc[0])
        ep_masc_list.append(ep_masc[0])

        tf.keras.backend.clear_session()

        #--------------------------
        # DENSE MODEL
        #--------------------------
        print("Retraining MLP dense model...")
        # CREATE AND TRAIN NEW MLP MODEL WITH PRUNING MASCS
        model_layers.pop(layer_config['n_layer'])

        if extreme:
            masc = ep_masc[0]
        else:
            masc = hp_masc[0]

        pruned_params = pruned_params + masc.size - np.count_nonzero(masc)

        new_w = np.multiply(masc, layer.get_weights()[0])
        bias = layer.get_weights()[1]

        if (layer_config['use_bias']):
            model_layers.insert(layer_config['n_layer'], tf.keras.layers.Dense(layer_config['units'], activation=layer_config['activation'], use_bias=True, kernel_initializer=Weights(new_w), kernel_constraint=PruningConstraint(masc), bias_initializer=Weights(bias)))
        else:
            model_layers.insert(layer_config['n_layer'], tf.keras.layers.Dense(layer_config['units'], activation=layer_config['activation'], use_bias=False, kernel_initializer=Weights(new_w), kernel_constraint=PruningConstraint(masc)))
            

        # Rename layers
        for x, model_layer in enumerate(model_layers):
            model_layer._name = "modelLayer_" + str(x) + '_' + str(i)
        
        # Re-train MLP model
        model_tool = tf.keras.Sequential(model_layers)
        if optimizer_config is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.01)
        else:
            optimizer = tf.keras.optimizers.deserialize(optimizer_config)

        model_tool.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'],
              jit_compile=True
            ) #from logits = Ture if no activation function on the output
        model_tool.build(input_shape=x_train.shape)
        # model_tool.layers[layer_config['n_layer']].set_weights(layer.get_weights()) # Set weights of the given model
        model_tool.fit(x_train, y_train, batch_size=batch_size, epochs=mlp_epochs_refit, validation_data=(x_val, y_val), verbose=2) # Refit model with the masc

        if i == (layers_count - 2): # The next layer is the output layer -> end pruning
            break
        
        #--------------------------
        # GET INPUT OF MPTOHOLOGICAL MODEL
        #--------------------------
        # Get output of the mlp layer for the dataset -> Train morphological En la ultima iteracion no hace falta hacer esto
        layer_output = model_tool.layers[layer_config['n_layer']].output
        output_func = backend.function([model_tool.input], [layer_output]) 
        train_data_morph = output_func(x_train)[0]
        val_data_morph = output_func(x_val)[0]
        test_data_morph = output_func(x_test)[0]


    # %%
    print("Retraining MLP model...")
    model_tool.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=mlp_epochs, batch_size=batch_size, verbose=2)
    print("Accuracy of the MLP model:")
    model_tool.evaluate(x_test, y_test)
    print("# Parameters of the model: " + str(model_tool.count_params()))
    print("# Parameters pruned: " + str(pruned_params))
    print("# Parameters remaining: " + str(model_tool.count_params()-pruned_params))
    print("Remaining weights (%): " + str((model_tool.count_params()-pruned_params)/model_tool.count_params()*100))

    return model_tool, hp_masc_list, ep_masc_list
