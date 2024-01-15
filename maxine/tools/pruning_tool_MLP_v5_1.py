import tensorflow as tf
from tensorflow import keras
from keras import backend
from keras import regularizers
import numpy as np
import maxine.layers
from maxine.layers.callbacks import PruningCallback_MLP_2
from maxine.tools import pruning_tool_v4

@keras.saving.register_keras_serializable(package="PruningConstraint")
class PruningConstraint(tf.keras.constraints.Constraint):
    """Constrains weight tensors to keep zero values from pruning mascs during training."""

    def __init__(self, masc):
        self.masc = masc

    def __call__(self, w):
        return tf.multiply(w, self.masc)
    
    def get_config(self):
        return {'masc': self.masc}

def pruning_MLP(model, x_train, y_train, x_val, y_val, x_test, y_test, batch_size, morph_epochs=70, mlp_epochs=100, mlp_epochs_refit=50, p=20, ommit_layers=[], extreme=False):
    """
        Prune MLP model with Tropical Pruning. Retrain only pruned layer.
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
            dense_layers_config.append({'n_layer':n_layer, 'units':_config['units'], 'dtype':layer.compute_dtype, 'use_bias':layer.use_bias})

    # %%
    # GET INPUT DATA OF THE MODEL
    model_input = model.input
    layers_count = len(model.layers)

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

    # Set non trainable layers. Only prune pruned layer
    for layer in model_layers:
        layer.trainable = False
    # model_layers[-1].trainable = True # Trainable output layer

    train_data_morph = train_data
    val_data_morph = val_data
    test_data_morph = test_data
    

    # %%
    #------------------------------------------------------------------------------------------------------------------------------ 

    # FOR EVERY DENSE LAYER, CREATE AN EQUIVALENT MORFOLOGICAL LAYER, TRAIN, PRUNE, AND SAVE OUTPUT OF MORF LAYER AND PRUNING MASC 

    #------------------------------------------------------------------------------------------------------------------------------ 

    hp_masc_list = []
    ep_masc_list = []

    for i, (layer, layer_config) in enumerate(zip(dense_layers, dense_layers_config)):

        if layer_config['n_layer'] == (layers_count - 1): #Output layer
            break
        
        ################### MORPHOLOGICAL ###################
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
        # Works with pruning_tool_v2.5
        _, hp_masc, ep_masc = pruning_tool_v4.pruning_morphological(morph_model, train_data_morph, y_train, val_data_morph, y_val, test_data_morph, y_test, p=p, batch_size=batch_size, extreme=extreme, optimizer='Adam', metrics=['accuracy'], finetuning=False)
        hp_masc_list.append(hp_masc[0])
        ep_masc_list.append(ep_masc[0])

        ################### DENSE ###################
        # CREATE AND TRAIN NEW MLP MODEL WITH PRUNING MASCS
        model_layers.pop(layer_config['n_layer'])

        if extreme:
            masc = ep_masc[0]
        else:
            masc = hp_masc[0]

        model_layers.insert(layer_config['n_layer'], tf.keras.layers.Dense(layer_config['units'], activation='relu', use_bias=layer_config['use_bias'], kernel_constraint=PruningConstraint(masc)))

        # Rename layers
        for x, model_layer in enumerate(model_layers):
            model_layer._name = "modelLayer_" + str(x) + '_' + str(i)

        tf.keras.backend.clear_session()
        
        # Re-train MLP model
        model_tool = tf.keras.Sequential(model_layers)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, weight_decay=0.01)
        model_tool.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'],
              jit_compile=True
            ) #from logits = Ture if no activation function on the output
        model_tool.build(input_shape=train_data.shape)
        model_tool.layers[layer_config['n_layer']].set_weights(layer.get_weights()) # Set weights of the given model
        model_tool.fit(train_data, y_train, batch_size=batch_size, epochs=mlp_epochs_refit, validation_data=(val_data, y_val)) # Refit model with the masc

        # Set pruned layer to non-trainable after pruning
        model_tool.layers[layer_config['n_layer']].trainable = False
        
        # Get output of the mlp layer for the dataset -> Train morphological
        layer_output = model_tool.layers[layer_config['n_layer']].output
        output_func = backend.function([model_tool.input], [layer_output]) 
        train_data_morph = output_func(train_data)[0]
        val_data_morph = output_func(val_data)[0]
        test_data_morph = output_func(test_data)[0]


    # %%
    ################### MODEL ###################
    # model_tool.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=mlp_epochs, batch_size=batch_size, verbose=2)
    model_tool.evaluate(x_test, y_test)

    return model_tool, hp_masc_list, ep_masc_list
