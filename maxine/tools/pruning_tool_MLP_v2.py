import tensorflow as tf
from keras import backend
import numpy as np
import maxine.layers
from maxine.layers.callbacks import PruningCallback_MLP_2
from maxine.tools import pruning_tool_v3

def pruning_MLP(model, x_train, y_train, x_val, y_val, x_test, y_test, batch_size, morph_epochs=70, mlp_epochs=100, surv_weights=20, ommit_layers=[], extreme=False):

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
            dense_layers_config.append({'n_layer':n_layer, 'units':_config['units'], 'dtype':layer.compute_dtype})

    # %%
    # GET INPUT DATA OF THE MODEL
    #train_data = tf.concat([train_images, val_images], 0)
    #labels = np.concatenate([train_labels, val_labels], 0)
    model_input = model.input
    layers_count = len(model.layers)
    # layer_output = model.layers[0].output # Input of the morphological layer
    # output_func = backend.function([model_input], [layer_output]) 
    # layer_input = output_func(train_data)[0] # Expected only one output

    # Obtain the input of the dense layer
    n_layer_output = dense_layers_config[0]['n_layer']
    if n_layer_output > 0: # 1st Dense layer is not the first layer of the model
        layer_output = model.layers[n_layer_output - 1].output # Input of the morphological layer
        output_func = backend.function([model_input], [layer_output]) 
        #layer_input = output_func(train_data)[0] # Expected only one output
        train_data = output_func(x_train)[0]
        val_data = output_func(x_val)[0]
        test_data = output_func(x_test)[0]

    else: # NO FLATTEN LAYER OR INPUT LAYER
        #layer_input = train_data
        train_data = x_train
        val_data = x_val
        test_data = x_test

    #layer_input_tf = tf.convert_to_tensor(layer_input, dtype=dense_layers_config[0]['dtype'])


    # %%
    #------------------------------------------------------------------------------------------------------------------------------ 

    # FOR EVERY DENSE LAYER, CREATE AN EQUIVALENT MORFOLOGICAL LAYER, TRAIN, PRUN, AND SAVE OUTPUT OF MORF LAYER AND PRUNING MASC 

    #------------------------------------------------------------------------------------------------------------------------------ 

    hp_masc_list = []
    ep_masc_list = []

    for layer, layer_config in zip(dense_layers, dense_layers_config):

        # CREATE MORPH LAYER WITH SAME NUMBER OF NEURONS max_neurons=units/2 default
        morph_model = tf.keras.Sequential([
            maxine.layers.SoftMaxMin(units=layer_config['units']),
            tf.keras.layers.Dense(10, activation='softmax', use_bias=False)
        ])

        morph_model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
        
        morph_model.build(input_shape=train_data.shape)
        morph_model.summary()
        
        morph_model.fit(train_data, y_train, validation_data=(test_data, y_test), epochs=morph_epochs, verbose=2)
        # Works with pruning_tool_v2.5
        pruned_model, hp_masc, ep_masc = pruning_tool_v3.pruning_morphological(morph_model, train_data, y_train, val_data, y_val, test_data, y_test, surv_weights=surv_weights, batch_size=batch_size, extreme=extreme, optimizer='Adam', metrics=['accuracy'])
        hp_masc_list.append(hp_masc[0])
        ep_masc_list.append(ep_masc[0])

        if layer_config['n_layer'] == (layers_count - 2): #Output layer
            break
        
        # Get output of the morphological layer for the dataset
        layer_output = pruned_model.layers[0].output
        output_func = backend.function([pruned_model.input], [layer_output]) 
        #layer_input = output_func(layer_input)[0] # Expected only one output
        train_data = output_func(train_data)[0]
        val_data = output_func(val_data)[0]
        test_data = output_func(test_data)[0]
        #layer_input_tf = tf.convert_to_tensor(layer_input, dtype=layer_config['dtype'])

        # PRUNE THE MODEL

    # %%
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=mlp_epochs, callbacks=[PruningCallback_MLP_2(dense_layers, hp_masc_list)])

    for layer, masc in zip(dense_layers, hp_masc_list):
        weights = layer.get_weights()
        new_weights = np.multiply(weights[0], masc)
        weights[0] = new_weights
        layer.set_weights(weights)

    return model, hp_masc_list, ep_masc_list
