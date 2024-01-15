import tensorflow as tf
from keras import backend
import numpy as np
import maxine

#--------------------------
# FORMWARD STEP OF THE SOFTMAXMIN NET
#--------------------------
def forward_pass(data, weights, M_max, units, dtype):
    activation_list = np.zeros_like(weights, dtype=int)
    activation_count = np.zeros_like(weights, dtype=int)
    max_index = np.arange(0, M_max, dtype=int)
    min_index = np.arange(M_max, units, dtype=int)
    t_weights = tf.convert_to_tensor(weights, dtype=dtype)

    for input in data:
        input_expanded = tf.expand_dims(input, axis=-1)
        sum = tf.add(input_expanded, t_weights)
        max_pos = tf.argmax(sum[..., :M_max], axis=0)
        min_pos = tf.argmin(sum[..., M_max:], axis=0)

        _max_pos = max_pos.numpy()
        _min_pos = min_pos.numpy()

        activation_list[_max_pos, max_index] = 1
        activation_list[_min_pos, min_index] = 1

        activation_count[_max_pos, max_index] += 1
        activation_count[_min_pos, min_index] += 1

    return activation_list, activation_count


def pruning_morphological(trained_model, x_train, y_train, x_val, y_val, x_test, y_test, surv_weights, batch_size, extreme=False, optimizer=None, metrics=None, loss=None, finetuning=False):
    """

    Morphological nets pruning tool


    Args:
    -----
    trained_model : tf.keras.models.Model
        The model compiled and trained with a specific batch size. 
    x_train : np.array
    y_train : np.array
    x_val : np.array
    y_val : np.array
    x_test : np.array
    y_test : np.array
    surv_weights : int
        Percentage of weights to keep, per morphological layer, after the hard pruning
    batch_size : int
        The batch size used to train the model
    extreme : boolean
        True for extreme pruning. It will drop even more weights after the hard pruning, according to surv_weights arg, and it will train the model again. 
    optimizer : string, optional
        Optimizer to retrain the model.
    metrics : string, optional
        Metrics for training the model
    loss : str
        Loss function.
    """
    
    # FOR MIXED PRECISION ENSURE DATA INPUT TYPE IS THE SAME THAN COMPUTE TYPE 
    model_compute_dtype = trained_model.compute_dtype
    x_train = x_train.astype(model_compute_dtype)
    x_val = x_val.astype(model_compute_dtype)
    x_test = x_test.astype(model_compute_dtype)

    # Original model test accuracy
    print('\nBaseline test accuracy: ')
    trained_model.evaluate(x_test, y_test, batch_size=batch_size) 
    
    #--------------------------
    # GET MORPHOLOGICAL LAYERS AND CONFIG
    #--------------------------
    morph_layers = []
    morph_layers_config = []
    M_max = 100
    n_morph_parameters = 0
    for n_layer, layer in enumerate(trained_model.layers):
        layer_type = str(type(layer))
        if layer_type.find("SoftMaxMin") != -1:
            morph_layers.append(layer)
            n_morph_parameters += layer.count_params()
            _config = layer.get_config()
            if 'max_units' in _config: # Number of Max neurons
                M_max = _config['max_units']
            else:
                M_max = _config['units']//2 # int div
            morph_layers_config.append({'n_layer':n_layer, 'units':_config['units'], 'max_units':M_max, 'dtype':layer.compute_dtype})

    #--------------------------
    # COMPUTE MASCS
    #--------------------------
    # Training data
    train_data=tf.concat([x_train, x_val], axis=0)

    #input_morph_layers = []
    hp_masc = []
    hp_M_masc = []
    ep_masc = []
    ep_M_masc = []
    layer_input = trained_model.input # Input of the model
    for morph_layer, layer_config in zip(morph_layers, morph_layers_config):

        # Get the input of the morphological layer
        if layer_config['n_layer'] == 0:
            input_morph_layer = train_data
        else:
            n_layer_output = layer_config['n_layer'] - 1 # Prev layer output
            layer_output = trained_model.layers[n_layer_output].output # Input of the morphological layer
            output_func = backend.function([layer_input], [layer_output]) 
            input_morph_layer = output_func(train_data)[0] # Expected only one output

        M_max = layer_config['max_units']
        dtype = layer_config['dtype']

        morph_layer_weights = morph_layer.get_weights()[0]

        #--------------------------
        # FORWARD PASS
        #--------------------------
        activation_list, activation_count = forward_pass(input_morph_layer, morph_layer_weights, M_max, layer_config['units'], dtype)

        #--------------------------
        # COMPUTE HARD PRUNING MASC
        #--------------------------
        _p_masc = activation_list.astype("bool")
        _p_masc = np.invert(_p_masc)
        _p_masc = _p_masc.astype(dtype)
        _p_masc[:, :M_max] = _p_masc[:, :M_max]*np.finfo(dtype).min
        _p_masc[:, M_max:] = _p_masc[:, M_max:]*np.finfo(dtype).max
        hp_masc.append(activation_list)
        hp_M_masc.append(_p_masc)

        #--------------------------
        # COMPUTE EXTREME PRUNING MASC
        #--------------------------
        #----------------------------------------------------------------------------------------------------------------------------------
        # PRUNING EXTREMO: solo se almacena la fracción de los pesos activos que indica el usuario (se eliminan los menos relevantes)
        # -------PASO 1
        # Se define el porcentage de los pesos activos que se quiere conservar
        weight_surv_percentage=tf.constant([surv_weights/100], dtype=dtype)
        # Se cuenta la cantidad de pesos activos que tiene cada una de las neuronas
        p_masc = tf.convert_to_tensor(activation_list, dtype=dtype)
        activ_weights=tf.math.count_nonzero(p_masc, axis=0, dtype=dtype)
        # Se calcula la cantidad de pesos que seguirà teniendo activos cada neurona una vez aplicado el pruning drástico
        activ_weights_after_prun=tf.math.round(tf.math.multiply(activ_weights, weight_surv_percentage))

        # --------PASO 2
        # Número de veces que ha ganado cada peso de cada neurona
        #weight_winner_count=tf.convert_to_tensor(np.concatenate((graf_5_activation_max_count,graf_5_activation_min_count),axis=1),dtype='float32')
        weight_winner_count=tf.convert_to_tensor(activation_count, dtype=dtype)
        # Total de winners para cada neurona
        total_winners_neuron=tf.math.reduce_sum(weight_winner_count, axis=0)
        # Frecuncia de winners de cada neurona
        frec_winners_count=tf.math.divide(weight_winner_count, total_winners_neuron)
        _p_masc_extra = np.zeros_like(morph_layer_weights, dtype=int)
        # Se obtiene la nueva máscara de pruning
        for i in range(_p_masc_extra.shape[1]):
            neuron=tf.math.top_k(frec_winners_count[:,i],k=activ_weights_after_prun[i].numpy().astype('int32'))
            _p_masc_extra[neuron.indices.numpy(),i] = 1

        # Extrem pruning masc
        _p_masc_extra_M = _p_masc_extra.astype("bool")
        _p_masc_extra_M = np.invert(_p_masc_extra_M)
        _p_masc_extra_M = _p_masc_extra_M.astype(dtype)
        _p_masc_extra_M[:, :M_max] = _p_masc_extra_M[:, :M_max]*np.finfo(dtype).min
        _p_masc_extra_M[:, M_max:] = _p_masc_extra_M[:, M_max:]*np.finfo(dtype).max
        ep_masc.append(_p_masc_extra)
        ep_M_masc.append(_p_masc_extra_M)
    
    #--------------------------
    # COMPUTE NUMBER OF PARAMETERS
    #--------------------------
    # Total weights model
    total_weights = trained_model.count_params()    
    n_pruned_weights = 0
    n_unactive_units = 0
    if extreme:
        for masc in ep_masc:
            n_pruned_weights += (masc.size - np.count_nonzero(masc))
            n_unactive_units += masc.shape[1] - np.count_nonzero(np.sum(masc, axis=0))
    else:
        for masc in hp_masc:
            n_pruned_weights += (masc.size - np.count_nonzero(masc))
            n_unactive_units += masc.shape[1] - np.count_nonzero(np.sum(masc, axis=0))

    # Pruning data
    print('Total paramaters: ' + str(total_weights))
    print("Weights that can be pruned: " + str(n_pruned_weights))
    print("Left parameters: " + str(total_weights - n_pruned_weights))
    print("Unactive neurons: " + str(n_unactive_units))
    print('Parameter reduction of the model: '+ str(int(((n_pruned_weights)*100)/total_weights))+'%')
    print('Parameter reduction of the morph layers: '+ str(int(((n_pruned_weights)*100)/n_morph_parameters))+'%')

    #--------------------------
    # PRUNE WEIGHTS
    #--------------------------
    for n_layer, (morph_layer, layer_config) in enumerate(zip(morph_layers, morph_layers_config)):
        if extreme:
            # Extreme pruning
            layer_weights = morph_layer.get_weights()[0] # Numpy
            layer_weights = np.multiply(layer_weights, ep_masc[n_layer])
            layer_weights = np.add(layer_weights, ep_M_masc[n_layer])
            p_weights = tf.convert_to_tensor(layer_weights, dtype=layer_config['dtype'])
            #trained_model.layers[layer_config['n_layer']].set_weights([p_weights])
            morph_layer.set_weights([p_weights])
            
        else:
            # Hard pruning
            layer_weights = morph_layer.get_weights()[0] # Numpy
            layer_weights = np.multiply(layer_weights, hp_masc[n_layer])
            layer_weights = np.add(layer_weights, hp_M_masc[n_layer])
            p_weights = tf.convert_to_tensor(layer_weights, dtype=layer_config['dtype'])
            #trained_model.layers[layer_config['n_layer']].set_weights([p_weights])
            morph_layer.set_weights([p_weights])

    #--------------------------
    # EVALUATE PRUNED MODEL
    #--------------------------
    if extreme:
        print('\nExtreme pruned model with with only ' + str(int(surv_weights)) +'%'+' active weights of the morphological layers remaining. Test acuraccy')
        
    else:
        print('\nHard pruned model. Test acuraccy')

    trained_model.evaluate(x_test, y_test, batch_size=batch_size)
    # Se almacena el modelo prunado
    #pruned_model.save('Modelos/Modelo_Entrenado_Mnist_100_epoch_200_neuronas_Extrem_Pruning', overwrite=True)

    #--------------------------
    # FINETUNING EXTREME PRUNED MODEL
    #--------------------------
    if extreme & finetuning:
        pruned_model = tf.keras.Sequential()
        n_morph_layer = 0
        for layer in trained_model.layers:
            if layer in morph_layers:
                _p_weights = tf.convert_to_tensor(layer.get_weights()[0], dtype=pruned_model.dtype)
                pruned_model.add(maxine.layers.SoftMaxMin_Pruned_v02(units=morph_layers_config[n_morph_layer]['units'], max_units=morph_layers_config[n_morph_layer]['max_units'], p_weigths=_p_weights, p_masc_for_sum=None))
                n_morph_layer += 1
            else:
                pruned_model.add(layer)

        pruned_model.compile(
            optimizer=optimizer,
            loss=trained_model.loss, #from logits = Ture if no activation function on the output
            metrics=metrics
        )

        pruned_model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=100,verbose=2,validation_data=(x_val, y_val))
        print('\nPruned model. Finetuning: Test acuraccy ')
        pruned_model.evaluate(x_test, y_test, batch_size=batch_size)

        return pruned_model, hp_masc, ep_masc

    return trained_model, hp_masc, ep_masc
