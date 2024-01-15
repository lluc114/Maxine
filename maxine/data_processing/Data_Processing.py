#-------------------------------------------------------------------------------------------------------------------------------------------------------------
##############################################################################################################################################################
######################################### LIBRERIA: CONSTRUCCIÓN AUTOMÀTICA DE REDES NEURONALES NO CONVENCIONALES ############################################
################################# AUTORES: Lluc Josep Crespí Castanyer, Deva Murti Baer, Dr. Vicenç Canals Guinand############################################
#################################################################### VERSIÓN: v-01 ###########################################################################
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#%% SE CARGA LAS LIBRERIAS NECESARIAS PARA PROCESAR LOS DATASETS DE INTERÉS
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from keras.utils import np_utils
import requests
import os
import shutil
from tensorflow import keras
from keras import layers # mantiene memoria para poder hacer backpropagation
from keras.datasets import cifar10
from keras import activations
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam, SGD
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from tensorflow.python.ops.numpy_ops import np_config
from sklearn import preprocessing
np_config.enable_numpy_behavior()
import time

''' CADA UNA DE LAS FUNCIONES DEFINIDAS EN ESTE SCRIPT SE ENCARGA DE CARGAR, Y PREPARAR LOS DATOS DE UN DATASET
    PARA QUE ESTOS SEAN UTILIZADOS PARA ENTRENAR UNA RED NEURONAL '''

 # MNIST DATASET
def loadDataset_Mnist(labels,train_percentage,shuffle_files,output_size,padding_mode,normalization, layer_type):
   
    '''
    Args:
      labels: Can be 'integer' or 'hot'.
      train_percentage:  Sets the percentage of training data to be used for training and for validation
      shuffle_files: Can be TRUE or FALSE
      output_size: It is a tuple of two integers that sets the desired height and width of the images.
      normalization: A linear normalization of the data is performed. It is a tuple that can be (-1,1) or (1,1).
      layer_type: Can be 'conv' or 'norm'. If the data are to be used as input to a normal morphological neural network, the tensors are resized so that they are (bach, (height-width)
   
    '''
    # Set training and validation percentage of training data split
    train='train[0%:'+str(train_percentage)+'%]' # from 0 to train_percentage of the train split is train data, rest validation data
    validation='train['+str(train_percentage)+'%:]'
    (x_train,y_train),(x_test,y_test),(x_val,y_val)=tfds.as_numpy(tfds.
    load('mnist', # Nombre del dataset
    split=[train,'test',validation], # se divide en training test y validation
    batch_size=-1, 
    as_supervised=True, 
    shuffle_files=shuffle_files # se hace un shuffle para asegurar que los datos entran de forma aleatoria a la red
    ))
    print(' Initial MNIST Dataset Shape:')
    print('X_train: ' + str(x_train.shape))
    print('Y_train: ' + str(y_train.shape))
    print('X_test:  '  + str(x_test.shape))
    print('Y_test:  '  + str(y_test.shape))
    print('X_val:  '  + str(x_val.shape))
    print('Y_val:  '  + str(y_val.shape))
    # Image size
    image_size=(x_train.shape[1],x_train.shape[2])
    if output_size!=image_size:
        if output_size<image_size:
                raise ValueError("The desired output width and output height have to be equal or bigger than the original image width and height")
        print('Image size: (' + str(x_train.shape[1])+','+ str(x_train.shape[2]) +')')
        print('Wanted image size: '+ str(output_size))
        print('Required padding: Width: '+ str(output_size[0]-image_size[0])+ ' Height: '+ str(output_size[1]-image_size[1]))
    # Padding
        x_train=tf.image.resize_with_pad(x_train,output_size[0],output_size[1])
        x_test=tf.image.resize_with_pad(x_test,output_size[0],output_size[1])
        x_val=tf.image.resize_with_pad(x_val,output_size[0],output_size[1])
    # Datatype to float
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    x_val=x_val.astype('float32')
    # If data is introduced into a normal morphological layer, it is reshaped to (batch,(image_width·image_height))
    if layer_type=='norm':
        x_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
        x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
        x_val=x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2])
    # Working with integer or one hot codet labels
    if (labels=='hot'):
        nb_classes=10 # número de categorias que hay en el MNIST dataset
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        y_val=np_utils.to_categorical(y_val, nb_classes)
    # Linear data normalization: [0,1] o [-1,1]
    max_RGB=255
    min_RGB=0
    if normalization==(0,1):
        x_train=(x_train-min_RGB)/max_RGB
        x_test=(x_test-min_RGB)/max_RGB
        x_val=(x_val-min_RGB)/max_RGB
    if normalization==(-1,1):
        x_train=2*((x_train-min_RGB)/max_RGB)-1
        x_test=2*((x_test-min_RGB)/max_RGB)-1
        x_val=2*((x_val-min_RGB)/max_RGB)-1
    # Prepared dataset
    print("\n\t  Final Dataset")
    print("\n\t Training: ")
    print("\t\t ---> X_train: " + str (x_train.shape))  
    print("\t\t ---> Y_train: " + str (y_train.shape))
    print("\n\t Validation: ")
    print("\t\t ---> X_val: " + str (x_val.shape))  
    print("\t\t ---> Y_val: " + str (y_val.shape))
    print("\n\t Testing ")
    print("\t\t ---> X_test: " + str(x_test.shape))
    print("\t\t ---> Ttest: " + str(y_test.shape))
    # Output
    return x_train, y_train, x_test, y_test, x_val, y_val

# Test del MNIST 
# x_train, y_train, x_test, y_test, x_val,y_val=loadDataset_Mnist(labels='integer',train_percentage=90,shuffle_files=True,output_sze=(32,32),padding_mode='CONSTANT',normalization=(-1,1), layer_type='norm')

#%% FASHION MNIST
def loadDataset_FashionMnist(labels,train_percentage,shuffle_files,output_size,padding_mode,normalization, layer_type):
   
    '''
    Args:
      labels: Can be 'integer' or 'hot'.
      train_percentage:  Sets the percentage of training data to be used for training and for validation
      shuffle_files: Can be TRUE or FALSE
      output_size: It is a tuple of two integers that sets the desired height and width of the images.
      normalization: A linear normalization of the data is performed. It is a tuple that can be (-1,1) or (1,1).
      layer_type: Can be 'conv' or 'norm'. If the data are to be used as input to a normal morphological neural network, the tensors are resized so that they are (bach, (height-width)
   
    '''
    # Set training and validation percentage of training data split
    train='train[0%:'+str(train_percentage)+'%]' # from 0 to train_percentage of the train split is train data, rest validation data
    validation='train['+str(train_percentage)+'%:]'
    (x_train,y_train),(x_test,y_test),(x_val,y_val)=tfds.as_numpy(tfds.
    load('fashion_mnist', # Nombre del dataset
    split=[train,'test',validation], # se divide en training test y validation
    batch_size=-1, 
    as_supervised=True, 
    shuffle_files=shuffle_files # se hace un shuffle para asegurar que los datos entran de forma aleatoria a la red
    ))
    print(' Initial FASHION MNIST Dataset Shape:')
    print('X_train: ' + str(x_train.shape))
    print('Y_train: ' + str(y_train.shape))
    print('X_test:  '  + str(x_test.shape))
    print('Y_test:  '  + str(y_test.shape))
    print('X_val:  '  + str(x_val.shape))
    print('Y_val:  '  + str(y_val.shape))
    # Image size
    image_size=(x_train.shape[1],x_train.shape[2])
    if output_size!=image_size:
        if output_size<image_size:
                raise ValueError("The desired output width and output height have to be equal or bigger than the original image width and height")
        print('Image size: (' + str(x_train.shape[1])+','+ str(x_train.shape[2]) +')')
        print('Wanted image size: '+ str(output_size))
        print('Required padding: Width: '+ str(output_size[0]-image_size[0])+ ' Height: '+ str(output_size[1]-image_size[1]))
    # Padding
        x_train=tf.image.resize_with_pad(x_train,output_size[0],output_size[1])
        x_test=tf.image.resize_with_pad(x_test,output_size[0],output_size[1])
        x_val=tf.image.resize_with_pad(x_val,output_size[0],output_size[1])
    # Datatype to float
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    x_val=x_val.astype('float32')
    # If data is introduced into a normal morphological layer, it is reshaped to (batch,(image_width·image_height))
    if layer_type=='norm':
        x_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
        x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
        x_val=x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2])
    # Working with integer or one hot codet labels
    if (labels=='hot'):
        nb_classes=10 # número de categorias que hay en el MNIST dataset
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        y_val=np_utils.to_categorical(y_val, nb_classes)
    # Linear data normalization: [0,1] o [-1,1]
    max_RGB=255
    min_RGB=0
    if normalization==(0,1):
        x_train=(x_train-min_RGB)/max_RGB
        x_test=(x_test-min_RGB)/max_RGB
        x_val=(x_val-min_RGB)/max_RGB
    if normalization==(-1,1):
        x_train=2*((x_train-min_RGB)/max_RGB)-1
        x_test=2*((x_test-min_RGB)/max_RGB)-1
        x_val=2*((x_val-min_RGB)/max_RGB)-1
    # Prepared dataset
    print("\n\t  Final Dataset")
    print("\n\t Training: ")
    print("\t\t ---> X_train: " + str (x_train.shape))  
    print("\t\t ---> Y_train: " + str (y_train.shape))
    print("\n\t Validation: ")
    print("\t\t ---> X_val: " + str (x_val.shape))  
    print("\t\t ---> Y_val: " + str (y_val.shape))
    print("\n\t Testing ")
    print("\t\t ---> X_test: " + str(x_test.shape))
    print("\t\t ---> Ttest: " + str(y_test.shape))
    # Output
    return x_train, y_train, x_test, y_test, x_val, y_val
# TEST DEL FASHION MNIST
# x_train, y_train, x_test, y_test, x_val,y_val=loadDataset_FashionMnist(labels='integer',train_percentage=90,shuffle_files=True,output_size=(32,32),padding_mode='CONSTANT',normalization=(-1,1), layer_type='norm')

#%% CIFAR 10
def loadDataset_Cifar10(labels,train_percentage,shuffle_files,output_size,padding_mode,normalization, layer_type):
   
    '''
    Args:
      labels: Can be 'integer' or 'hot'.
      train_percentage:  Sets the percentage of training data to be used for training and for validation
      shuffle_files: Can be TRUE or FALSE
      output_size: It is a tuple of two integers that sets the desired height and width of the images.
      normalization: A linear normalization of the data is performed. It is a tuple that can be (-1,1) or (1,1).
      layer_type: Can be 'conv' or 'norm'. If the data are to be used as input to a normal morphological neural network, the tensors are resized so that they are (bach, (height-width)
   
    '''
    # Set training and validation percentage of training data split
    train='train[0%:'+str(train_percentage)+'%]' # from 0 to train_percentage of the train split is train data, rest validation data
    validation='train['+str(train_percentage)+'%:]'
    (x_train,y_train),(x_test,y_test),(x_val,y_val)=tfds.as_numpy(tfds.
    load('cifar10', # Nombre del dataset
    split=[train,'test',validation], # se divide en training test y validation
    batch_size=-1, 
    as_supervised=True, 
    shuffle_files=shuffle_files # se hace un shuffle para asegurar que los datos entran de forma aleatoria a la red
    ))
    print(' Initial CIFAR10 Dataset Shape:')
    print('X_train: ' + str(x_train.shape))
    print('Y_train: ' + str(y_train.shape))
    print('X_test:  '  + str(x_test.shape))
    print('Y_test:  '  + str(y_test.shape))
    print('X_val:  '  + str(x_val.shape))
    print('Y_val:  '  + str(y_val.shape))
    # Image size
    image_size=(x_train.shape[1],x_train.shape[2])
    if output_size!=image_size:
        if output_size<image_size:
                raise ValueError("The desired output width and output height have to be equal or bigger than the original image width and height")
        print('Image size: (' + str(x_train.shape[1])+','+ str(x_train.shape[2]) +')')
        print('Wanted image size: '+ str(output_size))
        print('Required padding: Width: '+ str(output_size[0]-image_size[0])+ ' Height: '+ str(output_size[1]-image_size[1]))
    # Padding
        x_train=tf.image.resize_with_pad(x_train,output_size[0],output_size[1])
        x_test=tf.image.resize_with_pad(x_test,output_size[0],output_size[1])
        x_val=tf.image.resize_with_pad(x_val,output_size[0],output_size[1])
    # Datatype to float
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    x_val=x_val.astype('float32')
    # If data is introduced into a normal morphological layer, it is reshaped to (batch,(image_width·image_height))
    if layer_type=='norm':
        x_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2],x_train.shape[3])
        x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2],x_test.shape[3])
        x_val=x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2],x_val.shape[3])
    # Working with integer or one hot codet labels
    if (labels=='hot'):
        nb_classes=10 # número de categorias que hay en el MNIST dataset
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        y_val=np_utils.to_categorical(y_val, nb_classes)
    # Linear data normalization: [0,1] o [-1,1]
    max_RGB=255
    min_RGB=0
    if normalization==(0,1):
        x_train=(x_train-min_RGB)/max_RGB
        x_test=(x_test-min_RGB)/max_RGB
        x_val=(x_val-min_RGB)/max_RGB
    if normalization==(-1,1):
        x_train=2*((x_train-min_RGB)/max_RGB)-1
        x_test=2*((x_test-min_RGB)/max_RGB)-1
        x_val=2*((x_val-min_RGB)/max_RGB)-1
    # Prepared dataset
    print("\n\t  Final Dataset")
    print("\n\t Training: ")
    print("\t\t ---> X_train: " + str (x_train.shape))  
    print("\t\t ---> Y_train: " + str (y_train.shape))
    print("\n\t Validation: ")
    print("\t\t ---> X_val: " + str (x_val.shape))  
    print("\t\t ---> Y_val: " + str (y_val.shape))
    print("\n\t Testing ")
    print("\t\t ---> X_test: " + str(x_test.shape))
    print("\t\t ---> Ttest: " + str(y_test.shape))
    # Output
    return x_train, y_train, x_test, y_test, x_val, y_val
#TEST
# x_train, y_train, x_test, y_test, x_val,y_val=loadDataset_Cifar10(labels='integer',train_percentage=90,shuffle_files=True,output_size=(32,32),padding_mode='CONSTANT',normalization=(-1,1), layer_type='norm')

#%% CIFAR 100 DATASET
def loadDataset_Cifar100(labels,train_percentage,shuffle_files,output_size,padding_mode,normalization, layer_type):
   
    '''
    Args:
      labels: Can be 'integer' or 'hot'.
      train_percentage:  Sets the percentage of training data to be used for training and for validation
      shuffle_files: Can be TRUE or FALSE
      output_size: It is a tuple of two integers that sets the desired height and width of the images.
      normalization: A linear normalization of the data is performed. It is a tuple that can be (-1,1) or (1,1).
      layer_type: Can be 'conv' or 'norm'. If the data are to be used as input to a normal morphological neural network, the tensors are resized so that they are (bach, (height-width)
   
    '''
    # Set training and validation percentage of training data split
    train='train[0%:'+str(train_percentage)+'%]' # from 0 to train_percentage of the train split is train data, rest validation data
    validation='train['+str(train_percentage)+'%:]'
    (x_train,y_train),(x_test,y_test),(x_val,y_val)=tfds.as_numpy(tfds.
    load('cifar100', # Nombre del dataset
    split=[train,'test',validation], # se divide en training test y validation
    batch_size=-1, 
    as_supervised=True, 
    shuffle_files=shuffle_files # se hace un shuffle para asegurar que los datos entran de forma aleatoria a la red
    ))
    print(' Initial CIFAR10 Dataset Shape:')
    print('X_train: ' + str(x_train.shape))
    print('Y_train: ' + str(y_train.shape))
    print('X_test:  '  + str(x_test.shape))
    print('Y_test:  '  + str(y_test.shape))
    print('X_val:  '  + str(x_val.shape))
    print('Y_val:  '  + str(y_val.shape))
    # Image size
    image_size=(x_train.shape[1],x_train.shape[2])
    if output_size!=image_size:
        if output_size<image_size:
                raise ValueError("The desired output width and output height have to be equal or bigger than the original image width and height")
        print('Image size: (' + str(x_train.shape[1])+','+ str(x_train.shape[2]) +')')
        print('Wanted image size: '+ str(output_size))
        print('Required padding: Width: '+ str(output_size[0]-image_size[0])+ ' Height: '+ str(output_size[1]-image_size[1]))
    # Padding
        x_train=tf.image.resize_with_pad(x_train,output_size[0],output_size[1])
        x_test=tf.image.resize_with_pad(x_test,output_size[0],output_size[1])
        x_val=tf.image.resize_with_pad(x_val,output_size[0],output_size[1])
    # Datatype to float
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    x_val=x_val.astype('float32')
    # If data is introduced into a normal morphological layer, it is reshaped to (batch,(image_width·image_height))
    if layer_type=='norm':
        x_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2],x_train.shape[3])
        x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2],x_test.shape[3])
        x_val=x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2],x_val.shape[3])
    # Working with integer or one hot codet labels
    if (labels=='hot'):
        nb_classes=100 # número de categorias que hay en el MNIST dataset
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        y_val=np_utils.to_categorical(y_val, nb_classes)
    # Linear data normalization: [0,1] o [-1,1]
    max_RGB=255
    min_RGB=0
    if normalization==(0,1):
        x_train=(x_train-min_RGB)/max_RGB
        x_test=(x_test-min_RGB)/max_RGB
        x_val=(x_val-min_RGB)/max_RGB
    if normalization==(-1,1):
        x_train=2*((x_train-min_RGB)/max_RGB)-1
        x_test=2*((x_test-min_RGB)/max_RGB)-1
        x_val=2*((x_val-min_RGB)/max_RGB)-1
    # Prepared dataset
    print("\n\t  Final Dataset")
    print("\n\t Training: ")
    print("\t\t ---> X_train: " + str (x_train.shape))  
    print("\t\t ---> Y_train: " + str (y_train.shape))
    print("\n\t Validation: ")
    print("\t\t ---> X_val: " + str (x_val.shape))  
    print("\t\t ---> Y_val: " + str (y_val.shape))
    print("\n\t Testing ")
    print("\t\t ---> X_test: " + str(x_test.shape))
    print("\t\t ---> Ttest: " + str(y_test.shape))
    # Output
    return x_train, y_train, x_test, y_test, x_val, y_val

# x_train, y_train, x_test, y_test, x_val,y_val=loadDataset_Cifar100(labels='hot',train_percentage=90,shuffle_files=True,output_size=(32,32),padding_mode='CONSTANT',normalization=(-1,1), layer_type='norm')

# IRIS DATASET
def loadDataset_Iris(labels,train_percentage,shuffle_files,output_size,padding_mode,normalization, layer_type):
   
    '''
    Args:
      labels: Can be 'integer' or 'hot'.
      train_percentage:  Sets the percentage of training data to be used for training and for validation
      shuffle_files: Can be TRUE or FALSE
      output_size: It is a tuple of two integers that sets the desired height and width of the images.
      normalization: A linear normalization of the data is performed. It is a tuple that can be (-1,1) or (1,1).
      layer_type: Can be 'conv' or 'norm'. If the data are to be used as input to a normal morphological neural network, the tensors are resized so that they are (bach, (height-width)
   
    '''
    # Set training and validation percentage of training data split
    train='train[0%:'+str(train_percentage)+'%]' # from 0 to train_percentage of the train split is train data, rest validation data
    test='train['+ str(train_percentage)+'%:'+ str(train_percentage+10)+'%]'
    validation='train['+str(train_percentage+10)+'%:]'
    (x_train,y_train),(x_test,y_test),(x_val,y_val)=tfds.as_numpy(tfds.
    load('iris', # Nombre del dataset
    split=[train,test,validation], # se divide en training test y validation
    batch_size=-1, 
    as_supervised=True, 
    shuffle_files=shuffle_files # se hace un shuffle para asegurar que los datos entran de forma aleatoria a la red
    ))
    print(' Initial Iris Dataset Shape:')
    print('X_train: ' + str(x_train.shape))
    print('Y_train: ' + str(y_train.shape))
    print('X_test:  '  + str(x_test.shape))
    print('Y_test:  '  + str(y_test.shape))
    print('X_val:  '  + str(x_val.shape))
    print('Y_val:  '  + str(y_val.shape))
    # Datatype to float
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    x_val=x_val.astype('float32')
    # Working with integer or one hot codet labels
    if (labels=='hot'):
        nb_classes=3 # número de categorias que hay en el MNIST dataset
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        y_val=np_utils.to_categorical(y_val, nb_classes)
    # Linear data normalization: [0,1] o [-1,1]
    max_RGB=255
    min_RGB=0
    if normalization==(0,1):
        x_train=(x_train-min_RGB)/max_RGB
        x_test=(x_test-min_RGB)/max_RGB
        x_val=(x_val-min_RGB)/max_RGB
    if normalization==(-1,1):
        x_train=2*((x_train-min_RGB)/max_RGB)-1
        x_test=2*((x_test-min_RGB)/max_RGB)-1
        x_val=2*((x_val-min_RGB)/max_RGB)-1
    # Prepared dataset
    print("\n\t  Final Dataset")
    print("\n\t Training: ")
    print("\t\t ---> X_train: " + str (x_train.shape))  
    print("\t\t ---> Y_train: " + str (y_train.shape))
    print("\n\t Validation: ")
    print("\t\t ---> X_val: " + str (x_val.shape))  
    print("\t\t ---> Y_val: " + str (y_val.shape))
    print("\n\t Testing ")
    print("\t\t ---> X_test: " + str(x_test.shape))
    print("\t\t ---> Ttest: " + str(y_test.shape))
    # Output
    return x_train, y_train, x_test, y_test, x_val, y_val

#x_train, y_train, x_test, y_test, x_val,y_val=loadDataset_Iris(labels='hot',train_percentage=80,shuffle_files=True,output_size=(32,32),padding_mode='CONSTANT',normalization=(0,1), layer_type='norm')

#%% 
# MONK'S 2 PROBLEM
def loadDataset_monks_2(train_percentage, shuffle_files, normalization, labels=""):
    path = "maxine/data_processing/monk"
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)
    
    print("Downloading data...")
    urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.test", "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.train"]
    paths = []
    for i, url in enumerate(urls):
        r = requests.get(url) # Response
        paths.append(os.path.join(path, url.split("/")[-1]) )
        open(paths[i], "wb").write(r.content)

    x_train = np.loadtxt(paths[1], usecols=range(7), dtype="float32")
    x_test = np.loadtxt(paths[0], usecols=range(7), dtype="float32")
    
    if shuffle_files:
        np.random.shuffle(x_train)
        np.random.shuffle(x_test)
    
    # Separate class labels
    y_train = x_train[:, 0]
    x_train = x_train[:, 1:]
    y_test = x_test[:, 0]
    x_test = x_test[:, 1:]

    """if normalization == (0, 1):
        x_train = x_train/4
        x_test = x_test/4
    elif normalization == (-1, 1):
        x_train = (x_train - 2)/2
        x_test = (x_test - 2)/2"""

    if normalization != None:
        _max = 4
        _min = 1
        _diff_array = _max - _min
        _diff_norm = normalization[1] - normalization[0]
        x_train = (x_train - _min)*_diff_norm/_diff_array + normalization[0]
        x_test = (x_test - _min)*_diff_norm/_diff_array + normalization[0]
        x_val = (x_val - _min)*_diff_norm/_diff_array + normalization[0]

    if (labels=='hot'):
        n_classes = 2 # Number of classes
        y_train = np_utils.to_categorical(y_train, n_classes)
        y_test = np_utils.to_categorical(y_test, n_classes)
        if train_percentage < 100:
            y_val=np_utils.to_categorical(y_val, n_classes)
    
    if train_percentage < 100:
        n_train = int((1-train_percentage/100)*x_train.shape[0])
        x_val = x_train[-n_train:]
        y_val = y_train[-n_train:]
        x_train = x_train[:-n_train]
        y_train = y_train[:-n_train]

        print("\n\t Final Dataset")
        print("\n\t Training: ")
        print("\t\t ---> X_train: " + str (x_train.shape))  
        print("\t\t ---> Y_train: " + str (y_train.shape))
        print("\n\t Validation: ")
        print("\t\t ---> X_val: " + str (x_val.shape))  
        print("\t\t ---> Y_val: " + str (y_val.shape))
        print("\n\t Testing ")
        print("\t\t ---> X_test: " + str(x_test.shape))
        print("\t\t ---> Y_test: " + str(y_test.shape))

        return x_train, y_train, x_test, y_test, x_val, y_val
    
    print("\n\t Final Dataset")
    print("\n\t Training: ")
    print("\t\t ---> X_train: " + str (x_train.shape))  
    print("\t\t ---> Y_train: " + str (y_train.shape))
    print("\n\t Testing ")
    print("\t\t ---> X_test: " + str(x_test.shape))
    print("\t\t ---> Y_test: " + str(y_test.shape))

    return x_train, y_train, x_test, y_test

    """# Deduce labels
    # MONK-2: EXACTLY TWO of {a1 = 1, a2 = 1, a3 = 1, a4 = 1, a5 = 1, a6 = 1}
    def get_labels(a):
        return 1*(a[1] == 1 & a[2] == 1 & a[3] == 1 & a[4] & a[5] == 1 & a[6] == 1)


    y_train = np.apply_along_axis(get_labels, 1, x_train)
    y_train = y_train.reshape((-1,1))
    y_test = np.apply_along_axis(get_labels, 1, x_test)
    y_test = y_test.reshape((-1,1))"""
    """if shuffle_files:
        _train = np.concatenate((x_train, y_train), axis=-1)
        _test = np.concatenate((x_test, y_test), axis=-1)

        np.random.shuffle(_train)
        np.random.shuffle(_test)

        x_train = _train[0:7]
        y_train = _train[-1:]
        x_test = _test[0:7]
        y_test = _test[-1:]"""

#%%
# Banknote authentication Data set
def loadDataset_banknote(train_percentage=90, test_percentage=10, shuffle_files=False, normalization=(0,1), labels=""):
    path = "maxine/data_processing/banknote"
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)
    
    print("Downloading data...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
  
    r = requests.get(url) # Response
    path = os.path.join(path, url.split("/")[-1])
    open(path, "wb").write(r.content)

    data = np.loadtxt(path, dtype="float32", delimiter=',')

    if shuffle_files:
        np.random.shuffle(data)

    # Separate class labels
    n_test = int((test_percentage/100)*data.shape[0]) # Number of Test instances
    n_training = int((train_percentage/100)*(data.shape[0]-n_test))
    x_train = data[:n_training, :4]
    y_train = data[:n_training, -1]
    x_val = data[n_training:-n_test, :4]
    y_val = data[n_training:-n_test, -1]
    x_test = data[-n_test:, :4]
    y_test = data[-n_test:, -1]

    if normalization != None:
        _max = np.amax(data[:, :4])
        _min = np.amin(data[:, :4])
        _diff_array = _max - _min
        _diff_norm = normalization[1] - normalization[0]
        x_train = (x_train - _min)*_diff_norm/_diff_array + normalization[0]
        x_test = (x_test - _min)*_diff_norm/_diff_array + normalization[0]
        x_val = (x_val - _min)*_diff_norm/_diff_array + normalization[0]
    
    """if normalization == (-1, 1):
        _max = np.amax(data[:, :4])
        _min = np.amin(data[:, :4])
        _diff = _max - _min
        x_train = (x_train - _min)/_diff
        x_test = (x_test - _min*2)/_diff
        x_val = (x_val - _min*2)/_diff"""

    if (labels=='hot'):
        n_classes = 2 # Number of classes
        y_train = np_utils.to_categorical(y_train, n_classes)
        y_test = np_utils.to_categorical(y_test, n_classes)
        y_val=np_utils.to_categorical(y_val, n_classes)

    if train_percentage < 100:
        print("\n\t Final Dataset")
        print("\n\t Training: ")
        print("\t\t ---> X_train: " + str (x_train.shape))  
        print("\t\t ---> Y_train: " + str (y_train.shape))
        print("\n\t Validation: ")
        print("\t\t ---> X_val: " + str (x_val.shape))  
        print("\t\t ---> Y_val: " + str (y_val.shape))
        print("\n\t Testing ")
        print("\t\t ---> X_test: " + str(x_test.shape))
        print("\t\t ---> Y_test: " + str(y_test.shape))

        return x_train, y_train, x_test, y_test, x_val, y_val
    
    else:
        print("\n\t Final Dataset")
        print("\n\t Training: ")
        print("\t\t ---> X_train: " + str (x_train.shape))  
        print("\t\t ---> Y_train: " + str (y_train.shape))
        print("\n\t Testing ")
        print("\t\t ---> X_test: " + str(x_test.shape))
        print("\t\t ---> Y_test: " + str(y_test.shape))

        return x_train, y_train, x_test, y_test

# Tic-Tac-Toe Endhame Data set
def loadDataset_TicTacToe(train_percentage=90, test_percentage=10, shuffle_files=False, normalization=(0,1), labels=""):
    path = "maxine/data_processing/TicTacToe"
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)
    
    print("Downloading data...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"
  
    r = requests.get(url) # Response
    path = os.path.join(path, url.split("/")[-1])
    open(path, "wb").write(r.content)

    print("The downloaded data is categorical. It will be transformed by the OneHot method.")

    data = np.loadtxt(path, dtype="str", delimiter=',')

    if shuffle_files:
        np.random.shuffle(data)

    enc = preprocessing.OneHotEncoder()
    data_enc = enc.fit_transform(data[:, :9]).toarray()

    enc_labels = preprocessing.OrdinalEncoder()
    labels = enc_labels.fit_transform(data[:,-1].reshape(-1, 1))

    # Separate class labels
    n_test = int((test_percentage/100)*data.shape[0]) # Number of Test instances
    n_training = int((train_percentage/100)*(data.shape[0]-n_test))
    x_train = data_enc[:n_training, :9]
    y_train = data_enc[:n_training, -1]
    x_val = data_enc[n_training:-n_test, :9]
    y_val = data_enc[n_training:-n_test, -1]
    x_test = data_enc[-n_test:, :9]
    y_test = data_enc[-n_test:, -1]

    if (labels=='hot'):
        n_classes = 2 # Number of classes
        y_train = np_utils.to_categorical(y_train, n_classes)
        y_test = np_utils.to_categorical(y_test, n_classes)
        if train_percentage < 100:
            y_val=np_utils.to_categorical(y_val, n_classes)
    
    if train_percentage < 100:
        print("\n\t Final Dataset")
        print("\n\t Training: ")
        print("\t\t ---> X_train: " + str (x_train.shape))  
        print("\t\t ---> Y_train: " + str (y_train.shape))
        print("\n\t Validation: ")
        print("\t\t ---> X_val: " + str (x_val.shape))  
        print("\t\t ---> Y_val: " + str (y_val.shape))
        print("\n\t Testing ")
        print("\t\t ---> X_test: " + str(x_test.shape))
        print("\t\t ---> Y_test: " + str(y_test.shape))

        return x_train, y_train, x_test, y_test, x_val, y_val
    
    else:
        print("\n\t Final Dataset")
        print("\n\t Training: ")
        print("\t\t ---> X_train: " + str (x_train.shape))  
        print("\t\t ---> Y_train: " + str (y_train.shape))
        print("\n\t Testing ")
        print("\t\t ---> X_test: " + str(x_test.shape))
        print("\t\t ---> Y_test: " + str(y_test.shape))

        return x_train, y_train, x_test, y_test


#%%
# Pen digits Data set
def loadDataset_penDigits(train_percentage=90, shuffle_files=False, normalization=(0,1), labels=""):
    path = "maxine/data_processing/penDigits"
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)

    print("Downloading data...")
    urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes", "http://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra"]
    paths = []
    for i, url in enumerate(urls):
        r = requests.get(url) # Response
        paths.append(os.path.join(path, url.split("/")[-1]) )
        open(paths[i], "wb").write(r.content)

    x_train = np.loadtxt(paths[1], dtype="float32", delimiter=',')
    x_test = np.loadtxt(paths[0], dtype="float32", delimiter=',')

    if shuffle_files:
        np.random.shuffle(x_train)
        np.random.shuffle(x_test)

    # Separate class labels
    n_test = int(((1 - train_percentage)/100)*(x_train.shape[0]))
    x_val = x_train[-n_test:, :16]
    y_val = x_train[-n_test:, -1]
    y_train = x_train[:-n_test, :-1]
    x_train = x_train[:-n_test, :16]
    y_test = x_test[:, -1]
    x_test = x_test[:, :16]
    
    if normalization != None:
        _max = max([np.amax(x_train[:, :16]), np.amax(x_test[:, :16])])
        _min = min([np.amin(x_train[:, :16]), np.amin(x_test[:, :16])])
        _diff_array = _max - _min
        _diff_norm = normalization[1] - normalization[0]
        x_train = (x_train - _min)*_diff_norm/_diff_array + normalization[0]
        x_test = (x_test - _min)*_diff_norm/_diff_array + normalization[0]
        x_val = (x_val - _min)*_diff_norm/_diff_array + normalization[0]

    if (labels=='hot'):
        n_classes = 9 # Number of classes
        y_train = np_utils.to_categorical(y_train, n_classes)
        y_test = np_utils.to_categorical(y_test, n_classes)
        y_val=np_utils.to_categorical(y_val, n_classes)

    if train_percentage < 100:
        print("\n\t Final Dataset")
        print("\n\t Training: ")
        print("\t\t ---> X_train: " + str (x_train.shape))  
        print("\t\t ---> Y_train: " + str (y_train.shape))
        print("\n\t Validation: ")
        print("\t\t ---> X_val: " + str (x_val.shape))  
        print("\t\t ---> Y_val: " + str (y_val.shape))
        print("\n\t Testing ")
        print("\t\t ---> X_test: " + str(x_test.shape))
        print("\t\t ---> Y_test: " + str(y_test.shape))

        return x_train, y_train, x_test, y_test, x_val, y_val
    
    else:
        print("\n\t Final Dataset")
        print("\n\t Training: ")
        print("\t\t ---> X_train: " + str (x_train.shape))  
        print("\t\t ---> Y_train: " + str (y_train.shape))
        print("\n\t Testing ")
        print("\t\t ---> X_test: " + str(x_test.shape))
        print("\t\t ---> Y_test: " + str(y_test.shape))

        return x_train, y_train, x_test, y_test
