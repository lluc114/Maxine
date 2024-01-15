# %%
import tensorflow as tf
print(tf.__version__)

# Helper libraries
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

#My classes
from maxine.layers import MaxMinPool

# %%
MaxMinPool

# %%
tf.keras.layers.Flatten

# %%
tf.keras.layers.Dense

# %%
tf.keras.datasets.fashion_mnist

# %%
MaxMinPool

# %%
MaxMinPool.keras

# %%
MaxMinPool.keras_export

# %%
import tensorflow as tf
print(tf.__version__)

# Helper libraries
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

#My classes
from maxine.layers import MaxMinPool

# %%
MaxMinPool

# %% 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    MaxMinPool(units=128, M=60, activation='linear'),
    #tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy']) #from logits = Ture if no activation function on the output

# %%
import tensorflow as tf
print(tf.__version__)

# Helper libraries
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

#My classes
from maxine.layers.MaxMinPool import MaxMinPool

# %% 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    MaxMinPool(units=128, M=60, activation='linear'),
    #tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy']) #from logits = Ture if no activation function on the output

# %%
import my_layer

# %%
my_layer.Linear

# %%
import module1

# %%
module1.function()

# %%
import tensorflow as tf
print(tf.__version__)

# Helper libraries
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

#My classes
from maxine.layers.MaxMinPool import MaxMinPool

# %%
import tensorflow as tf
print(tf.__version__)

# Helper libraries
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

#My classes
#from maxine.layers.MaxMinPool import MaxMinPool

# %%
import maxine.layers

# %%
maxine.layers.MaxMinPool

# %%
import tensorflow as tf
print(tf.__version__)

# Helper libraries
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import maxine

#My classes
#from maxine.layers.MaxMinPool import MaxMinPool

# %% 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    maxine.MaxMinPool(units=128, M=60, activation='linear'),
    #tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy']) #from logits = Ture if no activation function on the output

# %%
import tensorflow as tf
print(tf.__version__)

# Helper libraries
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import maxine

#My classes
#from maxine.layers.MaxMinPool import MaxMinPool

# %% 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    maxine.MaxMinPool(units=128, M=60, activation='linear'),
    #tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy']) #from logits = Ture if no activation function on the output

# %%
import tensorflow as tf
print(tf.__version__)

# Helper libraries
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import maxine

#My classes
#from maxine.layers.MaxMinPool import MaxMinPool

# %% 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    maxine.layers.MaxMinPool(units=128, M=60, activation='linear'),
    #tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy']) #from logits = Ture if no activation function on the output

# %%
maxine

# %%
maxine.layers

# %%
maxine.layers.MaxMinPool

# %%
maxine.layers.MaxMinPool.MaxMinPool

# %%
import tensorflow as tf
print(tf.__version__)

# Helper libraries
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import maxine

#My classes
#from maxine.layers.MaxMinPool import MaxMinPool

# %%
maxine.layers.MaxMinPool

# %%
maxine.layers

# %%
import maxine

# %%
maxine.layers.MaxMinPool

# %%
maxine.layers

# %%
import maxine

# %%
maxine.layers.MaxMinPool

# %%
import maxine

# %%
import maxine

# %%
maxine.layers

# %%
maxine.layers.MaxMinPool


