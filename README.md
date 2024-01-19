# Maxine
Morphological layers add-on for Tensorflow and Morphological Pruning Tools

## Pruning Tool Usage
Arguments:
baselineMode
x_train
y_train
x_val
y_val
x_test
y_test
batch_size
morph_epochs: number of epochs to train the morphological model
mlp_epochs: number of epochs to train the model when pruning is done
mlp_epochs_refit: number of epochs to train the model when a pruing masc is applied to one of its layers
p: minim usability parameter (%). Ignored when extreme=False
pList: list of p parameters, one for each layer, except for the output layer that remains unpruned. Ignored when extreme=False
optimizer_config: the serialized optimizer for training
preLoad: if True you can skip the morphological training
activLists: list of activations (true/false) (MxQ) for each layer of the morphological model. Ignored when preLoad=False
activCounts: list of number of times that each weight has been used (MxQ) for each layer of the morphological model. Ignored when preLoad=False

To use the pruning tool with a different minimum usability p for each layer 
```
baselineModel = tf.keras.models.load_model('./models/Lenet300100_Fashion')
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.975, nesterov=True)
pruned, pruned_hp1, pruned_ep1 = pruning_tool_MLP_v4_2.pruning_MLP(baselineModel, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=128, morph_epochs=30, mlp_epochs=10, mlp_epochs_refit=40, p=0.5, pList=[0.5, 0.05], optimizer_config=tf.optimizers.serialize(optimizer), extreme=True, preLoad=True, activLists=activLists, activCounts=activCounts)
```

baselineModel = tf.keras.models.load_model('./models/Lenet300100_Fashion')
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05, nesterov=True)
pruned, pruned_hp1, pruned_ep1 = pruning_tool_MLP_v4_2.pruning_MLP(baselineModel, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=128, morph_epochs=30, mlp_epochs=10, mlp_epochs_refit=40, p=0.5, optimizer_config=tf.optimizers.serialize(optimizer), extreme=True, preLoad=False, activLists=activLists, activCounts=activCounts)
