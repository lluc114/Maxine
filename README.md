# Maxine
Morphological layers add-on for Tensorflow and Morphological Pruning Tools

## Pruning Tool Usage
<b>Arguments:</b>
<dl>
  <dt> baselineMode </dt>
  <dt> x_train </dt>  
  <dt> y_train </dt>
  <dt> x_val </dt>
  <dt> y_val </dt>
  <dt> x_test </dt>
  <dt> y_test </dt>
  <dt> batch_size </dt>
  <dt> morph_epochs: </dt>
    <dd> number of epochs to train the morphological model </dd>
  <dt> mlp_epochs: </dt>
    <dd> number of epochs to train the model when pruning is done </dd>
  <dt> mlp_epochs_refit: </dt>
    <dd> number of epochs to train the model when a pruing masc is applied to one of its layers </dd>
  <dt> p: </dt>
    <dd> minim usability parameter (%). Ignored when extreme=False </dd>
  <dt> pList: </dt>
    <dd> list of <i>p</i> parameters, one for each layer, except for the output layer that remains unpruned. Ignored when extreme=False </dd>
  <dt> optimizer_config: </dt>
    <dd> the serialized optimizer for training </dd>
  <dt> preLoad: </dt>
    <dd> if True you can skip the morphological training </dd>
  <dt> activLists: </dt>
    <dd> list of activations (true/false) (MxQ) for each layer of the morphological model. Ignored when preLoad=False </dd>
  <dt> activCounts: </dt> 
    <dd> list of number of times that each weight has been used (MxQ) for each layer of the morphological model. Ignored when preLoad=False </dd>
</dl>

<b>Example:</b>
To use the pruning tool with a different minimum usability <i>p</i> for each layer without preload of activations of the morph. model. When passing a pList of length > 1, <i>p</i> is ignored. 
```
baselineModel = tf.keras.models.load_model('./models/Lenet300100_Fashion')
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.975, nesterov=True)
pruned, pruned_hp1, pruned_ep1 = pruning_tool_MLP_v4_2.pruning_MLP(baselineModel, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=128, morph_epochs=30, mlp_epochs=10, mlp_epochs_refit=40, p=0.5, pList=[0.5, 0.05], optimizer_config=tf.optimizers.serialize(optimizer), extreme=True, preLoad=True, activLists=[], activCounts=[])
```
<b>Result:</b>
```
...
Accuracy of the MLP model:
313/313 [==============================] - 2s 3ms/step - loss: 0.3754 - accuracy: 0.8659
# Parameters of the model: 266610
# Parameters pruned: 261987
# Parameters remaining: 4623
Remaining weights (%): 1.7339934736131428
```
