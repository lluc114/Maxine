# Maxine
Morphological layers add-on for Tensorflow and Morphological Pruning Tools

## Pruning Tool Usage

baselineModel = tf.keras.models.load_model('./models/Lenet300100_Fashion')
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.975, nesterov=True)
pruned, pruned_hp1, pruned_ep1 = pruning_tool_MLP_v4_2.pruning_MLP(baselineModel, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=128, morph_epochs=30, mlp_epochs=10, mlp_epochs_refit=40, p=0.5, pList=[0.5, 0.05], optimizer_config=tf.optimizers.serialize(optimizer), extreme=True, preLoad=True, activLists=activLists, activCounts=activCounts)

baselineModel = tf.keras.models.load_model('./models/Lenet300100_Fashion')
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05, nesterov=True)
pruned, pruned_hp1, pruned_ep1 = pruning_tool_MLP_v4_2.pruning_MLP(baselineModel, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=128, morph_epochs=30, mlp_epochs=10, mlp_epochs_refit=40, p=0.5, optimizer_config=tf.optimizers.serialize(optimizer), extreme=True, preLoad=False, activLists=activLists, activCounts=activCounts)
