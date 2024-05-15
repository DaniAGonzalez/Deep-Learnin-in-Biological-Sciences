# Daniela Alejandra Gonzalez

import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

# Creation of the model
model = dc.models.TensorGraph(batch_size=1000)
features = layers.Feature(shape=(None, 101, 4)) #Size of each sample 101 and 4 bases for each position
labels = layers.Label(shape=(None, 1))
weights = layers.Weights(shape=(None, 1)) #Necessary because unbalanced data

"Creation of stack of 3 convolutional layers"
# Width of convolutional kernels = 10
# Filters or outputs = 15
# To prevent overfitting, we add a dropout layer after each convolutional layer. The dropout probability is set to 0.5,
# meaning that 50% of all output values are randomly set to 0.
prev = features
for i in range(3):
    prev = layers.Conv1D(filters=15, kernel_size=10,
                             activation=tf.nn.relu, padding='same',
                             in_layers=prev)
    prev = layers.Dropout(dropout_prob=0.5, in_layers=prev)

"Creation of a dense layer to compute the outputs"
logits = layers.Dense(out_channels = 1, in_layers=layers.Flatten(prev))
outputs = layers.Sigmoid(logits) #To compress the Prob values to a desire range
model.add_output(output)

"Computation the cross entropy for each sample and multiply by the weightsto get the loss function"
loss = layers.SigmoidCrossEntropy(in_layers=[labels, logits])
weighted_loss = layers.WeigthedError(in_layers=[loss, weights])
model.set_loss(weighted_loss)

"Training the model and evaluating it"
train = dc.data.DiskDataset('train_dataset')
valid = dc.data.DiskDataset('valid_dataset')
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
for in range(20):
    model.fit(train, nb_epoch=10)
    print(model.evaluate(train, [metric]))
    print(model.evaluate(valid, [metric]))