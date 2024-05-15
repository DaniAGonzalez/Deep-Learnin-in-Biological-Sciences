"""Creation of a model to predict binding site for siRNA"""


model = dc.models.TensorGraph()
features = layers.Feature(shape=(None, 21, 4)) labels = layers.Label(shape=(None, 1))
prev = features
for i in range(2):
      prev = layers.Conv1D(filters=10, kernel_size=10,
                           activation=tf.nn.relu, padding='same',
                           in_layers=prev)
      prev = layers.Dropout(dropout_prob=0.3, in_layers=prev)
    output = layers.Dense(out_channels=1, activation_fn=tf.sigmoid,
                          in_layers=layers.Flatten(prev))
    model.add_output(output)
    loss = layers.ReduceMean(layers.L2Loss(in_layers=[labels, output]))
    model.set_loss(loss)

"Train the model"
train = dc.data.DiskDataset('train_siRNA')
valid = dc.data.DiskDataset('valid_siRNA')
metric = dc.metrics.Metric(dc.metrics.pearsonr, mode='regression')
for i in range(20):
    model.fit(train, nb_epoch=10)
    print(model.evaluate(train, [metric]))
    print(model.evaluate(valid, [metric]))