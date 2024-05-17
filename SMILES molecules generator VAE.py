import deepchem as dc
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.backend import random_normal
from rdkit import Chem

# Loading training data
tasks, datasets, transformers = dc.molnet.load_muv()
train_dataset, valid_dataset, test_dataset = datasets
train_smiles = train_dataset.ids

# Number of tokens and maximun length of SMILES
tokens = set()
for s in train_smiles:
    tokens = tokens.union(set(s))
tokens = sorted(list(tokens))
max_length = max(len(s) for s in train_smiles)

# autoencoder dimensions
input_dim = max_length
num_tokens = len(tokens)
encoding_dim = 32
latent_dim = 196


def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# VAE architecture
inputs = Input(shape=(input_dim, num_tokens), name='encoder_input')
flattened_inputs = Flatten()(inputs)
x = Dense(encoding_dim, activation='relu')(flattened_inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(encoding_dim, activation='relu')(latent_inputs)
x = Dense(input_dim * num_tokens, activation='sigmoid')(x)
x = Reshape((input_dim, num_tokens))(x)
outputs = Softmax(axis=-1)(x)

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

outputs = decoder(encoder(inputs)[2])

# Loss function VAE
class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    def vae_loss(self, inputs, outputs, z_mean, z_log_var):
        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= input_dim * num_tokens
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5
        return tf.reduce_mean(reconstruction_loss + kl_loss)

    def call(self, inputs):
        inputs, outputs, z_mean, z_log_var = inputs
        loss = self.vae_loss(inputs, outputs, z_mean, z_log_var)
        self.add_loss(loss)
        return outputs

vae_loss_layer = VAELossLayer()([inputs, outputs, z_mean, z_log_var])

vae = Model(inputs, vae_loss_layer, name='vae')
vae.compile(optimizer='adam')
vae.summary()


def smiles_to_one_hot(smiles, max_length, tokens):
    token_to_idx = {token: idx for idx, token in enumerate(tokens)}
    one_hot_encoded = np.zeros((len(smiles), max_length, len(tokens)), dtype=np.float32)
    for i, smile in enumerate(smiles):
        for j, token in enumerate(smile):
            one_hot_encoded[i, j, token_to_idx[token]] = 1
    return one_hot_encoded

train_data = smiles_to_one_hot(train_smiles, max_length, tokens)

# Trainning the model
vae.fit(train_data, train_data, epochs=50, batch_size=100, shuffle=True)

# New generated molecules
predictions = decoder.predict(np.random.normal(size=(1000, latent_dim)))

def one_hot_to_smiles(one_hot, tokens):
    smiles = []
    for vec in one_hot:
        smile = ''.join(tokens[np.argmax(char)] for char in vec)
        smiles.append(smile)
    return smiles

generated_smiles = one_hot_to_smiles(predictions, tokens)

# Filtering valid smiles
valid_smiles = []
for smile in generated_smiles:
    if Chem.MolFromSmiles(smile) is not None:
        valid_smiles.append(smile)

print()
print('Generated valid molecules:')
for m in valid_smiles:
    print(m)
