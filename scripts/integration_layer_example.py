import sys
from functools import partial
import argparse
import time

import jax.numpy as jnp
import jax.random as random
from jax import vmap, jit
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

def net(params, layer, key, train, basis_layer, return_params=False):
    basis_channels = len(basis_layer[1])
    target_k = 1
    max_k = 4
    depth = 8
    num_conv_layers = 3

    for _ in range(num_conv_layers):
        layer, params = ml.batch_conv_layer(
            params,
            layer,
            { 'type': 'fixed', 'filters': conv_filters }, 
            depth, 
            max_k=max_k,
            mold_params=return_params,
        )
        layer = ml.batch_relu_layer(layer)

    layer, params = ml.batch_conv_layer(
        params,
        layer, 
        { 'type': 'fixed', 'filters': conv_filters }, 
        depth,
        target_k,
        mold_params=return_params,
    )
    
    layer = ml.batch_all_contractions(target_k, layer)
    layer, params = ml.batch_channel_collapse(params, layer, basis_channels, mold_params=return_params)
    recon_layer, coeffs = ml.batch_integration_layer(layer, basis_layer)

    return (recon_layer, coeffs, params) if return_params else (recon_layer, coeffs)

@partial(jit, static_argnums=4)
def map_and_loss(params, layer_x, layer_y, key, train, basis_layer, beta=1e-03):
    recon_layer, coeffs = net(params, layer_x, key, train, basis_layer)
    return jnp.mean(vmap(ml.rmse_loss)(recon_layer[1], layer_y[1])) + beta * jnp.mean(jnp.abs(coeffs))

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', help='number of epochs', type=float, default=200)
    parser.add_argument('-lr', '--learning_rate', help='learning rate', type=float, default=1e-4)
    parser.add_argument('-d', '--decay', help='learning rate decay', type=float, default=0.995)
    parser.add_argument('-b', '--batch', help='batch size', type=int, default=64)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='folder name to save the results', type=str, default=None)
    parser.add_argument('-l', '--load', help='folder name to load results from', type=str, default=None)
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)

    args = parser.parse_args()

    return (
        args.epochs,
        args.learning_rate,
        args.decay,
        args.batch,
        args.seed,
        args.save,
        args.load,
        args.verbose,
    )

# Main
epochs, lr, decay, batch_size, seed, save_folder, load_folder, verbose = handleArgs(sys.argv)

D = 2
N = 64 
num_images = 100

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
operators = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[1,2], parities=[0], D=D, operators=operators)

# Make basis library
img = geom.GeometricImage(jnp.ones((N,)*D + (D,)), 0, D) #just to get the shape
m = (img.N-1)/2 
x_y = (img.key_array() - m).reshape(img.shape())

x0 = jnp.stack([x_y[...,0], jnp.zeros((N,)*D)], axis=D)
x1 = x0[...,(1,0)]
y0 = jnp.stack([x_y[...,1], jnp.zeros((N,)*D)], axis=D)
y1 = y0[...,(1,0)]

basis_block_half = jnp.stack([
    jnp.stack([jnp.ones((N,)*D), jnp.zeros((N,)*D)], axis=D),
    x0,
    y0,
    x0**2,
    x0*y0,
    y0**2,
    x0**3,
    (x0**2)*y0,
    x0*(y0**2),
    y0**3,
])
basis_block = jnp.concatenate([basis_block_half, basis_block_half[...,(1,0)]])
basis_layer = geom.Layer({ 1: basis_block }, D)

# Get Training data
make_data = vmap(
    lambda basis,coeffs: jnp.sum(jnp.expand_dims(coeffs, axis=(1,2,3))*basis, axis=0, keepdims=True), 
    in_axes=(None,0),
)

key, subkey = random.split(key)
train_coeffs = random.normal(subkey, shape=(num_images, len(basis_block)))
X_train_data = make_data(basis_block, train_coeffs)
X_train = geom.BatchLayer({ 1: X_train_data }, D, False)

key, subkey = random.split(key)
X_val_data = make_data(basis_block, random.normal(subkey, shape=(batch_size,len(basis_block))))
X_val = geom.BatchLayer({ 1: X_val_data }, D, False)

# Get testing data.
key, subkey = random.split(key)
test_coeffs = random.normal(subkey, shape=(batch_size,len(basis_block)))
X_test_data = make_data(basis_block, test_coeffs)
X_test = geom.BatchLayer({ 1: X_test_data }, D, False)

key, subkey = random.split(key)
params = ml.init_params(partial(net, basis_layer=basis_layer), X_train.get_subset(jnp.array([0])), subkey)
print(f'Num Params: {ml.count_params(params)}')

key, subkey = random.split(key)
params, _, _ = ml.train(
    X_train,
    X_train, #reconstruction, so we want to get back to the input
    partial(map_and_loss, basis_layer=basis_layer, beta=0),
    params,
    subkey,
    ml.EpochStop(epochs, verbose=verbose),
    batch_size=batch_size,
    optimizer=optax.adam(optax.exponential_decay(
        lr, 
        transition_steps=int(X_train.L / batch_size), 
        decay_rate=decay,
    )),
    validation_X=X_val,
    validation_Y=X_val, # reconstruction, reuse X_val as the Y
)

recon_layer, out_coeffs = net(params, X_test, None, False, basis_layer)
print('coeffs mse:', ml.mse_loss(test_coeffs, out_coeffs))
print('losses:', jnp.mean(vmap(ml.rmse_loss)(recon_layer[1], X_test[1])), 1e-03 * jnp.mean(jnp.abs(out_coeffs)))