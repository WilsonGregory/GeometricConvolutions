# generate gravitational field
import numpy as np
import sys
from functools import partial
import argparse
import time

import jax.numpy as jnp
import jax.random as random
from jax import vmap, jit, checkpoint
import jax
from jax.ad_checkpoint import print_saved_residuals, checkpoint_name
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml


def net(params, layer, key, train, conv_filters, return_params=False):
    depth = 1
    num_conv_layers = 3

    for _ in range(num_conv_layers):
        layer, params = ml.batch_conv_contract(
            params,
            layer,
            conv_filters,
            depth,
            ((0, 0), (2, 0)),
            mold_params=return_params,
        )
        layer = ml.scalar_activation(layer, partial(jax.nn.leaky_relu, negative_slope=0.01))

    layer, params = ml.batch_conv_contract(
        params,
        layer,
        conv_filters,
        depth,
        ((2, 0),),
        mold_params=return_params,
    )

    layer, params = ml.batch_channel_collapse(params, layer, mold_params=return_params)

    return (layer, params) if return_params else layer


@partial(jit, static_argnums=4)
def map_and_loss(params, layer_x, layer_y, key, train, conv_filters):
    learned_x = net(params, layer_x, key, train, conv_filters)
    return ml.rmse_loss(learned_x[(2, 0)], layer_y[(2, 0)])


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="number of epochs", type=float, default=10)
    parser.add_argument("-lr", help="learning rate", type=float, default=0.01)
    parser.add_argument("-batch", help="batch size", type=int, default=1)
    parser.add_argument("-seed", help="the random number seed", type=int, default=None)
    parser.add_argument(
        "-s", "--save", help="folder name to save the results", type=str, default=None
    )
    parser.add_argument(
        "-l", "--load", help="folder name to load results from", type=str, default=None
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="levels of print statements during training",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    return (
        args.epochs,
        args.lr,
        args.batch,
        args.seed,
        args.save,
        args.load,
        args.verbose,
    )


# Main
epochs, lr, batch_size, seed, save_folder, load_folder, verbose = handleArgs(sys.argv)

D = 3
is_torus = False

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
operators = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3], ks=[0, 1, 2, 3, 4], parities=[0], D=D, operators=operators
)

# Get Training data
data = np.load("../data/3d_turb/downsampled_1024_to_64.npz")

rho = jnp.array(data["rho"])
vel = jnp.stack([data["vel1"], data["vel2"], data["vel3"]], axis=-1)
raw_tau = jnp.stack(
    [
        data["T11"],
        data["T12"],
        data["T13"],
        data["T12"],
        data["T22"],
        data["T23"],
        data["T13"],
        data["T23"],
        data["T33"],
    ],
    axis=-1,
)
tau = raw_tau.reshape(raw_tau.shape[:3] + (D, D))

layer_x = geom.BatchLayer(
    {
        (0, 0): jnp.expand_dims(rho, axis=(0, 1)),
        (1, 0): jnp.expand_dims(vel, axis=(0, 1)),
    },
    D,
    is_torus,
)
layer_y = geom.BatchLayer({(2, 0): jnp.expand_dims(tau, axis=(0, 1))}, D, is_torus)

key, subkey = random.split(key)
params = ml.init_params(partial(net, conv_filters=conv_filters), layer_x, subkey)
print(f"Model params: {ml.count_params(params)}")

# chkpt_func = checkpoint(
#     jax.value_and_grad(partial(map_and_loss, conv_filters=conv_filters)),
#     policy=jax.checkpoint_policies.nothing_saveable,
#     static_argnums=4,
# )
# X_batches, Y_batches = ml.get_batch_layer(layer_x, layer_y, 1, subkey)
# X_batch = X_batches[0]
# Y_batch = Y_batches[0]
# print_saved_residuals(chkpt_func, params, X_batch, Y_batch, key, True)

del data
del raw_tau

if load_folder:
    params = jnp.load(f"{load_folder}/params_{seed}.npy", allow_pickle=True).item()
else:
    key, subkey = random.split(key)
    params, _, _ = ml.train(
        layer_x,
        layer_y,
        partial(map_and_loss, conv_filters=conv_filters),
        params,
        subkey,
        ml.EpochStop(epochs=epochs, verbose=verbose),
        batch_size=1,
        optimizer=optax.adam(optax.exponential_decay(lr, transition_steps=1, decay_rate=0.999)),
        checkpoint_kwargs={
            "policy": jax.checkpoint_policies.nothing_saveable,
            "static_argnums": 4,
        },
    )

    if save_folder:
        jnp.save(f"{save_folder}/params_{seed}.npy", params, allow_pickle=True)

key, subkey = random.split(key)
net_out = net(params, layer_x, subkey, False, conv_filters)
