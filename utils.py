import os
import numpy as np
from sklearn.externals import joblib

import lasagne
from lasagne import layers as L
from model import build, network


def shuffle_ids(id_list, id_hash):
    return np.random.permutation(
        [id_hash[x] for x in id_list if x in id_hash])


def get_class_weight(y):
    """"""
    # if class weight...
    cw = sorted(Counter(y).items(), key=lambda x: x[0])
    return compute_class_weight(
        'balanced', map(lambda x: x[0], cw), y)


def save_check_point(network, params, train_id, path=None):
    """"""
    layers = L.get_all_layers(
        L.ConcatLayer(network.values(), axis=1))

    if path is None:
        path = os.getcwd()
    param_fn = os.path.join(path, str(train_id) + '.param')
    config_fn = os.path.join(path, str(train_id) + '.nnconfig.gz')

    np.savez(param_fn,
        *lasagne.layers.get_all_param_values(layers))
    joblib.dump(params, config_fn)


def load_check_point(train_id, path=None):
    """
    """
    if path is None:
        path = os.getcwd()

    param_fn = os.path.join(path, str(train_id) + '.param.npz')
    config_fn = os.path.join(path, str(train_id) + '.nnconfig.gz')
    params = joblib.load(config_fn)
    mdl, net = build(network(params), params)
    layers = L.get_all_layers(
        L.ConcatLayer(net.values(), axis=1))

    if os.path.exists(param_fn):
        try:
            print('Loadong pre-trained weight...')
            with np.load(param_fn) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            L.set_all_param_values(layers, param_values)
        except Exception as e:
            print(e)
            print('Cannot load parameters!')
    else:
        print('Cannot find parameters!')

    return net, mdl, params