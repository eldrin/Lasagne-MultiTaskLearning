from functools import partial
from collections import Counter

import theano
from theano import tensor as T

import lasagne
from lasagne import layers as L
from lasagne import nonlinearities as nl
from lasagne import objectives as obj
from lasagne import updates
from lasagne import regularization as reg

from sklearn.utils.class_weight import compute_class_weight

from networks import deep_cnn_2d_mtl


def network(params):
    """"""
    return deep_cnn_2d_mtl(params)


def build(layer_heads, params):
    """"""
    fns = {}  # model methods
    x = T.tensor4('input')

    for target in params['targets']:
        fns[target['name']] = {}
        out_layer = layer_heads[target['name']]

        y = T.matrix('target')
        o = L.get_output(out_layer, inputs=x)
        o_vl = L.get_output(out_layer, inputs=x, deterministic=True)

        if 'class_weight' in params and params['class_weight']:
            loss_fn = partial(weighted_cce, weights=params['class_weight'])
        else:
            loss_fn = obj.categorical_crossentropy

        loss = loss_fn(o, y).mean()
        loss_vl = loss_fn(o_vl, y).mean()
        wd_l2 = params['beta'] * reg.regularize_network_params(out_layer, reg.l2)

        acc = obj.categorical_accuracy(o, y).mean()
        acc_vl = obj.categorical_accuracy(o_vl, y).mean()

        updates_ = updates.adam(
            loss + wd_l2, L.get_all_params(out_layer, trainable=True),
            learning_rate=params['learning_rate'], epsilon=params['epsilon'])

        fns[target['name']]['train'] = theano.function(
            [x, y], updates=updates_, allow_input_downcast=True)
        fns[target['name']]['predict'] = theano.function(
            [x], o_vl, allow_input_downcast=True)
        fns[target['name']]['cost'] = theano.function(
            [x, y], loss_vl, allow_input_downcast=True)
        fns[target['name']]['acc'] = theano.function(
            [x, y], acc_vl, allow_input_downcast=True)
        fns[target['name']]['transform'] = theano.function(
            [x],
            L.get_output(
                L.get_all_layers(layer_heads[target['name']])[-2],
                inputs=x, deterministic=True
            ),
            allow_input_downcast=True)

    return fns, layer_heads


def weighted_cce(predictions, targets, weights):
    """"""
    return -T.sum(weights[None, :] * targets * T.log(predictions.clip(1e-10, 1.)),
                  axis=predictions.ndim - 1)
