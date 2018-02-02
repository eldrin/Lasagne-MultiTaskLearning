from collections import OrderedDict
from lasagne import layers as L
from lasagne import nonlinearities as nl
from sklearn.externals import joblib
import numpy as np


def swish(x):
    """"""
    return x * nl.sigmoid(x)


def deep_cnn_2d_vanilla(params):
    """"""
    nonlin = nl.rectify

    layers = L.InputLayer((None, 1, params['dur'], 128))
    print layers.output_shape

    sclr = joblib.load(params['scaler'])
    layers = L.standardize(
        layers, sclr.mean_.astype(np.float32),
        sclr.scale_.astype(np.float32), shared_axes=(0, 1, 2))
    print layers.output_shape

    n_filter = [16, 32, 64, 64, 128, 128, 256]  # l
    filter_sz = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (1, 1)]  # m
    if params['dur'] > 50:
        conv_strd = [(2, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]  # c
        pool_sz = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), None, None]  # n
    else:
        conv_strd = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]  # c
        pool_sz = [(1, 2), (2, 2), (2, 2), (2, 2), (2, 2), None, None]  # n
    pool_strd = [None, None, None, None, None, None, None]  # s
    batch_norm = [False, False, False, False, False, False, False]  # b
    dropout = [False, False, False, False, False, False, False]  # d # added
    conv_spec = zip(
        n_filter, filter_sz, conv_strd, pool_sz,
        pool_strd, batch_norm, dropout
    )

    for l, m, c, n, s, b, d in conv_spec:
        if b:
            layers = L.batch_norm(
                L.Conv2DLayer(
                    layers, l, m, stride=c,
                    pad='same', nonlinearity=nonlin),
            )
        else:
            layers = L.Conv2DLayer(
                layers, l, m, stride=c,
                pad='same', nonlinearity=nonlin
            )
        if n is not None:
            layers = L.MaxPool2DLayer(layers, pool_size=n, stride=s)

        if d:
            layers = L.dropout(layers, p=0.1)

        print layers.output_shape

    layers = L.GlobalPoolLayer(layers)
    print layers.output_shape

    layers = L.DenseLayer(layers, 256, nonlinearity=nonlin)
    print layers.output_shape

    layers = L.DenseLayer(layers, 16, nonlinearity=nl.softmax)
    print layers.output_shape

    layer_head = {}
    layer_head['tg'] = layers
    return layer_head


def deep_cnn_2d(params):
    """"""
    nonlin = nl.elu

    layers = L.InputLayer((None, 1, params['dur'], 128))
    print layers.output_shape

    sclr = joblib.load(params['scaler'])
    layers = L.standardize(
        layers, sclr.mean_.astype(np.float32),
        sclr.scale_.astype(np.float32), shared_axes=(0, 1, 2))
    print layers.output_shape

    n_filter = [16, 32, 64, 64, 128, 256, 256]  # l
    filter_sz = [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (1, 1)]  # m
    if params['dur'] > 50:
        conv_strd = [(2, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]  # c
        pool_sz = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), None, None]  # n
    else:
        conv_strd = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]  # c
        pool_sz = [(1, 2), (2, 2), (2, 2), (2, 2), (2, 2), None, None]  # n
    pool_strd = [None, None, None, None, None, None, None]  # s
    batch_norm = [False, True, False, True, False, False, False]  # b
    dropout = [True, True, False, True, False, False, False]  # d # added
    conv_spec = zip(
        n_filter, filter_sz, conv_strd, pool_sz,
        pool_strd, batch_norm, dropout
    )

    for l, m, c, n, s, b, d in conv_spec:
        if b:
            layers = L.batch_norm(
                L.Conv2DLayer(
                    layers, l, m, stride=c,
                    pad='same', nonlinearity=nonlin),
            )
        else:
            layers = L.Conv2DLayer(
                layers, l, m, stride=c,
                pad='same', nonlinearity=nonlin
            )
        if n is not None:
            layers = L.MaxPool2DLayer(layers, pool_size=n, stride=s)

        if d:
            layers = L.dropout(layers, p=0.1)

        print layers.output_shape

    layers = L.batch_norm(L.GlobalPoolLayer(layers))
    layers = L.dropout(layers)  # added
    print layers.output_shape

    layers = L.batch_norm(
        L.DenseLayer(layers, 256, nonlinearity=nonlin))
    layers = L.dropout(layers)
    print layers.output_shape

    layers = L.DenseLayer(layers, 16, nonlinearity=nl.softmax)
    print layers.output_shape

    return layers


def deep_cnn_2d_mtl_at_fc(params):
    """"""
    assert 'targets' in params
    nonlin = nl.elu

    layers = L.InputLayer((None, 1, params['dur'], 128))
    print layers.output_shape

    sclr = joblib.load(params['scaler'])
    layers = L.standardize(
        layers, sclr.mean_.astype(np.float32),
        sclr.scale_.astype(np.float32), shared_axes=(0, 1, 2))
    print layers.output_shape

    n_filter = [16, 32, 64, 64, 128, 256, 256]  # l
    filter_sz = [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (1, 1)]  # m
    if params['dur'] > 50:
        conv_strd = [(2, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]  # c
        pool_sz = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), None, None]  # n
    else:
        conv_strd = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]  # c
        pool_sz = [(1, 2), (2, 2), (2, 2), (2, 2), (2, 2), None, None]  # n
    pool_strd = [None, None, None, None, None, None, None]  # s
    batch_norm = [False, True, False, True, False, False, False]  # b
    dropout = [True, True, False, True, False, False, False]  # d # added
    conv_spec = zip(
        n_filter, filter_sz, conv_strd, pool_sz,
        pool_strd, batch_norm, dropout
    )

    for l, m, c, n, s, b, d in conv_spec:
        if b:
            layers = L.batch_norm(
                L.Conv2DLayer(
                    layers, l, m, stride=c,
                    pad='same', nonlinearity=nonlin),
            )
        else:
            layers = L.Conv2DLayer(
                layers, l, m, stride=c,
                pad='same', nonlinearity=nonlin
            )
        if n is not None:
            layers = L.MaxPool2DLayer(layers, pool_size=n, stride=s)

        if d:
            layers = L.dropout(layers, p=0.1)

        print layers.output_shape

    layers = L.batch_norm(L.GlobalPoolLayer(layers))
    layers = L.dropout(layers)  # added
    print layers.output_shape

    layers = L.batch_norm(
        L.DenseLayer(layers, 256, nonlinearity=nonlin))
    layers = L.dropout(layers)
    print layers.output_shape

    layer_heads = OrderedDict()
    for target in params['targets']:
        layer_heads[target['name']] = L.DenseLayer(
            layers, target['n_out'], nonlinearity=nl.softmax)
        print target['name'], layer_heads[target['name']].output_shape

    return layer_heads


def deep_cnn_2d_mtl_at_2(params):
    """"""
    assert 'targets' in params
    nonlin = nl.elu

    layers = L.InputLayer((None, 1, params['dur'], 128))
    print layers.output_shape

    sclr = joblib.load(params['scaler'])
    layers = L.standardize(
        layers, sclr.mean_.astype(np.float32),
        sclr.scale_.astype(np.float32), shared_axes=(0, 1, 2))
    print layers.output_shape

    n_filter = [32, 64, 64, 128, 256, 256]  # l
    filter_sz = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (1, 1)]  # m
    if params['dur'] > 50:
        conv_strd = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]  # c
        pool_sz = [(2, 2), (2, 2), (2, 2), (2, 2), None, None]  # n
    else:
        conv_strd = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]  # c
        pool_sz = [(2, 2), (2, 2), (2, 2), (2, 2), None, None]  # n
    pool_strd = [None, None, None, None, None, None]  # s
    batch_norm = [True, False, True, False, False, False]  # b
    dropout = [True, False, True, False, False, False]  # d # added
    conv_spec = zip(
        n_filter, filter_sz, conv_strd, pool_sz,
        pool_strd, batch_norm, dropout
    )

    # Shared first layer
    layers = L.Conv2DLayer(layers, 16, (5, 5), stride=(1, 1),
                           pad='same', nonlinearity=nonlin)
    layers = L.MaxPool2DLayer(layers, pool_size=(1, 2))
    layers = L.dropout(layers, p=0.1)

    layer_heads = OrderedDict()
    for target in params['targets']:
        first_trgt_spec_layer = True  # n_layer checker
        for l, m, c, n, s, b, d in conv_spec:
            if first_trgt_spec_layer:
                layer_heads[target['name']] = layers
                first_trgt_spec_layer = False
            if b:
                layer_heads[target['name']] = L.batch_norm(
                    L.Conv2DLayer(
                        layer_heads[target['name']], l, m, stride=c,
                        pad='same', nonlinearity=nonlin),
                )
            else:
                layer_heads[target['name']] = L.Conv2DLayer(
                    layer_heads[target['name']], l, m, stride=c,
                    pad='same', nonlinearity=nonlin
                )
            if n is not None:
                layer_heads[target['name']] = L.MaxPool2DLayer(
                    layer_heads[target['name']], pool_size=n, stride=s)

            if d:
                layer_heads[target['name']] = L.dropout(
                    layer_heads[target['name']], p=0.1)
            print layer_heads[target['name']].output_shape

        layer_heads[target['name']] = L.batch_norm(
            L.GlobalPoolLayer(layer_heads[target['name']]))
        layer_heads[target['name']] = L.dropout(
            layer_heads[target['name']])  # added
        print layer_heads[target['name']].output_shape

        layer_heads[target['name']] = L.batch_norm(
            L.DenseLayer(
                layer_heads[target['name']],
                256, nonlinearity=nonlin))
        layer_heads[target['name']] = L.dropout(
            layer_heads[target['name']])
        print layer_heads[target['name']].output_shape

        layer_heads[target['name']] = L.DenseLayer(
            layer_heads[target['name']],
            target['n_out'],
            nonlinearity=nl.softmax)
        print target['name'], layer_heads[target['name']].output_shape

    return layer_heads


def deep_cnn_2d_mtl_at_2_fusion_for_main(params):
    """"""
    assert 'targets' in params
    nonlin = nl.elu
    main_target_name = 'tg'

    layers = L.InputLayer((None, 1, params['dur'], 128))
    print layers.output_shape

    sclr = joblib.load(params['scaler'])
    layers = L.standardize(
        layers, sclr.mean_.astype(np.float32),
        sclr.scale_.astype(np.float32), shared_axes=(0, 1, 2))
    print layers.output_shape

    n_filter = [32, 64, 64, 128, 256, 256]  # l
    filter_sz = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (1, 1)]  # m
    if params['dur'] > 50:
        conv_strd = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]  # c
        pool_sz = [(2, 2), (2, 2), (2, 2), (2, 2), None, None]  # n
    else:
        conv_strd = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]  # c
        pool_sz = [(2, 2), (2, 2), (2, 2), (2, 2), None, None]  # n
    pool_strd = [None, None, None, None, None, None]  # s
    batch_norm = [True, False, True, False, False, False]  # b
    dropout = [True, False, True, False, False, False]  # d # added
    conv_spec = zip(
        n_filter, filter_sz, conv_strd, pool_sz,
        pool_strd, batch_norm, dropout
    )

    # Shared first layer
    layers = L.Conv2DLayer(layers, 16, (5, 5), stride=(1, 1),
                           pad='same', nonlinearity=nonlin)
    layers = L.MaxPool2DLayer(layers, pool_size=(1, 2))
    layers = L.dropout(layers, p=0.1)

    layer_heads = OrderedDict()
    for target in params['targets']:
        first_trgt_spec_layer = True  # n_layer checker
        for l, m, c, n, s, b, d in conv_spec:
            if first_trgt_spec_layer:
                layer_heads[target['name']] = layers
                first_trgt_spec_layer = False
            if b:
                layer_heads[target['name']] = L.batch_norm(
                    L.Conv2DLayer(
                        layer_heads[target['name']], l, m, stride=c,
                        pad='same', nonlinearity=nonlin),
                )
            else:
                layer_heads[target['name']] = L.Conv2DLayer(
                    layer_heads[target['name']], l, m, stride=c,
                    pad='same', nonlinearity=nonlin
                )
            if n is not None:
                layer_heads[target['name']] = L.MaxPool2DLayer(
                    layer_heads[target['name']], pool_size=n, stride=s)

            if d:
                layer_heads[target['name']] = L.dropout(
                    layer_heads[target['name']], p=0.1)
            print layer_heads[target['name']].output_shape

        layer_heads[target['name']] = L.batch_norm(
            L.GlobalPoolLayer(layer_heads[target['name']]))
        layer_heads[target['name']] = L.dropout(
            layer_heads[target['name']])  # added
        print layer_heads[target['name']].output_shape

        layer_heads[target['name']] = L.batch_norm(
            L.DenseLayer(
                layer_heads[target['name']],
                256, nonlinearity=nonlin))
        layer_heads[target['name']] = L.dropout(
            layer_heads[target['name']])
        print layer_heads[target['name']].output_shape

        if target['name'] == main_target_name:
            continue
        else:
            layer_heads[target['name']] = L.DenseLayer(
                layer_heads[target['name']],
                target['n_out'],
                nonlinearity=nl.softmax)
            print target['name'], layer_heads[target['name']].output_shape

    main_target = filter(lambda t: t['name'] == main_target_name, params['targets'])
    side_targets = filter(lambda t: t['name'] != main_target_name, params['targets'])
    layer_heads[main_target[0]['name']] = L.DenseLayer(
        L.ConcatLayer(
            [L.get_all_layers(layer_heads[t['name']])[-2]  # fc for side trg
             for t in side_targets],
            axis=1
        ),
        main_target[0]['n_out'], nonlinearity=nl.softmax
    )
    print(main_target[0]['name'],
          layer_heads[main_target[0]['name']].output_shoutput_shape)

    return layer_heads


def shallow_cnn_2d_vanilla(params):
    """"""
    layers = L.InputLayer((None, 1, params['dur'], 128))
    print layers.output_shape

    sclr = joblib.load(params['scaler'])
    layers = L.standardize(
        layers, sclr.mean_.astype(np.float32),
        sclr.scale_.astype(np.float32), shared_axes=(0, 1, 2))
    print layers.output_shape

    n_filter = [8, 16, 16, 32]  # l
    filter_sz = [(5, 5), (5, 5), (1, 1), (5, 5)]  # m
    pool_sz = [(3, 3), (3, 3), None, (3, 3)]  # n
    pool_strd = [None, None, None, None]  # s
    batch_norm = [False, False, False, False]  # b
    conv_spec = zip(n_filter, filter_sz, pool_sz, pool_strd, batch_norm)

    for l, m, n, s, b in conv_spec:
        if b:
            layers = L.batch_norm(
                L.Conv2DLayer(layers, l, m, nonlinearity=nl.rectify)
            )
        else:
            layers = L.Conv2DLayer(layers, l, m, nonlinearity=nl.rectify)
        if n is not None:
            layers = L.MaxPool2DLayer(layers, pool_size=n, stride=s)
        print layers.output_shape

    layers = L.DenseLayer(layers, 64, nonlinearity=nl.rectify)
    print layers.output_shape

    layers = L.DenseLayer(layers, 16, nonlinearity=nl.softmax)
    print layers.output_shape

    return layers


def pons_cnn(params):
    """"""
    layers = L.InputLayer((None, 1, params['dur'], 128))
    print layers.output_shape

    sclr = joblib.load(params['scaler'])
    layers = L.standardize(
        layers, sclr.mean_.astype(np.float32),
        sclr.scale_.astype(np.float32), shared_axes=(0, 1, 2))
    print layers.output_shape

    layers_timbre = L.GlobalPoolLayer(
        L.batch_norm(L.Conv2DLayer(layers, 64, (1, 96))))

    layers_rhythm = L.GlobalPoolLayer(
        L.batch_norm(L.Conv2DLayer(layers, 64, (params['dur'] - 10, 1))))

    layers = L.ConcatLayer(
        [layers_rhythm, layers_timbre], axis=-1)

    layers = L.DenseLayer(layers, 64, nonlinearity=nl.rectify)
    print layers.output_shape

    layers = L.DenseLayer(layers, 16, nonlinearity=nl.softmax)
    print layers.output_shape

    return layers
