import uuid
import logging
import copy
import json
from collections import namedtuple
import tensorboard_logger as tblog
import numpy as np

import theano
from theano import tensor as T

from lasagne import layers as L
from lasagne import objectives as obj
from lasagne import regularization as reg
from lasagne import updates

from networks import fcn_transfer

from sklearn.externals import joblib
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score

from prefetch_generator import background
from tqdm import trange, tqdm
import fire


logging.basicConfig(filename='results.log', level=logging.INFO)
SCALER_FN = './data/sclr_44k_logmel128.dat.gz'


@background()
def prepare_batch(X, y, bs, lb):
    """"""
    for i in trange(0, X.shape[0], bs, desc='Batch', ncols=80):
        slc = slice(i, i + bs)
        y_ = lb.transform(y[slc])
        X_ = X[slc]
        yield i, X_, y_


def transfer(sources, learning_rate, epsilon, beta,
             n_epochs, batch_sz, train_id=None, test_sources=False):
    """"""
    if train_id is None:
        train_id = uuid.uuid4()

    # launch logger
    logger = tblog.Logger('runs/{}'.format(train_id))

    # launch model
    net = fcn_transfer({'inputs': sources})
    input = T.matrix('input')
    target = T.matrix('target')
    o = L.get_output(net, inputs=input)
    o_vl = L.get_output(net, inputs=input, deterministic=True)

    loss = obj.categorical_crossentropy(o, target).mean()
    loss_vl = obj.categorical_crossentropy(o_vl, target).mean()
    wd_l2 = reg.regularize_network_params(net, reg.l2)
    wd_l2 *= beta

    acc_vl = obj.categorical_accuracy(o_vl, target).mean()

    updates_ = updates.adam(
        loss + wd_l2, L.get_all_params(net, trainable=True),
        learning_rate=learning_rate, epsilon=epsilon)

    Model = namedtuple('Model', 'partial_fit predict cost acc')
    model = Model(
        partial_fit=theano.function(
            [input, target], updates=updates_, allow_input_downcast=True),
        predict=theano.function(
            [input], o_vl, allow_input_downcast=True),
        cost=theano.function(
            [input, target], loss_vl, allow_input_downcast=True),
        acc=theano.function(
            [input, target], acc_vl, allow_input_downcast=True)
    )

    # load data
    D = [joblib.load(fn) for fn in sources]

    # prepare data
    trn_ix = np.where(D[0][2] == 'train')[0]
    val_ix = np.where(D[0][2] == 'valid')[0]
    ids_map = copy.deepcopy(D[0][3])

    y = copy.deepcopy(D[0][1])
    X = np.concatenate([d[0] for d in D], axis=1)
    lb = LabelBinarizer().fit(y)

    del D

    # TRAIN!
    iters = 0
    try:
        epoch = trange(n_epochs, desc='[Loss : -.--] Epoch', ncols=80)
        for n in epoch:
            np.random.shuffle(trn_ix)
            np.random.shuffle(val_ix)

            for i, X_, y_ in prepare_batch(X, y, batch_sz, lb):
                if iters % 100 == 0:
                    ix = np.random.choice(val_ix, batch_sz, replace=False)
                    X_v, y_v = X[ix], lb.transform(y[ix])

                    c = model.cost(X_, y_).item()
                    cv = model.cost(X_v, y_v).item()
                    a = model.acc(X_, y_).item()
                    av = model.acc(X_v, y_v).item()

                    logger.log_value('trns_cost_tr', c, iters)
                    logger.log_value('trns_cost_vl', cv, iters)
                    logger.log_value('trns_acc_tr', a, iters)
                    logger.log_value('trns_acc_vl', av, iters)

                    epoch.set_description(
                        '[v_loss: {:.4f} / v_acc: {:.4f}]Epoch'.format(cv, av))
                model.partial_fit(X_, y_)
                iters += 1

    except KeyboardInterrupt as kbe:
        print('User Stopped!')

    # evaluate
    uniq_ix_set = list(set(ids_map[val_ix]))
    Y_pred = []
    y_true = []
    Xvl = X[val_ix]
    yvl = y[val_ix]
    for i in tqdm(uniq_ix_set):
        ix = np.where(ids_map[val_ix] == i)[0]
        Y_pred.append(model.predict(Xvl[ix]).mean(axis=0))
        y_true.append(yvl[ix][0])
    Y_true = lb.transform(y_true)
    y_pred = [lb.classes_[i] for i in np.argmax(Y_pred, axis=1)]

    f1 = f1_score(y_true, y_pred, average='macro')
    ll = -np.mean(
        np.sum(Y_true * np.log(np.maximum(Y_pred, 1e-8)), axis=1)
    )

    if test_sources is not None:
        del X, y  # delete training data from memory
        # load test data

    # return result
    return train_id, f1, ll


def main(train_id):
    """"""
    params = json.load(open('config/{}.json'.format(train_id)))
    tid, f1, ll = transfer(**params)
    logging.info('{} - f1 - {:.5f} - ll - {:.5f}'.format(tid, f1, ll))
    print {'f1': f1, 'll': ll}


if __name__ == "__main__":
    """"""
    fire.Fire(main)



