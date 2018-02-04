import uuid
import json

from sklearn.externals import joblib

import os
import h5py
import fire
import tensorboard_logger as tblog

from model import build, network
from train import train
from utils import save_check_point, load_check_point, get_class_weight
from evaluate import evaluate, test


SCALER_FN = './data/sclr_44k_logmel128.dat.gz'


def ex(net, learning_rate, split, epsilon, beta, dur,
       n_epochs, targets, batch_sz, shuffle, data_path,
       overlap_chunk=True, kernel_multiplier=1, train_id=None):
    """"""
    if train_id is None:
        train_id = uuid.uuid4()

    logger = tblog.Logger('runs/{}'.format(train_id))

    params = {
        'network': net,
        'data_fn': os.path.join(data_path, 'train.h5'),
        'scaler': SCALER_FN,
        'split_fn': split,
        'learning_rate': learning_rate,
        'epsilon': epsilon,
        'beta': beta,
        'verbose': True,
        'n_epochs': n_epochs,
        'batch_sz': batch_sz,
        'dur': dur,  # frames
        'overlap_chunk': True if overlap_chunk else False,
        'kernel_multiplier': kernel_multiplier,
        'report_every': 100,
        'class_weight': False,
        'prepare_submission': False,
        'iter': 0
    }
    params['targets'] = targets

    with h5py.File(params['data_fn']) as hf:
        # load split info
        split = joblib.load(params['split_fn'])
        params.update(
            {'split': {k: map(int, v) for k, v in split.iteritems()}})

        # load class weight if needed
        if params['class_weight']:
            params.update(
                {'class_weight': get_class_weight(hf['y'][:])}
            )

        mdl, net = build(network(params), params)
        train(mdl, hf, params, shuffle, logger)
        save_check_point(net, params, train_id, path='results/')
        f1, ll = evaluate(mdl, hf, params)

        if params['prepare_submission']:
            # predict test dataset and prepare submission
            test(mdl, hf, train_id,
                 os.path.join(data_path, 'test.h5'), params)

    return train_id, f1, ll


def experiment(train_id, continue_ex=False):
    """"""
    if continue_ex:
        net, mdl, params = load_check_point(train_id, path='results/')
        logger = tblog.Logger('runs/{}'.format(train_id))
        with h5py.File(params['data_fn']) as hf:
            train(mdl, hf, params, logger)
            f1, ll = evaluate(mdl, hf, params)
            save_check_point(net, params, train_id, path='results/')
    else:
        # load config file
        params = json.load(open('config/{}.json'.format(train_id)))
        tid, f1, ll = ex(**params)

    print {'f1': f1, 'll': ll}


if __name__ == "__main__":
    """"""
    fire.Fire(experiment)
