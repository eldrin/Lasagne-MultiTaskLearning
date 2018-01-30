import uuid

from sklearn.externals import joblib

import os
import h5py
import tensorboard_logger as tblog

from model import build, network
from train import train
from utils import save_check_point, get_class_weight
from evaluate import evaluate, test

DATA_PATH = '/mnt/bulk2/WWW_FMA/mel_spec/'
TRAIN_DATA_FN = os.path.join(DATA_PATH, 'train.h5')
TEST_DATA_FN = os.path.join(DATA_PATH, 'test.h5')
SCALER_FN = './data/sclr_44k_logmel128.dat.gz'


def ex(learning_rate, split, epsilon, beta, dur,
       n_epochs, targets, batch_sz, shuffle, train_id=None):
    """"""
    if train_id is None:
        train_id = uuid.uuid4()

    logger = tblog.Logger('runs/{}'.format(train_id))

    params = {
        'data_fn': TRAIN_DATA_FN,
        'scaler': SCALER_FN,
        'split_fn': split,
        'learning_rate': learning_rate,
        'epsilon': epsilon,
        'beta': beta,
        'verbose': True,
        'n_epochs': n_epochs,
        'batch_sz': batch_sz,
        'dur': dur,  # 5 seconds
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
            test(mdl, hf, train_id, TEST_DATA_FN, params)

    return train_id, f1, ll


if __name__ == "__main__":
    """"""
    # == EXPERIMENT == #
    results = {}
    tid, f1, ll = ex(
        learning_rate=0.001,
        split='./data/fma_labels.split',
        epsilon=1e-8,
        beta=1e-5,
        dur=44,
        n_epochs=400,
        batch_sz=64,
        targets=[
            {'name': 'tg', 'n_out': 16, 'prob': 1./2},
            # {'name': 'po', 'n_out': 40, 'prob': 1./2}
            # {'name': 'adm', 'n_out': 40, 'prob': 1./3},
            {'name': 'am', 'n_out': 40, 'prob': 1./2}
        ],
        shuffle=False,
        train_id='mtl_tg_am_@2'
    )
    results[tid] = {'f1': f1, 'll': ll}
    print results

    # ### CONTINUE TRAINING ###
    # train_id = 'mtl_tg_3'
    # net, mdl, params = load_check_point(train_id, path='results/')
    # logger = tblog.Logger('runs/{}'.format(train_id))
    # with h5py.File(params['data_fn']) as hf:
    #     train(mdl, hf, params, logger)
    #     f1, ll = evaluate(mdl, hf, params)
    #     save_check_point(net, params, train_id, path='results/')
    # print {'f1': f1, 'll': ll}
