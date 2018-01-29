import numpy as np
from tqdm import trange
from data import prepare_batch, random_crop
from sklearn.preprocessing import LabelBinarizer


def shuffle_ids(id_list, id_hash):
    return np.random.permutation(
        [id_hash[x] for x in id_list if x in id_hash])


def train(model, data, params, tblogger=None):
    """"""
    # actual training
    id_hash = {v:k for k, v in enumerate(data['ids'][:])}

    # only for string labels
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer().fit(data['y']['tg'][:])

    # params['iter'] = 0
    try:
        if params['verbose']:
            epoch = trange(params['n_epochs'], desc='[Loss : -.--] Epoch', ncols=80)
        else:
            epoch = range(params['n_epochs'])

        for n in epoch:
            trn_ids = shuffle_ids(params['split']['train'], id_hash)
            val_ids = shuffle_ids(params['split']['valid'], id_hash)

            for i, X_, y_, target in prepare_batch(data, trn_ids, params, lb):

                if params['iter'] % params['report_every'] == 0:
                    # draw validation samples
                    idx_v = sorted(np.random.choice(
                            val_ids, params['batch_sz'], replace=False))
                    if target == 'tg':
                        y_v = lb.transform(data['y'][target][idx_v])
                    else:
                        y_v = data['y'][target][idx_v]
                    X_v, y_v = random_crop(
                        data['X'][idx_v], data['mask'][idx_v], y_v, params['dur'])

                    c = model[target]['cost'](X_, y_).item()
                    cv = model[target]['cost'](X_v, y_v).item()
                    a = model[target]['acc'](X_, y_).item()
                    av = model[target]['acc'](X_v, y_v).item()

                    if tblogger is not None:
                        tblogger.log_value('%s_cost_tr' % target, c, params['iter'])
                        tblogger.log_value('%s_cost_vl' % target, cv, params['iter'])
                        tblogger.log_value('%s_acc_tr' % target, a, params['iter'])
                        tblogger.log_value('%s_acc_vl' % target, av, params['iter'])

                    if params['verbose']:
                        epoch.set_description(
                            '[v_loss : {:.4f} / v_acc: {:.4f}]Epoch'.format(cv, av))

                model[target]['train'](X_, y_)
                params['iter'] += 1

    except KeyboardInterrupt as kbe:
        print('User Stopped!')