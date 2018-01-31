import numpy as np
import pandas as pd
import h5py
from tqdm import trange
from sklearn.metrics import (f1_score,
                             classification_report,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from utils import load_check_point, shuffle_ids
import fire


def log_loss(y_true, y_pred):
    """"""
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def evaluate(model, data, params):
    """"""
    # only for string labels
    le = LabelEncoder().fit(data['y']['tg'][:])
    lb = LabelBinarizer().fit(data['y']['tg'][:])
    id_hash = {v: k for k, v in enumerate(data['ids'][:])}
    val_ids = sorted(shuffle_ids(params['split']['valid'], id_hash))

    Ypr = []
    for i in trange(0, len(val_ids), params['batch_sz'], ncols=80):
        X = data['X'][val_ids[i:i+params['batch_sz']]]
        M = data['mask'][val_ids[i:i+params['batch_sz']]]
        pred = []
        for j in range(0, X.shape[-2], params['dur'] / 2):
            x = X[:, :, j:j+params['dur']]
            if x.shape[-2] >= params['dur']:
                pred.append(model['tg']['predict'](x))
        Ypr.append(np.array(pred).mean(axis=0))
    Ypr = np.concatenate(Ypr, axis=0)
    Y = lb.transform(data['y']['tg'][val_ids])

    y, ypr = np.argmax(Y, axis=1), np.argmax(Ypr, axis=1)
    y_label = le.inverse_transform(y)
    ypr_label = le.inverse_transform(ypr)

    f1 = f1_score(y, ypr, average='macro')
    ll = log_loss(Y, Ypr)
    print
    print
    print 'LogLoss: {:.4f}'.format(ll)
    print 'F1: {:.4f}'.format(f1)
    print classification_report(y_label, ypr_label)
    print confusion_matrix(y, ypr)

    return f1, ll


def test(model, data, train_id, test_fn, params):
    """"""
    le = LabelEncoder().fit(data['y']['tg'][:])
    with h5py.File(test_fn) as test_data:
        Ypr = []
        for i in trange(0, test_data['X'].shape[0], params['batch_sz'], ncols=80):
            X = test_data['X'][i:i+params['batch_sz']]
            M = test_data['mask'][i:i+params['batch_sz']]
            pred = []
            for j in range(0, X.shape[-2], params['dur'] / 2):
                x = X[:, :, j:j+params['dur']]
                if x.shape[-2] >= params['dur']:
                    pred.append(model['tg']['predict'](x))
            Ypr.append(np.array(pred).mean(axis=0))
        Ypr = np.concatenate(Ypr, axis=0)

        out_df = pd.DataFrame(
            Ypr, columns=le.classes_, index=test_data['ids'][:])
        out_df.index.name = 'file_id'
        out_df.sort_index(inplace=True)
        out_df.to_csv('results/{}.csv'.format(train_id))


def feature(model, data, params,
            target='tg', agg=[np.mean, np.std, np.median]):
    """"""
    id_hash = {v: k for k, v in enumerate(data['ids'][:])}
    trn_ids = [(id_hash[int(i)], 'train') for i in params['split']['train']
               if int(i) in id_hash]
    val_ids = [(id_hash[int(i)], 'valid') for i in params['split']['valid']
               if int(i) in id_hash]
    ix_dset_map = dict(sorted(trn_ids + val_ids, key=lambda x: x[0]))

    # N = data['X'].shape[0]
    N = len(ix_dset_map)
    bs = params['batch_sz']
    Z = []  # feature
    Y = []  # label
    DSET = []  # set identifier
    IDS = []
    for i in trange(0, N, bs, ncols=80):
        slc = slice(i, i + bs)
        if i + bs > N:
            ids = range(i, N)
        else:
            ids = range(i, i + bs)
        dset = map(lambda j: ix_dset_map[j], ids)
        X = data['X'][slc]
        M = data['mask'][slc]
        y = data['y']['tg'][slc]
        feat = []
        for j in range(0, X.shape[-2], params['dur'] / 2):
            x = X[:, :, j:j+params['dur']]
            if x.shape[-2] >= params['dur']:
                feat.append(model[target]['transform'](x))
        feat = np.array(feat)
        if isinstance(agg, list):
            F = np.concatenate([
                stat(feat, axis=0) for stat in agg],
                axis=-1
            )
            Y.append(y)
            DSET.extend(dset)
            IDS.extend(ids)
        elif agg is None:
            # now it's (batch * chunk, feat)
            Y.append(np.repeat(y, feat.shape[0]))
            DSET.append(np.repeat(dset, feat.shape[0]))
            IDS.append(np.repeat(ids, feat.shape[0]))
            F = np.concatenate(feat.transpose(1, 0, 2))
        else:
            raise Exception
        Z.append(F)

    if agg is None:
        Y = np.concatenate(Y)
        DSET = np.concatenate(DSET)
        IDS = np.concatenate(IDS)

    Z = np.concatenate(Z, axis=0)
    return Z, Y, DSET, IDS


def prepare_submission(train_id, test_fn, path='results/'):
    """"""
    net, mdl, params = load_check_point(train_id, path=path)
    with h5py.File(params['data_fn']) as hf:
        test(mdl, hf, train_id, test_fn, params)


if __name__ == "__main__":
    fire.Fire(prepare_submission)
