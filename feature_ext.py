import os
import numpy as np
from utils import load_check_point
from model import build_dist_feat_fnc
from evaluate import feature
import h5py
from sklearn.externals import joblib
import fire


def feature_ext(train_id, data_fn, out_path,
                agg=[np.mean, np.std, np.median, np.amin, np.amax],
                target='tg', model_path='results/', distributed=False):
    """"""
    print 'Extracting!...'
    net, mdl, params = load_check_point(train_id, path=model_path)
    if distributed:
        print 'Compile Distributed Feature Function!...'
        mdl = build_dist_feat_fnc(net, target)

    with h5py.File(data_fn, 'r') as hf:
        Z, y, dset, ids = feature(mdl, hf, params, target, agg)  # feature

    print 'Saving output...'
    out_fn = os.path.join(out_path, train_id + '_{}.dat.gz'.format(target))
    joblib.dump((Z, y, dset, ids), out_fn)


if __name__ == "__main__":
    fire.Fire(feature_ext)
