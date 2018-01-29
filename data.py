import numpy as np
from tqdm import trange
from prefetch_generator import background


def random_crop(X, M, Y, dur, offset=45):
    """"""
    XX = []
    YY = []
    for x, m, y in zip(X, M, Y):
        if m - (dur + offset) > 0:
            st = np.random.choice(m - (dur + offset))
            XX.append(x[:, st:st + dur])
            YY.append(y)
        else:
            continue
    return np.array(XX).astype(np.float32), np.array(YY)


@background()
def prepare_batch(data, ids, params, lb):
    """"""
    # draw task
    target = np.random.choice(
        a=map(lambda x: x['name'], params['targets']),
        size=1, replace=False,
        p=map(lambda x: x['prob'], params['targets']))[0]
    target = str(target)

    bs = params['batch_sz']
    n_epochs = params['n_epochs']

    if params['verbose']:
        t = trange(0, len(ids), params['batch_sz'], desc='Batch', ncols=80)
    else:
        t = range(0, len(ids), params['batch_sz'])

    for i in t:
        # draw training samples
        idx = sorted(ids[slice(i, i+params['batch_sz'])])
        if target == 'tg':
            y_ = lb.transform(data['y'][target][idx])
        else:
            y_ = data['y'][target][idx]
        X_, y_ = random_crop(
            data['X'][idx], data['mask'][idx], y_, params['dur'])
        yield i, X_, y_, target