import numpy as np
from tqdm import trange
from prefetch_generator import background


def random_crop(X, M, Y, dur, overlap=True, offset=45):
    """"""
    XX = []
    YY = []
    for x, m, y in zip(X, M, Y):
        if m - (dur + offset) > 0:
            if overlap:
                choice_trg = m - (dur + offset)
            else:
                choice_trg = range(0, m, dur)
            st = np.random.choice(choice_trg)
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

    if params['verbose']:
        t = trange(0, len(ids), bs, desc='Batch', ncols=80)
    else:
        t = range(0, len(ids), bs)

    for i in t:
        # draw training samples
        idx = sorted(ids[slice(i, i + bs)])
        if target == 'tg':
            y_ = lb.transform(data['y'][target][idx])
        else:
            y_ = data['y'][target][idx]
        X_, y_ = random_crop(
            data['X'][idx], data['mask'][idx], y_, params['dur'])
        yield i, X_, y_, target
