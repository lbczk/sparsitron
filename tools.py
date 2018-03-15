import time
import numpy as np


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', f.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('{}  {:10.4f} s'.format(f.__name__, te - ts))
        return result
    return timed


@timeit
def get_cor(itt, nb=2000):
    b = np.zeros(len(itt.next()))
    for i in range(nb):
        a = itt.next()
        b += a[-1] * a
    return b / nb