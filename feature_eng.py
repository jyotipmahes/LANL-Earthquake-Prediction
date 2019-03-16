import numpy as np
import pandas as pd
import os
from scipy.signal import hilbert, hann, convolve
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")


data = np.array(pd.read_csv('train1.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}))


def classic_sta_lta(a, nsta, nlta):
    """
    Computes the standard STA/LTA from a given input array a. The length of
    the STA is given by nsta in samples, respectively is the length of the
    LTA given by nlta in samples. Written in Python.
    .. note::
        There exists a faster version of this trigger wrapped in C
        called :func:`~obspy.signal.trigger.classic_sta_lta` in this module!
    :type a: NumPy :class:`~numpy.ndarray`
    :param a: Seismic Trace
    :type nsta: int
    :param nsta: Length of short time average window in samples
    :type nlta: int
    :param nlta: Length of long time average window in samples
    :rtype: NumPy :class:`~numpy.ndarray`
    :return: Characteristic function of classic STA/LTA
    """
    # The cumulative sum can be exploited to calculate a moving average (the
    # cumsum function is quite efficient)
    sta = np.cumsum(a ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad zeros
    sta[:nlta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta


def calculate(i):
#     for i in range(0, x.shape[0]-150000, stepsize):
    feature = dict()
    x_val = data[i:i+150000,0]
    feature['seg'] = 'a'+str(i)
    feature['y_target'] = data[i+150000,1]
    feature['mean'] = float(x_val.mean())
    feature['std'] = float(x_val.std())
    feature['max1'] = float(x_val.max())
    feature['min1'] = float(x_val.min())
    feature['sum1'] = float(x_val.sum())
    feature['abs_sum'] = float(np.abs(x_val).sum())
    feature['mean_change_abs'] = float(np.mean(np.abs(np.diff(x_val))))
    feature['mean_change_rate'] = float(np.mean(np.nonzero((np.diff(x_val) / x_val[:-1]))[0]))
    feature['abs_max'] = float(np.abs(x_val).max())
    feature['abs_min'] = float(np.abs(x_val).min())
    feature['std_first_50000'] = float(x_val[:50000].std())
    feature['std_last_50000'] = float(x_val[-50000:].std())
    feature['std_first_10000'] = float(x_val[:10000].std())
    feature['std_last_10000'] = float(x_val[-10000:].std())
    feature['avg_first_50000'] = float(x_val[:50000].mean())
    feature['avg_last_50000'] = float(x_val[-50000:].mean())
    feature['avg_first_10000'] = float(x_val[:10000].mean())
    feature['avg_last_10000'] = float(x_val[-10000:].mean())
    feature['min_first_50000'] = float(x_val[:50000].min())
    feature['min_last_50000'] = float(x_val[-50000:].min())
    feature['min_first_10000'] = float(x_val[:10000].min())
    feature['min_last_10000'] = float(x_val[-10000:].min())
    feature['max_first_50000'] = float(x_val[:50000].max())
    feature['max_last_50000'] = float(x_val[-50000:].max())
    feature['max_first_10000'] = float(x_val[:10000].max())
    feature['max_last_10000'] = float(x_val[-10000:].max())
    feature['max_to_min'] = float(x_val.max() / np.abs(x_val.min()))
    feature['max_to_min_diff'] = float(x_val.max() - np.abs(x_val.min()))
    feature['count_big'] = float(len(x_val[np.abs(x_val) > 500]))
    feature['mean_change_rate_first_50000'] = float(np.mean(np.nonzero((np.diff(x_val[:50000]) / x_val[:50000][:-1]))[0]))
    feature['mean_change_rate_last_50000'] = float(np.mean(np.nonzero((np.diff(x_val[-50000:]) / x_val[-50000:][:-1]))[0]))
    feature['mean_change_rate_first_10000'] = float(np.mean(np.nonzero((np.diff(x_val[:10000]) / x_val[:10000][:-1]))[0]))
    feature['mean_change_rate_last_10000'] = float(np.mean(np.nonzero((np.diff(x_val[-10000:]) / x_val[-10000:][:-1]))[0]))

    feature['q70'] = float(np.quantile(x_val, 0.70)) 
    feature['q75'] = float(np.quantile(x_val, 0.75)) 
    feature['q60'] = float(np.quantile(x_val, 0.60))
    feature['q65'] = float(np.quantile(x_val, 0.65)) 
    feature['q85'] = float(np.quantile(x_val, 0.85))
    feature['q90'] = float(np.quantile(x_val, 0.90))
    feature['q80'] = float(np.quantile(x_val, 0.80))
    feature['q95'] = float(np.quantile(x_val, 0.95))
    feature['q99'] = float(np.quantile(x_val, 0.99))

    ########### added
    feature['iqr'] = np.subtract(*np.percentile(x_val, [75, 25]))
    feature['ave10'] = stats.trim_mean(x_val, 0.1)
    feature['q999'] = np.quantile(x_val,0.999)
    feature['q001'] = np.quantile(x_val,0.001)

    for w in [10, 100, 1000]:
        x_roll_std = pd.DataFrame(x_val).rolling(w).std().dropna().values
        feature['ave_roll_std_' + str(w)] = x_roll_std.mean()
        feature['std_roll_std_' + str(w)] = x_roll_std.std()
        feature['max_roll_std_' + str(w)] = x_roll_std.max()
        feature['min_roll_std_' + str(w)] = x_roll_std.min()
        feature['q01_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.01)
        feature['q05_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.05)
        feature['q95_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.95)
        feature['q99_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.99)

    feature['mad'] = pd.DataFrame(x_val).mad().values[0]
    feature['kurt'] = pd.DataFrame(x_val).kurtosis().values[0]
    feature['skew'] = pd.DataFrame(x_val).skew().values[0]
    feature['med'] = pd.DataFrame(x_val).median().values[0]

    feature['Hilbert_mean'] = np.abs(hilbert(x_val)).mean()
    feature['Hann_window_mean'] = (convolve(x_val, hann(150), mode='same') / sum(hann(150))).mean()
    feature['classic_sta_lta5_mean'] = classic_sta_lta(x_val, 50, 1000).mean()
    feature['Moving_average_700_mean'] = pd.DataFrame(x_val).rolling(window=700).mean().mean(skipna=True).values[0]
    return feature

size = data.shape[0]
stepsize = 1000
jobs = [i for i in range(0, (size-150000), stepsize)]


import subprocess
from fastprogress import progress_bar, master_bar
from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel(func, job_list, n_jobs=8):
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(func, job) for job in job_list]
        for f in progress_bar(as_completed(futures), total=len(job_list)):
            pass
    return [f.result() for f in futures]

result = parallel(calculate, jobs)

pd.DataFrame(result).to_csv('Features1.csv',index=False)
