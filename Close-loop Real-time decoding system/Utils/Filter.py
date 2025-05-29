from scipy import signal
import numpy as np


def Filter_init(l_freq_p=0.5, h_freq_p=4, fs=500, order=2, withDelay=True):
    LowPassFre = h_freq_p
    HighPassFre = l_freq_p

    h_freq_p = h_freq_p * 2 / fs
    l_freq_p = l_freq_p * 2 / fs

    if h_freq_p >= 1:
        h_freq_p = 1 - np.finfo(np.float32).eps

    if l_freq_p <= 0:
        l_freq_p = np.finfo(np.float32).eps

    wp = [l_freq_p, h_freq_p]

    b, a = signal.butter(order, wp, "bandpass")

    if withDelay:
        w, gd = signal.group_delay((b, a), w=500, fs=500)

        Filter_Delay = np.mean(gd[np.min(np.where(w > HighPassFre)): np.max(np.where(w < LowPassFre))])

        factor = 1 * 1000 / fs

        Filter_Delay = round(Filter_Delay / factor)

        if Filter_Delay < 0:
            Filter_Delay = 0
            delay = None
        else:
            delay = np.zeros(Filter_Delay, dtype=np.float64)

        return b, a, Filter_Delay, delay
    else:
        return b, a


def notch_filt(f_remove=50, fs=500):
    w0 = 2 * f_remove / fs
    Q = 300
    b, a = signal.iirnotch(w0, Q)
    return b, a


def Stage_preProcess(b, a, x, factor=None):
    y = signal.filtfilt(b, a, x)

    return y


def Decode_preProcess(b, a, x):
    for i in range(len(x)):
        x[i] = signal.filtfilt(b, a, x[i])
    return x


