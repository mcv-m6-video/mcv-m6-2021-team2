import numpy as np

def calc_optical_flow(gt: np.array, pred: np.array):
    discard = gt[:,:, -1] != 0

    ch0 = gt[:,:,0] - pred[:,:,0]
    ch1 = gt[:,:,1] - pred[:,:,1]

    msen = np.sqrt(ch0**2 + ch1**2)
    msen = msen[discard]

    pepn = np.sum(msen > 3) / msen.shape[0]

    return np.mean(msen), pepn * 100