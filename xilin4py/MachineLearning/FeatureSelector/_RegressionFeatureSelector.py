# Author: 赩林, xilin0x7f@163.com
import numpy as np
from scipy.stats import pearsonr

def corr_score(X, y):
    corr = np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
    return np.abs(corr), [pearsonr(X[:, i], y)[1] for i in range(X.shape[1])]