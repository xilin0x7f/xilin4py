# Author: 赩林, xilin0x7f@163.com
import numpy as np
from math import sqrt
from sklearn.metrics import roc_auc_score

def roc_auc_ci(y_true, y_score, positive=1):
    auc = roc_auc_score(y_true, y_score)
    n1 = np.sum(y_true == positive)
    n2 = np.sum(y_true != positive)
    q1 = auc / (2 - auc)
    q2 = 2*auc**2 / (1 + auc)
    se_auc = sqrt((auc*(1 - auc) + (n1 - 1)*(q1 - auc**2) + (n2 - 1)*(q2 - auc**2)) / (n1*n2))
    lower = auc - 1.96*se_auc
    upper = auc + 1.96*se_auc
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return lower, upper
