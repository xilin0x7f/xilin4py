# Author: 赩林, xilin0x7f@163.com
import numpy as np
from scipy.stats import pearsonr

class BinaryClassifierSelector:
    def f_score(x, y):
        scores = []
        # F score
        for col in x.T:
            data = np.column_stack([col, y])
            data_pos = data[data[:, 1] == 1]
            data_neg = data[data[:, 1] == 0]
            score = (np.power(np.mean(data_pos[:, 0]) - np.mean(data[:, 0]), 2) +
                     np.power(np.mean(data_neg[:, 0]) - np.mean(data[:, 0]), 2)) / (
                            np.sum(np.power(data_pos[:, 0] - np.mean(data_pos[:, 0]), 2)) / (len(data_pos) - 1) +
                            np.sum(np.power(data_neg[:, 0] - np.mean(data_neg[:, 0]), 2)) / (len(data_neg) - 1)
                    )
            scores.append(score)

        return scores, 1/np.array(scores)

class RegressionSelector:
    def corr_score(X, y):
        corr = np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
        return np.abs(corr), [pearsonr(X[:, i], y)[1] for i in range(X.shape[1])]