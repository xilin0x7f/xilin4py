# Author: 赩林, xilin0x7f@163.com
from sklearn.model_selection import BaseCrossValidator
import numpy as np


class CustomSplit(BaseCrossValidator):
    def __init__(self, n_splits=1, test_size=0.3, random_state=None, shuffle=True):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle

    def split(self, x, y=None, groups=None):
        n_samples = len(x)
        indices = np.arange(n_samples)
        train_samples = int((1 - self.test_size) * n_samples)

        # 设置随机种子
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_splits):
            if self.shuffle:
                rng.shuffle(indices)
            yield indices[:train_samples], indices[train_samples:]

    def get_n_splits(self, x=None, y=None, groups=None):
        return self.n_splits


if __name__ == "__main__":
    # 使用自定义的交叉验证
    splitter = CustomSplit(n_splits=1, test_size=0.3, random_state=0, shuffle=True)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])

    for train_index, test_index in splitter.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print("X_train:", X_train)
        print("X_test:", X_test)

