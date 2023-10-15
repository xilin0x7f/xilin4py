# Author: 赩林, xilin0x7f@163.com
import numpy as np
from skrebate import ReliefF
from sklearn.base import BaseEstimator

class ExtendedReliefF(ReliefF):
    def __init__(self, **kwargs):
        super(ExtendedReliefF, self).__init__(**kwargs)
        self.support_mask_ = None
        self.kwargs = kwargs

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **params):
        self.kwargs.update(params)
        for key, value in params.items():
            setattr(self, key, value)

        super(ExtendedReliefF, self).set_params(**self.kwargs)
        return self

    def fit(self, x, y):
        super(ExtendedReliefF, self).fit(x, y)
        self.support_mask_ = self._get_support_mask()
        return self

    def _get_support_mask(self):
        if not hasattr(self, "top_features_"):
            raise AttributeError("No feature top_features_ attribute found. Ensure that the parent class computes it.")

        indices = self.top_features_[:self.n_features_to_select]

        mask = np.zeros(len(self.feature_importances_), dtype=bool)
        mask[indices] = True
        return mask

    def get_support(self, indices=False):
        # 检查是否拟合模型
        if self.support_mask_ is None:
            raise RuntimeError("You must fit the estimator before calling `get_support`")

        if indices:
            return np.where(self.support_mask_)[0]
        else:
            return self.support_mask_
