# Author: 赩林, xilin0x7f@163.com
import numpy as np
from skrebate import ReliefF

class ExtendedReliefF(ReliefF):
    def __init__(self, n_features_to_select=10, **kwargs):
        super(ExtendedReliefF, self).__init__(**kwargs)
        self.n_features_to_select = n_features_to_select
        self.support_mask_ = None

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
