# Author: 赩林, xilin0x7f@163.com
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from xilin4py.Radiomics import icc_compute_optimized
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import copy

def f_score(x, y):
    # Calculate means once to avoid repeated calculations
    mean_data = np.mean(x, axis=0)
    data_pos = x[y == 1]
    data_neg = x[y == 0]

    mean_data_pos = np.mean(data_pos, axis=0)
    mean_data_neg = np.mean(data_neg, axis=0)

    # Vectorized numerator calculation
    numerator = (np.square(mean_data_pos - mean_data) + np.square(mean_data_neg - mean_data))

    # Vectorized denominator calculation
    denominator = (
            (np.sum(np.square(data_pos - mean_data_pos), axis=0) / (len(data_pos) - 1)) +
            (np.sum(np.square(data_neg - mean_data_neg), axis=0) / (len(data_neg) - 1))
    )

    scores = numerator / denominator

    return scores


def compute_p_values(x, y):
    p_values = np.zeros(x.shape[1])

    # Step 1: T-test or Chi-Squared Test
    for i in range(x.shape[1]):
        unique_values = np.unique(x[:, i]).size
        if unique_values > 5:
            # Assume y is binary and perform t-test
            t_stat, p_value = ttest_ind(x[y == 1, i], x[y == 0, i])
        else:
            # Chi-Squared Test
            contingency_table = np.histogram2d(x[:, i], y, bins=(unique_values, np.unique(y).size))[0] + 1
            chi2_stat, p_value, dof, ex = chi2_contingency(contingency_table)
        p_values[i] = p_value

    return p_values

def updating_index(old_index_dict, deleted_index):
    deleted_index = np.sort(deleted_index)
    if isinstance(deleted_index, np.ndarray):
        deleted_index = deleted_index.ravel()

    for index in deleted_index:
        old_index_dict.pop(index)

    new_dict = {i: old_index_dict[key] for i, key in enumerate(sorted(old_index_dict.keys()))}

    return new_dict

class RecursivePCorrFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_p_value=0.05, threshold_correlation=0.9):
        # 首先计算p值，排除p值低于threshold_p_value的特征，然后递归地删除相关性较高的特征，相关性很高时删除p值较高的特征
        self.threshold_p_value = threshold_p_value
        self.threshold_correlation = threshold_correlation
        self.selected_columns_ = None
        self.support_mask_ = None
        self.n_features_in_ = None

    def fit(self, x, y=None):
        self.n_features_in_ = x.shape[1]
        p_values = compute_p_values(x, y)
        index_mapping = dict(zip(range(x.shape[1]), range(x.shape[1])))
        deleted_columns = np.argwhere(p_values >= self.threshold_p_value)
        index_mapping = updating_index(index_mapping, deleted_columns)
        x_selected = np.delete(x, deleted_columns, axis=1)
        while True:
            # 如果特征数量已经低于correlation的阈值，则退出
            # 此步用于根据相关性筛选时以保留的特征数量为依据，而非相关性为依据
            if x_selected.shape[1] <= self.threshold_correlation:
                break

            correlation_matrix = np.corrcoef(x_selected, rowvar=False)

            if not isinstance(correlation_matrix, np.ndarray):
                break

            correlation_matrix[np.tril_indices_from(correlation_matrix)] = 0
            if np.max(np.abs(correlation_matrix)) < self.threshold_correlation:
                break

            row, col = np.unravel_index(np.argmax(np.abs(correlation_matrix)), correlation_matrix.shape)
            if p_values[index_mapping[row]] >= p_values[index_mapping[col]]:
                deleted_columns = [row]
            else:
                deleted_columns = [col]

            index_mapping = updating_index(index_mapping, deleted_columns)
            x_selected = np.delete(x_selected, deleted_columns, axis=1)

        self.selected_columns_ = list(index_mapping.values())
        self.support_mask_ = self._get_support_mask()
        return self

    def transform(self, x, y=None):
        return x[:, self.selected_columns_]

    def get_support(self):
        check_is_fitted(self)
        return copy.deepcopy(self.support_mask_)

    def _get_support_mask(self):
        support_mask = np.zeros(self.n_features_in_, dtype=bool)
        support_mask[self.selected_columns_] = True
        return support_mask


class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cv=5, Cs=10, random_state=None):
        self.cv = cv
        self.Cs = Cs  # Cs describes the inverse of regularization strength
        self.random_state = random_state
        self.log_reg = None
        self.selected_columns_ = None
        self.support_mask_ = None
        self.n_features_in_ = None

    def fit(self, x, y):
        self.n_features_in_ = x.shape[1]
        self.log_reg = LogisticRegressionCV(
            cv=self.cv, penalty='l1', solver='liblinear', Cs=self.Cs, random_state=self.random_state)
        self.log_reg.fit(x, y)
        self.selected_columns_ = np.nonzero(self.log_reg.coef_[0])[0]
        self.support_mask_ = self._get_support_mask()
        return self

    def transform(self, x, y=None):
        return x[:, self.selected_columns_]

    def get_support(self):
        check_is_fitted(self)
        return copy.deepcopy(self.support_mask_)

    def _get_support_mask(self):
        support_mask = np.zeros(self.n_features_in_, dtype=bool)
        support_mask[self.selected_columns_] = True
        return support_mask

class FeatureSelectorICC(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.n_features_in_ = 0
        self.support_mask_ = None
        self.selected_columns_ = None

    def fit(self, x, y=None):
        self.n_features_in_ = x.shape[1]
        # Split the data into two halves
        mid_col = x.shape[1] // 2
        data1 = x[:, :mid_col]
        data2 = x[:, mid_col:]

        # Convert to dataframes
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        icc = icc_compute_optimized(data1=df1, data2=df2)
        self.selected_columns_ = icc[icc["icc"] > self.threshold].index
        self.support_mask_ = self._get_support_mask()
        return self

    def transform(self, x, y=None):
        mid_col = x.shape[1] // 2
        data1 = x[:, :mid_col]
        return data1[:, self.selected_columns_]

    def get_support(self):
        check_is_fitted(self)
        return copy.deepcopy(self.support_mask_)

    def _get_support_mask(self):
        support_mask = np.zeros(self.n_features_in_, dtype=bool)
        support_mask[self.selected_columns_] = True
        return support_mask


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score

    # 我们将使用上面定义的 RecursivePCorrFeatureSelector 类
    # 创建数据
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建 pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selector', RecursivePCorrFeatureSelector(threshold_p_value=1.1)),
        ('classifier', LogisticRegression())
    ])

    # 训练 pipeline
    pipeline.fit(X_train, y_train)

    # 预测
    y_pred = pipeline.predict(X_test)

    # 计算准确度
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
