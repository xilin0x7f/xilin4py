# Author: 赩林, xilin0x7f@163.com
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from xilin4py.Radiomics import icc_compute_optimized

def f_score_old(x, y):
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

    return scores, 1/np.array(scores)


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
        self.threshold_p_value = threshold_p_value
        self.threshold_correlation = threshold_correlation

    def fit(self, x, y=None):
        p_values = compute_p_values(x, y)
        index_mapping = dict(zip(range(x.shape[1]), range(x.shape[1])))
        deleted_columns = np.argwhere(p_values >= self.threshold_p_value)
        index_mapping = updating_index(index_mapping, deleted_columns)
        x_selected = np.delete(x, deleted_columns, axis=1)
        while True:
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
        return self

    def transform(self, x, y=None):
        return x[:, self.selected_columns_]


class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cv=5, Cs=10, random_state=None):
        self.cv = cv
        self.Cs = Cs  # Cs describes the inverse of regularization strength
        self.random_state = random_state

    def fit(self, x, y):
        self.log_reg = LogisticRegressionCV(
            cv=self.cv, penalty='l1', solver='liblinear', Cs=self.Cs, random_state=self.random_state)
        self.log_reg.fit(x, y)
        self.selected_features_ = np.nonzero(self.log_reg.coef_[0])[0]
        return self

    def transform(self, x, y=None):
        return x[:, self.selected_features_]

class FeatureSelectorICC(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, x, y=None):
        # Split the data into two halves
        mid_col = x.shape[1] // 2
        data1 = x[:, :mid_col]
        data2 = x[:, mid_col:]

        # Convert to dataframes
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        icc = icc_compute_optimized(data1=df1, data2=df2)
        self.selected_columns_ = icc[icc["icc"] > self.threshold].index
        return self

    def transform(self, x, y=None):
        mid_col = x.shape[1] // 2
        data1 = x[:, :mid_col]
        return data1[:, self.selected_columns_]


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
