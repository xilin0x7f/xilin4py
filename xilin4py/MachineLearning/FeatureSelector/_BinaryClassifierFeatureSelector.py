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

class RecursivePCorrFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_p_value=0.05, threshold_correlation=0.9):
        self.threshold_p_value = threshold_p_value
        self.threshold_correlation = threshold_correlation

    def fit(self, x, y=None):
        df = pd.DataFrame(x)
        p_values = []

        # Step 1: T-test or Chi-Squared Test
        for column in df.columns:
            unique_values = df[column].nunique()
            if unique_values > 5:
                # Assume y is binary and perform t-test
                t_stat, p_value = ttest_ind(df[column][y == 1], df[column][y == 0])
            else:
                # Chi-Squared Test
                contingency_table = pd.crosstab(df[column], y)
                chi2_stat, p_value, dof, ex = chi2_contingency(contingency_table)
            p_values.append(p_value)

        # Select features with p-values below the threshold
        selected_columns = df.columns[np.array(p_values) < self.threshold_p_value]

        # Step 2: Recursive Correlation Elimination
        df_selected = df[selected_columns]
        correlation_matrix = df_selected.corr().abs()
        upper_triangle_indices = np.triu_indices_from(correlation_matrix, k=1)
        correlation_list = correlation_matrix.values[upper_triangle_indices]
        column_pair_list = list(zip(*upper_triangle_indices))

        if len(correlation_list) == 0:
            self.selected_columns_ = selected_columns.astype(int)
            return self

        while max(correlation_list) > self.threshold_correlation:
            correlated_pairs = [pair for pair, corr_value in zip(column_pair_list, correlation_list) if corr_value > self.threshold_correlation]
            p_values_of_correlated = [(p_values[pair[0]], pair[0]) for pair in correlated_pairs] + [(p_values[pair[1]], pair[1]) for pair in correlated_pairs]
            column_to_remove_index = max(p_values_of_correlated, key=lambda x: x[0])[1]
            column_to_remove = selected_columns[column_to_remove_index]

            # Convert selected_columns to a list, remove the column, and convert it back to an Index
            selected_columns_list = selected_columns.tolist()
            selected_columns_list.remove(column_to_remove)
            selected_columns = pd.Index(selected_columns_list)

            df_selected = df[selected_columns]
            correlation_matrix = df_selected.corr().abs()
            upper_triangle_indices = np.triu_indices_from(correlation_matrix, k=1)
            correlation_list = correlation_matrix.values[upper_triangle_indices]
            column_pair_list = list(zip(*upper_triangle_indices))

        # Store column indices instead of column names
        self.selected_columns_ = selected_columns.astype(int)

        return self

    def transform(self, X, y=None):
        return X[:, self.selected_columns_]


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