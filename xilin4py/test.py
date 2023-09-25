# Author: 赩林, xilin0x7f@163.com
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind, chi2_contingency

x, y = make_classification(n_samples=1000, n_features=20, random_state=42)
p_values = np.zeros(x.shape[1])

# Step 1: T-test or Chi-Squared Test
for i in range(x.shape[1]):
    unique_values = np.unique(x[:, i]).size
    if unique_values > 5:
        # Assume y is binary and perform t-test
        t_stat, p_value = ttest_ind(x[y == 1, i], x[y == 0, i])
    else:
        # Chi-Squared Test
        contingency_table = np.histogram2d(x[:, i], y, bins=(unique_values, 2))[0]
        chi2_stat, p_value, dof, ex = chi2_contingency(contingency_table)
    p_values[i] = p_value

# Select features with p-values below the threshold
selected_columns = np.where(p_values < 1.1)[0]
x_selected = x[:, selected_columns]

print('selected_columns_index', selected_columns)
# Step 2: Recursive Correlation Elimination
while True:
    correlation_matrix = np.corrcoef(x_selected, rowvar=False)
    correlation_matrix[np.tril_indices_from(correlation_matrix)] = 0
    correlated_pairs = np.column_stack(np.where(np.abs(correlation_matrix) > 0.3))
    print('-'*100, 'corr')
    # print(correlation_matrix)
    print(correlated_pairs)
    if correlated_pairs.size == 0:
        break  # Exit the loop if no correlations are above the threshold

    p_values_of_correlated = [(p_values[pair[0]], pair[0]) for pair in correlated_pairs] + [(p_values[pair[1]], pair[1]) for pair in correlated_pairs]
    column_to_remove_index = max(p_values_of_correlated, key=lambda x: x[0])[1]
    print('column_to remove index', column_to_remove_index)
    x_selected = np.delete(x_selected, column_to_remove_index, axis=1)
    selected_columns = np.delete(selected_columns, column_to_remove_index)
    print(selected_columns)