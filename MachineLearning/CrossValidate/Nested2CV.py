# Author: 赩林, xilin0x7f@163.com
import numpy as np
from sklearn.model_selection import check_cv
from .CV import CrossValidationEvaluator
# test import
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

class NestedCrossValidationEvaluator:
    def __init__(self, x, y, pipeline_out, pipeline_in, cv_out, cv_in, params):
        """
        Initialize the CrossValidationEvaluator with data, pipeline, and cv strategy.

        Parameters:
        - x: Input data (features).
        - y: Target labels.
        - pipeline: Scikit-learn pipeline object.
        - cv: Cross-validation splitting strategy (e.g., KFold, StratifiedKFold, or an int).
        """
        self.x = x
        self.y = y
        self.pipeline_out = pipeline_out
        self.pipeline_in = pipeline_in
        self.cv_out = cv_out
        self.cv_in = cv_in
        self.params = params

    def perform_cross_validation(self):
        cv_out_strategy = check_cv(self.cv_out)  # Ensure cv is a valid CV splitter
        self.best_params = []
        for i, (out_train_index, out_test_index) in enumerate(cv_out_strategy.split(self.x, self.y)):
            x_out_train, x_out_test = x[out_train_index], x[out_test_index]
            y_out_train, y_out_test = y[out_train_index], y[out_test_index]

            pass


if __name__ == "__main__":
    x = np.random.rand(100, 10)
    y = np.random.choice([0, 1], size=100)
    model = SVC(probability=True)
    pipeline = Pipeline([("model", model)])

    cv = KFold(5, shuffle=True, random_state=0)
    cv_evaluator = CrossValidationEvaluator(x, y, pipeline, cv)
    cv_evaluator.perform_cross_validation()
    print(cv_evaluator.pipeline_fitted[0])
    print(cv_evaluator.y_prob)
    print(cv_evaluator.y_pred)
    pass