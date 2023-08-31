# Author: 赩林, xilin0x7f@163.com
import numpy as np
from sklearn import base
from sklearn.model_selection import cross_val_predict, check_cv
# test import
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

class CrossValidationEvaluator:
    def __init__(self, x, y, pipeline, cv):
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
        self.pipeline = pipeline
        self.cv = cv
        self.pipeline_fitted = []


    def perform_cross_validation(self):
        """
        Perform cross-validation on the stored data using the provided pipeline and cv.

        Results are stored in the 'results_' attribute.
        """
        self.train_index_ = []
        self.test_index_ = []
        self.y_pred = []
        if hasattr(self.pipeline, "predict_proba"):
            self.y_prob = []

        cv_strategy = check_cv(self.cv)  # Ensure cv is a valid CV splitter
        for i, (train_index, test_index) in enumerate(cv_strategy.split(self.x, self.y)):
            self.train_index_.append(train_index)
            self.test_index_.append(test_index)
            X_train, X_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            pipeline = base.clone(self.pipeline)
            pipeline.fit(X_train, y_train)
            self.pipeline_fitted.append(pipeline)
            y_pred = pipeline.predict(X_test)
            self.y_pred.append(y_pred)
            if hasattr(pipeline, "predict_proba"):
                self.y_prob.append(pipeline.predict_proba(X_test))

        if hasattr(self.pipeline, "predict_proba"):
            self.y_prob = np.vstack(self.y_prob)
        self.y_pred = np.hstack(self.y_pred)

if __name__ == "__main__":
    x = np.random.rand(100, 10)
    y = np.random.choice([0, 1], size=100)
    model = SVC(probability=True)
    pipeline = Pipeline([("model", model)])
    print(hasattr(pipeline, "predict_proba"))
    cv = KFold(5, shuffle=True, random_state=0)
    cv_evaluator = CrossValidationEvaluator(x, y, pipeline, cv)
    cv_evaluator.perform_cross_validation()
    print(id(cv_evaluator.pipeline_fitted[0]))
    print(id(cv_evaluator.pipeline_fitted[1]))
    print(cv_evaluator.pipeline_fitted[0].predict_proba(x[0:2, :]))
    print(cv_evaluator.pipeline_fitted[1].predict_proba(x[0:2, :]))
    print(cv_evaluator.y_prob)
    print(cv_evaluator.y_pred)
    pass

