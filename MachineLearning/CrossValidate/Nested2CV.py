# Author: 赩林, xilin0x7f@163.com
import numpy as np
from sklearn.model_selection import check_cv
from sklearn import base
from sklearn import metrics
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, ParameterGrid
from sklearn import preprocessing
from .CV import CrossValidationEvaluator

class NestedCrossValidationEvaluator:
    def __init__(self, x, y, pipeline_out, pipeline_in, cv_out, cv_in, transform_by_out, transform_range, search_params,
                 scoring="accuracy", **params):
        self.x = x
        self.y = y
        self.pipeline_out = pipeline_out
        self.pipeline_in = pipeline_in
        self.cv_out = cv_out
        self.cv_in = cv_in
        self.transform_by_out = transform_by_out
        self.transform_range = transform_range
        self.search_params = ParameterGrid(search_params)
        self.scoring = scoring

    def run(self):
        cv_out_strategy = check_cv(self.cv_out)  # Ensure cv is a valid CV splitter
        self.best_params = []
        self.pipeline_out_fitted = []
        self.y_pred = []
        self.y_true = []
        if hasattr(self.pipeline_out, "predict_proba"):
            self.y_prob = []
        for i, (out_train_index, out_test_index) in enumerate(cv_out_strategy.split(self.x, self.y)):
            x_out_train, x_out_test = self.x[out_train_index], self.x[out_test_index]
            y_out_train, y_out_test = self.y[out_train_index], self.y[out_test_index]
            if self.transform_by_out:
                pipeline_out_transform = base.clone(self.pipeline_out[self.transform_range])
                pipeline_out_transform.fit(x_out_train, y_out_test)
                x_out_train_transformed = pipeline_out_transform.transform(x_out_train)
                best_param = self.grid_search(x_out_train_transformed, y_out_train)
            else:
                best_param = self.grid_search(x_out_train, y_out_train)
            self.best_params.append(best_param)

            pipeline_out = base.clone(self.pipeline_out)
            pipeline_out.set_params(**best_param)
            pipeline_out.fit(x_out_train, y_out_train)
            self.pipeline_out_fitted.append(pipeline_out)
            y_pred = pipeline_out.predict(x_out_test)
            self.y_pred.append(y_pred)
            self.y_true.append(y_out_test)
            if hasattr(pipeline_out, "predict_proba"):
                self.y_prob.append(pipeline_out.predict_proba(x_out_test))

        if hasattr(self.pipeline_out, "predict_proba"):
            self.y_prob = np.vstack(self.y_prob)
        self.y_pred = np.hstack(self.y_pred)
        self.y_true = np.hstack(self.y_true)

    def grid_search(self, x, y):
        search_params = self.search_params
        pipeline = base.clone(self.pipeline_in)
        scores = []
        for i, search_param in enumerate(search_params):
            pipeline_current = base.clone(pipeline)
            pipeline_current.set_params(**search_param)
            CV = CrossValidationEvaluator(x, y, pipeline, self.cv_in)
            CV.run()
            y_pred = CV.y_pred
            y_true = CV.y_true
            y_prob = None
            if hasattr(CV, "y_prob"):
                y_prob = CV.y_prob
            if callable(self.scoring):
                scores.append(self.scoring(y_true, y_pred, y_prob=None))
            elif isinstance(self.scoring, str):
                if self.scoring == "accuracy":
                    scores.append(metrics.accuracy_score(y_true, y_pred))

        best_param = search_params[int(np.argmax(scores))]
        return best_param


if __name__ == "__main__":
    x = np.random.rand(100, 10)
    y = np.random.choice([0, 1], size=100)
    model = SVC(probability=True)
    pipeline_out = Pipeline([("scaler", preprocessing.StandardScaler()),
                             ("model", model)])
    pipeline_in = Pipeline([("model", model)])

    cv = KFold(5, shuffle=True, random_state=0)
    cv_evaluator = NestedCrossValidationEvaluator(x, y, pipeline_out, pipeline_in, cv, cv, True, slice(0, 1), {"model__C": [0.1, 0.2, 0.3]})
    cv_evaluator.run()
    print(cv_evaluator.pipeline_out_fitted[0])
    print(id(cv_evaluator.pipeline_out_fitted[0]))
    print(id(cv_evaluator.pipeline_out_fitted[1]))
    print(cv_evaluator.best_params)
    print(cv_evaluator.y_prob)
    print(cv_evaluator.y_pred)