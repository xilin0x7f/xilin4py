# Author: 赩林, xilin0x7f@163.com
import numpy as np
from sklearn.model_selection import check_cv
from sklearn import base
from sklearn import metrics
from sklearn.model_selection import ParameterGrid

class CrossValidationEvaluator:
    def __init__(self, x, y, pipeline, cv):
        self.x = x
        self.y = y
        self.pipeline = pipeline
        self.cv = cv
        self.pipeline_fitted = []

    def run(self):
        self.train_index_ = []
        self.test_index_ = []
        self.y_true = []
        self.y_pred = []
        if hasattr(self.pipeline, "predict_proba"):
            self.y_prob = []

        cv_strategy = check_cv(self.cv)  # Ensure cv is a valid CV splitter
        for i, (train_index, test_index) in enumerate(cv_strategy.split(self.x, self.y)):
            self.train_index_.append(train_index)
            self.test_index_.append(test_index)
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            pipeline = base.clone(self.pipeline)
            pipeline.fit(x_train, y_train)
            self.pipeline_fitted.append(pipeline)
            y_pred = pipeline.predict(x_test)
            self.y_pred.append(y_pred)
            self.y_true.append(y_test)
            if hasattr(pipeline, "predict_proba"):
                self.y_prob.append(pipeline.predict_proba(x_test))

        if hasattr(self.pipeline, "predict_proba"):
            self.y_prob = np.vstack(self.y_prob)
        self.y_pred = np.concatenate(self.y_pred)
        self.y_true = np.concatenate(self.y_true)


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

    def run(self, print_scores=False):
        self.print_scores = print_scores
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
                pipeline_out_transform.fit(x_out_train, y_out_train)
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
        self.y_pred = np.concatenate(self.y_pred)
        self.y_true = np.concatenate(self.y_true)

    def grid_search(self, x, y):
        search_params = self.search_params
        pipeline = base.clone(self.pipeline_in)
        scores = []
        for i, search_param in enumerate(search_params):
            pipeline_current = base.clone(pipeline)
            pipeline_current.set_params(**search_param)
            CV = CrossValidationEvaluator(x, y, pipeline_current, self.cv_in)
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
                elif self.scoring == "auc":
                    scores.append(metrics.roc_auc_score(y_true, y_prob[:, -1]))

        if self.print_scores:
            print(scores)
        best_param = search_params[int(np.argmax(scores))]
        return best_param