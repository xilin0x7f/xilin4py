# Author: 赩林, xilin0x7f@163.com
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import check_cv
from sklearn import base
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from imblearn.over_sampling import SMOTE

class CrossValidationEvaluator:
    def __init__(self, x, y, my_pipeline, cv, save_pipeline=False, verbose=False):
        self.x = x
        self.y = y
        self.my_pipeline = my_pipeline
        self.cv = cv
        self.pipeline_fitted = []
        self.train_index_ = None
        self.test_index_ = None
        self.y_prob = None
        self.y_pred, self.y_true = None, None
        self.verbose = verbose
        self.save_pipeline = save_pipeline

    def run(self):
        self.train_index_ = []
        self.test_index_ = []
        self.y_true = []
        self.y_pred = []
        if hasattr(self.my_pipeline, "predict_proba"):
            self.y_prob = []

        cv_strategy = check_cv(self.cv)  # Ensure cv is a valid CV splitter
        for i, (train_index, test_index) in enumerate(cv_strategy.split(self.x, self.y)):
            if self.verbose:
                print(f'\rCV {i}th.', end="", flush=True)
            self.train_index_.append(train_index)
            self.test_index_.append(test_index)
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            my_pipeline = base.clone(self.my_pipeline)
            my_pipeline.fit(x_train, y_train)
            if self.save_pipeline:
                self.pipeline_fitted.append(my_pipeline)
            y_pred = my_pipeline.predict(x_test)
            self.y_pred.append(y_pred)
            self.y_true.append(y_test)
            if hasattr(my_pipeline, "predict_proba"):
                self.y_prob.append(my_pipeline.predict_proba(x_test))

        if self.verbose:
            print()
        if hasattr(self.my_pipeline, "predict_proba"):
            self.y_prob = np.vstack(self.y_prob)
        self.y_pred = np.concatenate(self.y_pred)
        self.y_true = np.concatenate(self.y_true)


class NestedCrossValidationEvaluator:
    def __init__(self, x, y, pipeline_out, pipeline_in, cv_out, cv_in, transform_by_out, transform_range, search_params,
                 scoring="accuracy", verbose=False):
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
        self.print_scores = None
        self.best_params = None
        self.pipeline_out_fitted = None
        self.y_prob, self.y_true, self.y_pred = None, None, None
        self.verbose = verbose
        if isinstance(self.transform_range, slice):
            self.transform_range = list(range(*self.transform_range.indices(self.transform_range.stop)))

    def smote_resample(self, scaler=None, random_state=0):
        if scaler is not None:
            self.x = scaler.fit_transform(self.x)

        self.x, self.y = SMOTE(random_state=random_state).fit_resample(self.x, self.y)
        if scaler is not None:
            self.x = scaler.inverse_transform(self.x)

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
            if self.verbose:
                print(f"Nested CV {i}th. "
                      f"\nTrain shape {x_out_train.shape}, {y_out_train.shape}."
                      f"\nTest shape {x_out_test.shape}, {y_out_test.shape}")
            if self.transform_by_out:
                pipeline_out_transform = base.clone(Pipeline([self.pipeline_out.steps[i] for i in self.transform_range]))
                pipeline_out_transform.fit(x_out_train, y_out_train)
                x_out_train_transformed = pipeline_out_transform.transform(x_out_train)
                best_param = self.grid_search(x_out_train_transformed, y_out_train)
            else:
                best_param = self.grid_search(x_out_train, y_out_train)

            if self.verbose:
                print(f'Best param: {best_param}')

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

    def run_use_fitted(self, print_scores=False, out_range=None):
        self.pipeline_out = base.clone(Pipeline([step for i, step in enumerate(self.pipeline_out.steps) if i not in
                                                 out_range]))
        self.print_scores = print_scores
        cv_out_strategy = check_cv(self.cv_out)  # Ensure cv is a valid CV splitter
        self.best_params = []
        # self.pipeline_out_fitted = []
        self.y_pred = []
        self.y_true = []
        if hasattr(self.pipeline_out, "predict_proba"):
            self.y_prob = []
        for i, (out_train_index, out_test_index) in enumerate(cv_out_strategy.split(self.x, self.y)):
            if self.verbose:
                print(f"Nested CV {i}th.")
            x_out_train, x_out_test = self.x[out_train_index], self.x[out_test_index]
            y_out_train, y_out_test = self.y[out_train_index], self.y[out_test_index]
            fitted_pipeline_for_transform = Pipeline([self.pipeline_out_fitted[i].steps[j] for j in out_range])
            x_out_train = fitted_pipeline_for_transform.transform(x_out_train)
            x_out_test = fitted_pipeline_for_transform.transform(x_out_test)
            if self.transform_by_out:
                if isinstance(self.transform_range, slice):
                    self.transform_range = list(range(*self.transform_range.indices(self.transform_range.stop)))

                pipeline_out_transform = base.clone(Pipeline([self.pipeline_out.steps[i] for i in self.transform_range]))
                pipeline_out_transform.fit(x_out_train, y_out_train)
                x_out_train_transformed = pipeline_out_transform.transform(x_out_train)
                best_param = self.grid_search(x_out_train_transformed, y_out_train)
            else:
                best_param = self.grid_search(x_out_train, y_out_train)
            self.best_params.append(best_param)

            pipeline_out = base.clone(self.pipeline_out)
            pipeline_out.set_params(**best_param)
            pipeline_out.fit(x_out_train, y_out_train)
            # self.pipeline_out_fitted.append(Pipeline(fitted_pipeline_for_transform.steps + self.pipeline_out.steps))
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
        my_pipeline = base.clone(self.pipeline_in)
        scores = []
        for i, search_param in enumerate(search_params):
            if self.verbose:
                print(f"\rSearch parameter: {search_param}")
            pipeline_current = base.clone(my_pipeline)
            pipeline_current.set_params(**search_param)
            CV = CrossValidationEvaluator(x, y, pipeline_current, self.cv_in, verbose=self.verbose)
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
