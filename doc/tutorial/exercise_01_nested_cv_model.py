# Author: 赩林, xilin0x7f@163.com
import copy
import multiprocessing
import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn import pipeline, preprocessing, metrics, feature_selection, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xilin4py.MachineLearning import NestedCrossValidationEvaluator
from xilin4py.MachineLearning.FeatureSelector import f_score
warnings.filterwarnings("ignore")

results_dir = r"results_dir"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

ml_params = [
    ("SVM", SVC(kernel="linear", random_state=0, probability=True, class_weight="balanced"),
     {"model__C": np.logspace(-5, 5, base=2)}),
    ("LR", LogisticRegression(random_state=0, class_weight="balanced", n_jobs=-1),
     {"model__C": np.logspace(-5, 5, base=2)}),
    ("RF", RandomForestClassifier(random_state=0, class_weight="balanced", n_jobs=-1),
     {"model__n_estimators": [10 * i for i in range(5, 21)]})
]
cv_out = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
cv_in = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

n_peru = 5000
n_peru_start = 0


def worker(task):
    data_path, model_name, idx, pipeline_out, pipeline_in, search_params = task
    data = pd.read_excel(data_path)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x, y = np.array(x), np.array(y)
    y = copy.deepcopy(y)
    random.shuffle(y)
    cv_evaluator = NestedCrossValidationEvaluator(x, y, pipeline_out, pipeline_in, cv_out, cv_in, True, slice(0, 2),
                                                  search_params)
    cv_evaluator.run()
    np.savetxt(os.path.join(results_dir, f"pred_{model_name}_{idx:05}.txt"),
               np.column_stack([cv_evaluator.y_true, cv_evaluator.y_pred, cv_evaluator.y_prob]))


if __name__ == "__main__":
    data_path = r"data_path"
    data = pd.read_excel(data_path)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x, y = np.array(x), np.array(y)
    for ml_param in ml_params:
        model_name, model, search_params = ml_param
        pipeline_out = pipeline.Pipeline([("scaler", preprocessing.StandardScaler()),
                                          ("selection", feature_selection.SelectKBest(f_score, k=60)),
                                          ("model", model)])
        pipeline_in = pipeline.Pipeline([("model", model)])
        cv_evaluator = NestedCrossValidationEvaluator(x, y, pipeline_out, pipeline_in, cv_out, cv_in, True, slice(0, 2),
                                                      search_params)
        cv_evaluator.run(print_scores=True)
        print(cv_evaluator.best_params)
        print(metrics.accuracy_score(cv_evaluator.y_true, cv_evaluator.y_pred))
        print(metrics.roc_auc_score(cv_evaluator.y_true, cv_evaluator.y_prob[:, 1]))
        np.savetxt(os.path.join(results_dir, f"pred_{model_name}.txt"),
                   np.column_stack([cv_evaluator.y_true, cv_evaluator.y_pred, cv_evaluator.y_prob]))

        # continue
        run_counter = 0
        process_per_run = 1000
        while n_peru_start + run_counter * process_per_run < n_peru:
            current_run_start = n_peru_start + run_counter * process_per_run
            current_run_end = min(n_peru_start + (run_counter + 1) * process_per_run, n_peru)
            tasks = (
                (data_path, model_name, idx, pipeline_out, pipeline_in, search_params)
                for idx in range(current_run_start, current_run_end)
            )
            # 并行化处理
            with multiprocessing.Pool(processes=20) as executor:
                list(executor.imap(worker, tasks))
            run_counter += 1
