# Author: 赩林, xilin0x7f@163.com
import copy
import os
import random
import warnings
import itertools
import imblearn
import sys

# sys.path.insert(0, "C:/AppsData/PyCharm/xilin4py/xilin4py")
sys.path.insert(0, "/Users/johnnash/PyCharmProjects/xilin4py")
import numpy as np
import pandas as pd
# from sklearnex import patch_sklearn

# patch_sklearn()
from sklearn import pipeline, preprocessing, metrics, feature_selection, model_selection, base
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from xilin4py.MachineLearning import NestedCrossValidationEvaluator
from xilin4py.MachineLearning.FeatureSelector import f_score
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

# permutation function
def worker(task):
    results_dir, idx, cv_evaluator, model_name = task
    cv_evaluator = copy.deepcopy(cv_evaluator)
    random.shuffle(cv_evaluator.y)
    cv_evaluator.run()
    print(model_name, idx)
    print(f"ACC: {metrics.accuracy_score(y, cv_evaluator.y_pred)}")
    np.savetxt(os.path.join(results_dir, f"pred_{model_name}_{idx:05}.txt"),
               np.column_stack([cv_evaluator.y, cv_evaluator.y_pred, cv_evaluator.y_prob]))


results_dir = r"./doc/tutorial/data"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

transform_by_out, transform_range = True, [0, 1, 2]
pipeline_out_pre_steps = [
    ("scaler", preprocessing.StandardScaler()),
    ("smote", SMOTE(random_state=1)),
    ("fscore_selector", feature_selection.SelectKBest(f_score, k=10))
]

ml_params = [
    ("SVM", SVC(kernel="linear", random_state=0, probability=True, class_weight="balanced"),
     {"model__C": np.logspace(-5, 5, base=2)}),
    ("LR", LogisticRegression(random_state=0, class_weight="balanced", n_jobs=-1),
     {"model__C": np.logspace(-5, 5, base=2)}),
    ("RF", RandomForestClassifier(random_state=0, class_weight="balanced", n_jobs=-1),
     {"model__n_estimators": [10 * i for i in range(5, 21)]})
]

cv_out = model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
cv_in = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=0)

n_peru = 10
n_peru_start = 0

data_paths = ["./doc/tutorial/data/binary_data.csv"]
combinations = itertools.product(ml_params, data_paths)
for combination in combinations:
    ml_param, data_path = combination
    data = pd.read_csv(data_path)
    samples_0 = data[data['target'] == 0].sample(100, random_state=0)
    samples_1 = data[data['target'] == 1].sample(200, random_state=0)

    # 合并样本
    data = pd.concat([samples_0, samples_1])

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    columns = x.columns
    x, y = np.array(x), np.array(y)

    model_name, model, search_params = ml_param
    if len(data_paths) > 1:
        model_name = model_name + str(data_paths.index(data_path))

    print(model_name)
    pipeline_out_post_steps = [
        ("model", model)
    ]
    # pipeline_out = base.clone(pipeline.Pipeline(pipeline_out_pre_steps + pipeline_out_post_steps))
    # if transform_by_out:
    #     new_steps = [step for i, step in enumerate(pipeline_out.steps) if i not in transform_range]
    #     pipeline_in = base.clone(pipeline.Pipeline(new_steps))
    # else:
    #     pipeline_in = base.clone(pipeline_out)

    pipeline_out = imblearn.pipeline.Pipeline(pipeline_out_pre_steps + pipeline_out_post_steps)

    if transform_by_out:
        new_steps = [step for i, step in enumerate(pipeline_out.steps) if i not in transform_range]
        pipeline_in = imblearn.pipeline.clone(imblearn.pipeline.Pipeline(new_steps))
    else:
        pipeline_in = imblearn.pipeline.clone(pipeline_out)

    cv_evaluator = NestedCrossValidationEvaluator(x, y, pipeline_out, pipeline_in, cv_out, cv_in, transform_by_out,
                                                  transform_range, search_params, verbose=False)
    # cv_evaluator = CrossValidationEvaluator(x, y, pipeline_out, cv_out, save_pipeline=True, verbose=True)
    cv_evaluator.run(print_scores=True)
    print(f"Best Params: {cv_evaluator.best_params}")
    print("ACC: ", metrics.accuracy_score(cv_evaluator.y_true, cv_evaluator.y_pred))
    print("AUC: ", metrics.roc_auc_score(cv_evaluator.y_true, cv_evaluator.y_prob[:, 1]))
    print()
    np.savetxt(os.path.join(results_dir, f"pred_{model_name}.txt"),
               np.column_stack([cv_evaluator.y_true, cv_evaluator.y_pred, cv_evaluator.y_prob]))

    # get_strong_feature(cv_evaluator, columns, x, y, save_dir=results_dir, model_name=model_name)
    run_counter = 0
    process_per_run = 5000
    while n_peru_start + run_counter * process_per_run < n_peru:
        current_run_start = n_peru_start + run_counter * process_per_run
        current_run_end = min(n_peru_start + (run_counter + 1) * process_per_run, n_peru)
        tasks_info = (
            (results_dir, idx, cv_evaluator, model_name)
            for idx in range(current_run_start, current_run_end)
        )

        tasks = (
            delayed(worker)(task_info) for task_info in tasks_info
        )

        list(
            Parallel(n_jobs=5)(tasks)
        )

        run_counter += 1
