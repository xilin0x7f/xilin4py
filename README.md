# Example
```python
import sys
sys.path.append(r"C:\AppsData\PyCharm")
import numpy as np
from sklearn.model_selection import check_cv
from sklearn import base
from sklearn import metrics
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, ParameterGrid
from sklearn import preprocessing
from xilin4py.MachineLearning.CrossValidate.Nested2CV import NestedCrossValidationEvaluator

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
```
