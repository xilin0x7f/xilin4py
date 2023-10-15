# Author: 赩林, xilin0x7f@163.com
from sklearn import base, pipeline
import sys
sys.path.insert(0, '/Users/johnnash/PycharmProjects/xilin4py')
from xilin4py.MachineLearning.FeatureSelector import ExtendedReliefF
from skrebate import ReliefF

my_pipeline = pipeline.Pipeline([
    ("feature selector", ExtendedReliefF(n_jobs=10))
])

print(my_pipeline[0].n_jobs)
my_pipeline_clone = base.clone(my_pipeline)
print(my_pipeline_clone[0].n_jobs)

a = ExtendedReliefF(n_jobs=10)
print(a.n_jobs)
a_clone = base.clone(a)
print(a_clone.n_jobs)

