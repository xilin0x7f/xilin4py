# Author: 赩林, xilin0x7f@163.com
import pandas as pd
import pingouin as pg
import numpy as np
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

def icc_compute(data1, data2, icc_index=1):
    # 单个rater, 在不同时间进行评分，然后计算ICC时选用下面的方式 icc_index=1
    csv_data1 = data1.iloc[:, :].values.astype('float64')
    csv_data2 = data2.iloc[:, :].values.astype('float64')
    icc_all_features = []
    for i in range(csv_data1.shape[1]):
        data_1 = pd.DataFrame(data=csv_data1[:, i], columns=['f'], dtype=float)
        data_2 = pd.DataFrame(data=csv_data2[:, i], columns=['f'], dtype=float)
        data_1.insert(0, "reader", np.ones(data_1.shape[0]))
        data_2.insert(0, "reader", np.ones(data_2.shape[0]) * 2)
        data_1.insert(0, "target", np.array(range(data_1.shape[0])))
        data_2.insert(0, "target", np.array(range(data_2.shape[0])))
        data = pd.concat([data_1, data_2])

        icc_df = pg.intraclass_corr(data=data, targets="target", raters="reader", ratings="f")
        icc_value = icc_df.iloc[icc_index].ICC
        icc_all_features.append(icc_value)

    icc = np.array(icc_all_features)
    return pd.DataFrame(data={"feature": data1.columns, "icc": icc})

def compute_icc_for_feature(i, data, icc_index):
    icc_df = pg.intraclass_corr(data=data, targets='target', raters='reader', ratings=f"{i}")
    return icc_df.iloc[icc_index].ICC

def icc_compute_optimized(data1, data2, icc_index=1, n_jobs=-1):
    origin_columns = data1.columns
    data1.columns = [f"{idx}" for idx in range(data1.shape[1])]
    data2.columns = [f"{idx}" for idx in range(data2.shape[1])]
    data12 = pd.concat([data1, data2], axis=0, ignore_index=True)
    icc_all_features = np.empty(data1.shape[1])
    readers = np.concatenate([np.ones(data1.shape[0]), np.ones(data2.shape[0]) * 2])
    targets = np.arange(data1.shape[0]).tolist() * 2  # repeat the range twice
    data = pd.DataFrame({
        'target': targets,
        'reader': readers,
    })
    data = pd.concat([data, data12], axis=1)
    icc_all_features = Parallel(n_jobs=n_jobs)(
        delayed(compute_icc_for_feature)(i, data, icc_index) for i in range(data1.shape[1])
    )

    return pd.DataFrame(data={"feature": origin_columns, "icc": icc_all_features})


if __name__ == "__main__":
    import time
    # Example data
    np.random.seed(123)
    data1 = pd.DataFrame(np.random.rand(10000, 10))
    data2 = pd.DataFrame(np.random.rand(10000, 10))

    # Call the optimized function
    start_time = time.time()
    result = icc_compute(data1, data2)
    print(result)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")

    start_time = time.time()
    result = icc_compute_optimized(data1, data2)
    print(result)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")
