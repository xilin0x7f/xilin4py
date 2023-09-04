# Author: 赩林, xilin0x7f@163.com
import pandas as pd
import pingouin as pg
import numpy as np

def icc_compute(data1, data2):
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
        icc_value = icc_df.loc[2].ICC
        icc_all_features.append(icc_value)

    icc = np.array(icc_all_features)
    return pd.DataFrame(data={"feature": data1.columns, "icc": icc})
