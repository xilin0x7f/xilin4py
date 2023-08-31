# Author: 赩林, xilin0x7f@163.com
import numpy as np
a = []
b = np.array([[1, 3], [2, 4]])
a.append(b)
a.append(b)
a = np.vstack(a)
print(a)

for i, v in enumerate([1, 2, 3]):
    print(i, v)