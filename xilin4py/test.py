# Author: 赩林, xilin0x7f@163.com
from tqdm import tqdm
import time

outer_loop = tqdm(range(100), desc="Outer Loop", position=0)
for i in outer_loop:
    time.sleep(0.1)
    print('\n, aaaa')
    inner_loop = tqdm(range(100), desc="Inner Loop", position=1, leave=False)
    for j in inner_loop:
        time.sleep(0.1)
