# Author: 赩林, xilin0x7f@163.com
s = slice(2, 9)
lst = list(range(*s.indices(100)))
print(lst)