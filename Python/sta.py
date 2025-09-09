import numpy as np

a = [2, 4, 6, 8]

print(np.mean(a))  # 均值
# print(np.average(a, weights=[1, 2, 1, 1]))  # 带权均值

#print(np.var(a))  # 总体方差
#print(np.var(a, ddof=1))  # 样本方差

print(np.std(a))  # 总体标准差
print(np.std(a, ddof=1))  # 样本标准差

RSD = np.std(a, ddof=1)/np.mean(a)  # 相对标准偏差
print(RSD)

