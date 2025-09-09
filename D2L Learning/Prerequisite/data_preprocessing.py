import os
import torch
import pandas as pd
import numpy as np  # 显式导入numpy

os.makedirs('data', exist_ok=True)
data_file = os.path.join('data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

# 使用select_dtypes选择DataFrame中的数值列
inputs = data.iloc[:, 0:2]
outputs = data.iloc[:, 2]

# 计算平均值并填充缺失值
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True, dtype=float)
print(inputs)

X, Y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(Y)