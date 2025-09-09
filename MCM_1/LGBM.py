import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import lightgbm as lg

# 替换 '/path/to/your/file.xlsx' 为实际的文件路径
fp = '/home/chengyuan/PYT/standardized_1301.csv_p2_processed.xlsx'
# 使用pandas读取Excel文件，不使用列名
d = pd.read_excel(fp, header=None)

# 分离因变量和自变量
y = d.iloc[:, 0]
X = d.iloc[:, 1:17]  # 从第二列到第十九列，根据你的数据调整列索引

# 划分数据集为训练集和测试集
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.21, random_state=233)

# 数据标准化
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr)
Xte = sc.transform(Xte)

# 创建LightGBM分类器
mo = lg.LGBMClassifier()

# 训练模型
mo.fit(Xtr, ytr)

# 输出每个自变量的特征重要性
ft = mo.feature_importances_
for i, importance in enumerate(ft):
    print(f"Importance for X{i + 1}: {importance}")

# 在测试集上进行预测
ypr = mo.predict(Xte)

# 输出模型的准确性
acc = accuracy_score(yte, ypr)
print("Accuracy:", acc)

# 输出混淆矩阵
cm = confusion_matrix(yte, ypr)
print("Confusion Matrix:")
print(cm)
