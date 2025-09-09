import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 替换 '/path/to/your/file.xlsx' 为实际的文件路径
fp = '/home/chengyuan/PYT/merged.csv'

# 使用pandas读取Excel文件，不使用列名
d = pd.read_csv(fp, header=None)

# 分离因变量和自变量
y = d.iloc[:, 0]
X = d.iloc[:, 1:19]  # 从第二列到第十六列

# 划分数据集为训练集和测试集
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=80)

# 创建并训练逻辑斯蒂回归模型
mod = LogisticRegression()
mod.fit(Xtr, ytr)

# 在测试集上进行预测
ypr = mod.predict(Xte)

# 输出模型的准确性
acc = accuracy_score(yte, ypr)
print("Accuracy:", acc)

# 输出混淆矩阵
cm = confusion_matrix(yte, ypr)
print("Confusion Matrix:")
print(cm)

# 输出各个自变量的权重
ce = mod.coef_[0]
it = mod.intercept_[0]

print("\nIntercept:", it)
for i, coef in enumerate(ce):
    print(f"X{i+1} Coefficient: {coef}")

