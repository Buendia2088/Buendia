from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as mp
import seaborn as sb
import pandas as pd
# 替换 '/path/to/your/file.xlsx' 为实际的文件路径

fp = '/home/chengyuan/PYT/merged.csv'
# 使用pandas读取Excel文件，不使用列名
d = pd.read_csv(fp, header=None)

# 分离因变量和自变量
y = d.iloc[:, 0]
X = d.iloc[:, 1:19]  # 从第二列到第十九列，根据你的数据调整列索引

# 划分数据集为训练集和测试集
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=100)

# 创建并训练随机森林分类器
rc = RandomForestClassifier(n_estimators=1000, random_state=42)
rc.fit(Xtr, ytr)

# 在测试集上进行预测
ypr = rc.predict(Xte)

# 输出模型的准确性
acc = accuracy_score(yte, ypr)
print("Accuracy:", acc)

# 输出混淆矩阵
cm = confusion_matrix(yte, ypr)
print("Confusion Matrix:")
print(cm)

# 输出各个自变量的重要性
fi = rc.feature_importances_
for i, importance in enumerate(fi):
    print(f"X{i+1} Importance: {importance}")

# 如果需要可视化特征重要性，可以使用以下代码
mp.figure(figsize=(10, 6))
sb.barplot(x=fi, y=X.columns)
mp.title('Feature Importance')
mp.show()

