import pandas as pd
from sklearn.metrics import roc_curve, auc

# 读取CSV文件
data = pd.read_csv('/home/chengyuan/PYT/all_momentum.csv')

# 提取真实值和预测值
true_values = data.iloc[:, 0]
predicted_values = data.iloc[:, 1]

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(true_values, predicted_values)

# 计算曲线下面积（AUC）
roc_auc = auc(fpr, tpr)

# 找到最佳阈值的索引
best_threshold_index = (tpr - fpr).argmax()

# 获取最佳阈值
best_threshold = thresholds[best_threshold_index]

print(f"最佳阈值: {best_threshold}")

