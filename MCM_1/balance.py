import pandas as pd
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score

# 读取CSV文件
data = pd.read_csv('all_momentum.csv')

# 提取真实值和预测值
true_values = data.iloc[:, 0]
predicted_values = data.iloc[:, 1]

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(true_values, predicted_values)

# 初始化最佳阈值和最佳平衡精度
best_threshold = None
best_balanced_accuracy = 0

# 尝试不同的阈值
for threshold in thresholds:
    # 根据阈值转换预测值
    predicted_labels = (predicted_values >= threshold).astype(int)
    
    # 计算平衡精度
    balanced_accuracy = balanced_accuracy_score(true_values, predicted_labels)
    
    # 更新最佳阈值和最佳平衡精度
    if balanced_accuracy > best_balanced_accuracy:
        best_balanced_accuracy = balanced_accuracy
        best_threshold = threshold

print(f"最佳阈值: {best_threshold}，最佳平衡精度: {best_balanced_accuracy}")

