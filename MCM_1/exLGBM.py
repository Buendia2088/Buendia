import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as LGB
import matplotlib.pyplot as plt  # 添加这行导入
from imblearn.under_sampling import RandomUnderSampler

# 读取数据
df = pd.read_csv('wdata.csv', encoding='gbk')
X = df.iloc[:, 1:33]
y = df.iloc[:, 0]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义欠采样策略
under = RandomUnderSampler(sampling_strategy={0: int(0.5*len(y_train[y_train==0]))})

# 执行欠采样
X_train_res, y_train_res = under.fit_resample(X_train, y_train)

# 定义回归模型评估误差指标
def median_absolute_percentage_error(y_true, y_pred):
    non_zero_mask = y_true != 0
    non_zero_y_true = y_true[non_zero_mask]
    non_zero_y_pred = y_pred[non_zero_mask]

    if len(non_zero_y_true) == 0:
        return np.inf

    return np.median(np.abs((non_zero_y_pred - non_zero_y_true) / non_zero_y_true))

def regression_metrics(true, pred):
    print('回归模型评估指标结果:')
    print('绝对百分比误差中位数【MedianAPE】:', median_absolute_percentage_error(true, pred))

# 建立LGB的dataset格式数据
lgb_train = LGB.Dataset(X_train_res, y_train_res)
lgb_eval = LGB.Dataset(X_test, y_test, reference=lgb_train)

# 定义超参数dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'quantile',
    'metric': 'quantile',
    'max_depth': 10,
    'num_leaves': 40,
    'learning_rate': 0.2,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 7,
    'verbose': -1
}

# 定义callback回调
callback = [LGB.early_stopping(stopping_rounds=10, verbose=True),
            LGB.log_evaluation(period=10, show_stdv=True)]

# 训练模型
m1 = LGB.train(params, lgb_train, num_boost_round=2000,
               valid_sets=[lgb_train, lgb_eval], callbacks=callback)

# 打印各自变量的权重
print("Feature importance:")
for feature, importance in zip(X.columns, m1.feature_importance()):
    print(f"{feature}: {importance}")

# 画出特征重要性图
LGB.plot_importance(m1, max_num_features=10, importance_type='gain')
plt.show()

# 预测测试集
y_pred = m1.predict(X_test)

# 评估模型
regression_metrics(y_test, y_pred)

def Predict_One(filepath):
    # 预测新数据集
    X_new = pd.read_csv(filepath, encoding='gbk', header=0)
    y_new_true = X_new.iloc[:, 0]
    y_pred_new = m1.predict(X_new.iloc[:, 1:33])

    # 打印新数据集的回归评估指标
    regression_metrics(y_new_true, y_pred_new)

    # 将预测结果输出到新的表格
    result_df = pd.DataFrame({'True_Values': y_new_true, 'Predicted_Values': y_pred_new})
    result_df.to_csv('wres.csv', index=False, encoding='gbk')  # 请替换为输出路径

Predict_One('wdata.csv')
