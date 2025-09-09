import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error
import lightgbm as LGB
import time

# 读取数据
df = pd.read_csv('/home/chengyuan/PYT/merged.csv', encoding='gbk')
X = df.iloc[:, 1:18]
y = df.iloc[:, 0]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
    print('均方误差【MSE】:', mean_squared_error(true, pred))
    print('均方根误差【RMSE】:', np.sqrt(mean_squared_error(true, pred)))
    print('平均绝对误差【MAE】:', mean_absolute_error(true, pred))
    print('绝对误差中位数【MedianAE】:', median_absolute_error(true, pred))
    print('平均绝对百分比误差【MAPE】:', mean_absolute_percentage_error(true, pred))
    print('绝对百分比误差中位数【MedianAPE】:', median_absolute_percentage_error(true, pred))

# 建立LGB的dataset格式数据
lgb_train = LGB.Dataset(X_train, y_train)
lgb_eval = LGB.Dataset(X_test, y_test, reference=lgb_train)

# 定义超参数dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'max_depth': 10,
    'num_leaves': 40,
    'learning_rate': 0.1,
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

# 预测测试集
y_pred = m1.predict(X_test)

# 评估模型
regression_metrics(y_test, y_pred)

objective=['regression_l2','regression_l1','quantile','poisson','mape']
metrics=['l2','mae','quantile','poisson','mape']
metrics_test_data=pd.DataFrame(columns=['objective','metric','MAPE','Median APE','MAE'])
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              '开始目标函数与评估函数评估')
for i in objective:
    for k in metrics:
        size=metrics_test_data.size
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': i,
            'metric':k,
            'max_depth': 7,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        callback=LGB.early_stopping(stopping_rounds=10,verbose=0)
        gbm = LGB.train(params,lgb_train,num_boost_round=2000,
                valid_sets=lgb_eval,callbacks=[callback])
        y_pred = gbm.predict(X_test)
        metrics_test_data.loc[size]=[i,k,mean_absolute_percentage_error(y_test,y_pred),
                                     median_absolute_percentage_error(y_test,y_pred),
                                     mean_absolute_error(y_test,y_pred)
                                    ]
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),i,'+',k,' 完成评估',
              ' best iteration is:',gbm.best_iteration)
metrics_test_data.to_csv('/home/chengyuan/PYT/res.csv',encoding='gbk')
metrics_test_data