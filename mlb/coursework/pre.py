import json
from pathlib import Path
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB  # 如需朴素贝叶斯，取消注释
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

# ========= 固定路径（同目录）=========
DIR   = os.path.abspath(os.curdir) + "\\Dataset"
TRAIN_IN  = os.path.join(DIR,'diabetic_data_training.csv')          # 原始训练数据
TRAIN_OUT = os.path.join(DIR,'diabetic_data_preprocessed.csv')      # 预处理后训练数据
TEST_IN = os.path.join(DIR,'diabetic_data_test.csv')                # 原始测试数据
TEST_OUT = os.path.join(DIR,'diabetic_data_test_preprocessed.csv')  # 预处理后测试数据
NPZ_OUT = os.path.join(DIR,'diabetes_preprocessed.npz')          # X / y, 仅作预处理后的训练数据
# ===================================


def preprocess(df):
    # 对数据进行预处理

    ### 该部分有待进一步调整 ###

    # 数据中的问号替换为NaN
    df.replace("?", np.nan, inplace=True)
    
    # 去除未知性别数据
    df = df[df["gender"] != "Unknown/Invalid"].copy()
    
    # 去除无关数据（病人ID等）和严重缺失数据
    df.drop(columns=["weight", "medical_specialty", "payer_code", "encounter_id", "patient_nbr"], inplace=True)
    
    # 填补少量缺失的数据
    df["race"].fillna("Unknown", inplace=True)
    # df["diag_1"].fillna("Unknown", inplace=True)
    # df["diag_2"].fillna("Unknown", inplace=True)
    df["diag_3"].fillna("Unknown", inplace=True)


    ###
    # df.drop(columns=["acetohexamide", "troglitazone", "examide", "citoglipton", 
    #                  "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"], inplace=True)

    # 对readmitted（目标标签）进行三分类
    if "readmitted" in df.columns:
        df["readmitted"] = df["readmitted"].map({"NO": 0, ">30": 1, "<30": 2}).astype(np.int64)

    return df


def main():
    category_cols = ["race", "gender", "age", "weight", "admission_type_id", "discharge_disposition_id", 
                    "admission_source_id", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3", 
                    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide", "chlorpropamide", 
                    "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", 
                    "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", 
                    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone", "change", 
                    "diabetesMed"]
    
    # 读取原始训练数据
    df = pd.read_csv(TRAIN_IN)

    # 基本预处理
    df = preprocess(df)


    # one hot code the category type
    all_cols = df.columns
    category_cols = [col for col in all_cols if col in category_cols]

    cat_cols = category_cols
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print("\n✅ 训练数据One-Hot编码完成")
    df.to_csv(TRAIN_OUT, index=False)

    # 拆分 X / y
    y = df["readmitted"].to_numpy(dtype=np.int64)
    X = df.drop(columns=["readmitted"]).to_numpy(dtype=np.float32)

    # 保存仅预处理后训练数据
    np.savez(NPZ_OUT, X=X, y=y)
    
    # 读取并预处理测试集
    df_test = pd.read_csv(TEST_IN)
    df_test = preprocess(df_test)

    df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)
    print("\n✅ 训练数据One-Hot编码完成")
    # 确保测试集与训练集列对齐
    df_test = df_test.reindex(columns=df.columns, fill_value=0)
    df_test.to_csv(TEST_OUT, index=False)
    # 完成提示
    print("\n✅ 训练集、测试集预处理完成")

if __name__ == "__main__":
    main()

DIR   = os.path.abspath(os.curdir) + "\\Dataset"
CSV_IN = os.path.join(DIR,'diabetic_data_preprocessed.csv') 
NPZ_OUT = os.path.join(DIR,'diabetes_with_pca.npz')

def main():
    df = pd.read_csv(CSV_IN) # 读取需要做降维的训练数据

    # 手动指定分类特征
    category_cols = ["race", "gender", "age", "weight", "admission_type_id", "discharge_disposition_id", 
                    "admission_source_id", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3", 
                    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide", "chlorpropamide", 
                    "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", 
                    "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", 
                    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone", "change", 
                    "diabetesMed", "readmitted"]

    # 计算数值特征集合和真正的分类特征集合
    # 分类特征集合需要再次计算，因为此时输入数据是预处理过的，可能已经失去了部分分类特征
    all_cols = df.columns
    num_cols = [col for col in all_cols if col not in category_cols]

    # 标准化数值特征
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(df[num_cols])

    # PCA 降噪 初步测试在保持87%~93%方差时效果最好
    pca = PCA(n_components=0.93, random_state=42)
    X_final = pca.inverse_transform(pca.fit_transform(X_num_scaled))
    
    y = df["readmitted"].to_numpy(dtype=np.int64)

    # 结果保存为 npz 文件：
    np.savez(NPZ_OUT, X=X_final, y = y)
    print("✅ PCA 降维完成")

if __name__ == "__main__":
    main()

# ---------- 固定文件名 ----------
TRAIN_NPZ = os.path.join(DIR,'diabetes_with_pca.npz') # 预处理并PCA处理的训练数据
TEST_CSV = os.path.join(DIR,'diabetic_data_test_preprocessed.csv') # 预处理过的测试数据
# ---------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time

def print_progress(current, total, start_time, prefix=""):
    """打印进度条和预计剩余时间"""
    elapsed = time.time() - start_time
    progress = current / total
    remaining = elapsed / progress * (1 - progress) if progress > 0 else 0
    
    bar_length = 30
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f"\r{prefix}[{bar}] {current}/{total} | 用时: {elapsed:.1f}s | 剩余: {remaining:.1f}s", end="")
    if current == total:
        print()

def main():
    # ---------------- 1. 读取数据 ----------------
    print("="*50)
    print("开始加载数据集...")
    data = np.load(TRAIN_NPZ)
    X_train, y_train = data["X"].astype(np.float32), data["y"].astype(np.int64)
    
    df_test = pd.read_csv(TEST_CSV)
    if "readmitted" not in df_test.columns:
        raise ValueError("测试集缺少 'readmitted' 标签列，无法评估准确率")
    
    y_test = df_test["readmitted"].to_numpy(dtype=np.int64)
    X_test = df_test.drop(columns=["readmitted"]).to_numpy(dtype=np.float32)
    
    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError(f"特征维度不一致：train {X_train.shape[1]}, test {X_test.shape[1]}")
    print("✅ 数据集加载完成")
    
    # ---------------- 2. 数据标准化 ----------------
    print("\n正在进行数据标准化...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    print("✅ 数据标准化完成")
    
    # ---------------- 3. 逻辑回归调优 ----------------
    print("\n" + "="*50)
    print("开始逻辑回归模型调优")
    
    # 定义逻辑回归参数网格
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # 正则化强度的倒数
        'penalty': ['l2'],  # 正则化类型
        'solver': ['lbfgs', 'sag', 'saga'],  # 优化算法
        'max_iter': [100, 200, 500],  # 最大迭代次数
        'class_weight': [None, 'balanced']  # 类别权重
    }
    
    # 计算总参数组合数
    total_params = 1
    for v in param_grid.values():
        total_params *= len(v)
    
    # 使用分层K折交叉验证（保持类别分布）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 创建逻辑回归模型
    lr_model = LogisticRegression(multi_class='multinomial')
    
    # 网格搜索
    print(f"\n参数搜索空间: {param_grid}")
    print(f"开始交叉验证 (共 {total_params} 种参数组合 × 5 折交叉验证)...")
    search_start = time.time()
    
    grid = GridSearchCV(
        estimator=lr_model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1  # 使用sklearn内置的verbose显示进度
    )
    
    # 执行网格搜索
    grid.fit(X_train_sc, y_train)
    
    # ---------------- 4. 结果评估 ----------------
    print("\n" + "="*50)
    print("逻辑回归调优完成")
    print(f"总训练时间: {time.time() - search_start:.1f}秒")
    
    # 获取最佳模型
    best_lr = grid.best_estimator_
    
    # 在训练集上的表现
    y_train_pred = best_lr.predict(X_train_sc)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # 在测试集上的表现
    y_test_pred = best_lr.predict(X_test_sc)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\n🔍 最佳参数: {grid.best_params_}")
    print(f"🏆 交叉验证最佳准确率: {grid.best_score_:.4f}")
    print(f"📊 训练集准确率: {train_acc:.4f}")
    print(f"🧪 测试集准确率: {test_acc:.4f}")
    
    # 详细分类报告
    print("\n分类报告 (测试集):")
    print(classification_report(y_test, y_test_pred))
    
    print("\n混淆矩阵 (测试集):")
    print(confusion_matrix(y_test, y_test_pred))
    
    # 保存最佳模型
    import joblib
    model_path = os.path.join(DIR, 'best_lr_model.pkl')
    joblib.dump(best_lr, model_path)
    print(f"\n💾 最佳模型已保存到: {model_path}")

if __name__ == "__main__":
    main()