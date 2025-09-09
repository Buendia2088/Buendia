import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 替换 '/path/to/your/file.xlsx' 为实际的文件路径
xlsx_file_path = '/home/chengyuan/PYT/test.xlsx'

# 使用pandas读取Excel文件，不使用列名
df = pd.read_excel(xlsx_file_path, header=None)

# 分离因变量和自变量
y = df.iloc[:, 0]
X = df.iloc[:, 1:20]  # 从第二列到第十九列，根据你的数据调整列索引

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建神经网络模型
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 输出每个自变量的权重
for layer in model.layers:
    if isinstance(layer, Dense):
        weights, _ = layer.get_weights()
        for i, weight in enumerate(weights[0]):
            print(f"Weight for X{i + 1}: {weight}")

# 在测试集上进行预测
y_pred_proba = model.predict(X_test)
y_pred = np.round(y_pred_proba)

# 输出模型的准确性
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 输出混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

