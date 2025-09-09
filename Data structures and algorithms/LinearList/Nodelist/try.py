import numpy as np

# 生成一些虚拟数据
# 工作年数
years_of_experience = np.array([1, 3, 6, 7, 8])
# 领导的员工人数
number_of_subordinates = np.array([2, 5, 6, 7, 9])
# 员工薪资（假设的）
salary = np.array([32001, 35009, 43703, 48438, 51576])

# 使用最小二乘法计算回归参数
# 构建设计矩阵 X，包含一个全为1的列和自变量
X = np.column_stack((np.ones_like(years_of_experience), years_of_experience, number_of_subordinates))
# 计算参数
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(salary)

# 打印参数
print("回归参数:")
print("截距:", beta[0])
print("工作年数系数:", beta[1])
print("领导员工人数系数:", beta[2])

# 预测薪资
new_years_of_experience = 6
new_number_of_subordinates = 7
predicted_salary = beta[0] + beta[1] * new_years_of_experience + beta[2] * new_number_of_subordinates
print("预测薪资:", predicted_salary)
