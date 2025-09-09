import os
import pandas as pd
from collections import defaultdict

def count_elements_in_first_column(directory):
    # 创建一个默认字典来存储元素出现的次数
    element_count = defaultdict(int)
    
    # 遍历指定目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否是 Excel 文件（扩展名为 .xlsx）
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory, filename)
            print(f"正在处理文件: {file_path}")

            try:
                # 使用 pandas 读取 Excel 文件，只读取第一列，跳过表头
                excel_data = pd.read_excel(file_path, usecols=[0])  # 只读取第一列
                first_column_values = excel_data.iloc[:, 0]  # 获取第一列的所有值
                
                # 遍历第一列的每个元素，并统计出现次数
                for value in first_column_values:
                    element_count[value] += 1

            except Exception as e:
                print(f"读取文件时出错 {file_path}: {e}")
    
    return element_count

def save_counts_to_excel(counts, output_file):
    # 将字典转换为 pandas DataFrame
    df = pd.DataFrame(list(counts.items()), columns=['Element', 'Count'])
    
    # 将 DataFrame 保存到 Excel 文件
    df.to_excel(output_file, index=False)
    print(f"统计结果已保存到 {output_file}")

# 使用示例
directory_path = 'qndxx'  # 替换为你自己的 Excel 文件夹路径
output_file_path = 'results.xlsx'  # 替换为你想要保存结果的文件路径

# 统计元素出现次数
result_count = count_elements_in_first_column(directory_path)

# 保存统计结果到新的 Excel 文件
save_counts_to_excel(result_count, output_file_path)
