import os
import pandas as pd

def count_string_in_excel_files(directory, search_string):
    count = 0
    
    # 遍历指定目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否是 Excel 文件（扩展名为 .xlsx）
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory, filename)
            print(f"正在处理文件: {file_path}")

            try:
                # 使用 pandas 读取 Excel 文件
                excel_data = pd.read_excel(file_path, sheet_name=None)  # 读取所有工作表
                
                # 遍历每个工作表
                for sheet_name, sheet_data in excel_data.items():
                    # 将每个工作表的所有内容转为字符串，然后进行搜索
                    count += sheet_data.astype(str).apply(lambda x: x.str.contains(search_string, case=False, na=False)).sum().sum()

            except Exception as e:
                print(f"读取文件时出错 {file_path}: {e}")
    
    return count

# 使用示例
directory_path = 'qndxx'  # 替换为你自己的 Excel 文件夹路径
search_string = '韩霈霆'  # 替换为你要搜索的字符串

result_count = count_string_in_excel_files(directory_path, search_string)
print(f"字符串 '{search_string}' 在所有 Excel 文件中共出现了 {result_count} 次。")
