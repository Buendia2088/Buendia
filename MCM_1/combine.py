import os
import pandas as pd

# 指定包含表格的文件夹路径
folder_path = '/home/chengyuan/PYT/crosschecked_folder'

# 获取文件夹下所有表格文件的路径
table_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# 初始化一个空的DataFrame用于存储合并后的数据
merged_df = pd.DataFrame()

# 逐个读取表格并合并
for table_file in table_files:
    # 读取表格
    table_df = pd.read_csv(table_file, encoding='utf-8')  # 根据实际文件编码调整

    # 合并数据到总的DataFrame中
    merged_df = pd.concat([merged_df, table_df], ignore_index=True)

# 将合并后的数据写入新的CSV文件
merged_df.to_csv('/home/chengyuan/PYT/merged_table.csv', index=False, encoding='utf-8')  # 根据实际输出路径调整

