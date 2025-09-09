import os
import shutil
import time

# 获取当前脚本文件名，避免移动自身
script_name = os.path.basename(__file__)

for entry in os.scandir('/photo/byR7'):
    if entry.is_file() and entry.name != script_name:
        # 获取文件最后修改时间
        mtime = entry.stat().st_mtime
        time_struct = time.localtime(mtime)
        
        # 提取年份和月份
        year = time_struct.tm_year
        month = time_struct.tm_mon
        
        # 创建目标目录路径
        year_dir = f"{year:04}"
        month_dir = f"{year:04}年{month:02}月"
        target_path = os.path.join(year_dir, month_dir)
        
        # 创建目录（自动忽略已存在的目录）
        os.makedirs(target_path, exist_ok=True)
        
        # 移动文件到新位置
        try:
            shutil.move(entry.path, os.path.join(target_path, entry.name))
            print(f"已移动 {entry.name} => {target_path}/")
        except Exception as e:
            print(f"移动 {entry.name} 失败: {str(e)}")