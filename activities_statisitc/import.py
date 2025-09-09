"""
根据活动文本，把成员在各活动中的分工写到 Excel。
使用方法：
    1. 放好 activities.txt 和 members.xlsx
    2. python update_members.py
"""

import re
from pathlib import Path

import pandas as pd

TEXT_PATH   = Path("activities.txt")   # 活动文本
EXCEL_PATH  = Path("members.xlsx")     # 成员名单
NAME_COL    = 2                        # 姓名列索引（第 3 列 -> index 2）
ENCODING    = "utf-8"                  # 文本编码

def parse_activities(raw: str):
    blocks = re.split(r"\n\s*\n", raw.strip())

    activities = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        header = lines[0].rstrip("：")
        name_to_roles = {}

        for line in lines[1:]:
            if "：" not in line:
                continue
            role, names = line.split("：", 1)
            role = role.strip()
            # 姓名以“顿号、逗号”分隔
            for name in re.split(r"[、,，]", names):
                name = name.strip()
                if not name:
                    continue
                name_to_roles.setdefault(name, []).append(role)

        activities.append((header, name_to_roles))

    return activities


def update_excel(excel_path: Path, activities, name_col_idx: int = 2):
    df = pd.read_excel(excel_path)

    for header, name_to_roles in activities:
        if header not in df.columns:
            df[header] = pd.NA

        for name, roles in name_to_roles.items():
            mask = df.iloc[:, name_col_idx] == name
            if not mask.any():
                continue

            role_str = "、".join(roles)

            for row_idx in df.index[mask]:
                existing = df.at[row_idx, header]
                if pd.isna(existing) or existing == "":
                    df.at[row_idx, header] = role_str
                else:
                    merged = set(existing.split("、")) | set(roles)
                    df.at[row_idx, header] = "、".join(sorted(merged))

    df.to_excel(excel_path, index=False)
    print(f"✅ 更新完成 → {excel_path.absolute()}")


def main():
    raw_text = TEXT_PATH.read_text(ENCODING)
    activities = parse_activities(raw_text)
    update_excel(EXCEL_PATH, activities, NAME_COL)


if __name__ == "__main__":
    main()
