#!/usr/bin/env python3
import itertools
import subprocess

def main():
    # 读取 solution.txt 的所有行
    with open("solution.txt", "r") as f:
        lines = f.readlines()

    # 遍历 1-6 的所有排列（排列顺序按照字典序）
    for perm in itertools.permutations(["1", "2", "3", "4", "5", "6"]):
        # 构造排列字符串，例如 "4 6 1 2 3 5"
        perm_line = " ".join(perm) + "\n"
        # 修改第六行（注意：列表索引从 0 开始，第六行为索引5）
        lines[5] = perm_line

        # 保存文件
        with open("solution.txt", "w") as f:
            f.writelines(lines)

        # 运行 bomb 程序，将 solution.txt 作为输入
        try:
            # 使用 shell 运行命令，并捕获输出（包括 stdout 和 stderr）
            result = subprocess.run("./bomb < solution.txt", shell=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    universal_newlines=True)
            output = result.stdout.strip()
        except Exception as e:
            print("运行 bomb 出错:", e)
            continue

        # 输出调试信息
        last_line = output.splitlines()[-1] if output.splitlines() else ""
        print(f"尝试排列 {perm_line.strip()}，bomb 输出最后一行: {last_line}")

        # 如果最后一行为 "(100/100)" 则认为找到了正确排列
        if last_line == "(100/100)":
            print("正确排列为:", perm_line.strip())
            break

if __name__ == "__main__":
    main()

