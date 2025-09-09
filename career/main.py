from colorama import Fore, Style
from tqdm import tqdm
import sys
import time

t = 0.2
delay_t = 0.005

# 定义逐字符打印函数
def typewriter(text, color=Style.RESET_ALL, delay=0.05):
    sys.stdout.write(color)  # 设置颜色
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(Style.RESET_ALL + "\n")  # 重置颜色并换行

# Title
def title():
    #typewriter("《从知识渴望到教育使命，我的计算机科学教授之路》", color=Fore.GREEN, delay=4*delay_t)
    print("=" * 80)

# Evaluation Process
def evaluation_start():
    typewriter("系统正在评估您的潜力，请稍候...", color=Fore.YELLOW, delay=4*delay_t)
    for _ in tqdm(range(100), desc="准备中", bar_format="{l_bar}{bar} [时间: {elapsed}]"):
        time.sleep(0.02)
    typewriter("评估开始...\n", color=Fore.CYAN, delay=4*delay_t)
    time.sleep(t)

# Story 1
def story_1():
    typewriter("评价维度: 学术兴趣\n", color=Fore.BLUE, delay=4*delay_t)
    time.sleep(2*t)
    book = "阿西莫夫的《基地》系列"
    inspiration = "机器人三大定律"
    typewriter("兴趣点1: 启蒙与追求", color=Fore.MAGENTA, delay=4*delay_t)
    typewriter(f"中学时，我读到 {Fore.YELLOW}{book}{Style.RESET_ALL}，其中『{Fore.YELLOW}{inspiration}{Style.RESET_ALL}』让我着迷。\n", delay=3*delay_t)
    mission = "探索人工智能如何服务人类，造福社会"
    typewriter(f"这让我立志研究 AI 对齐，追求知识，并以此 {Fore.GREEN}{mission}{Style.RESET_ALL}。\n", delay=3*delay_t)

# Story 2
def story_2():
    typewriter("评价维度: 责任感与教育使命\n", color=Fore.BLUE, delay=4*delay_t)
    time.sleep(2*t)
    teacher_message = "光环是理想自我和社会责任的结合"
    typewriter("兴趣点2: 光环与责任", color=Fore.MAGENTA, delay=4*delay_t)
    typewriter(f"高中历史课上，老师告诉我们：『{Fore.YELLOW}{teacher_message}{Style.RESET_ALL}』", delay=3*delay_t)
    mission = "科研之外，传递理念，为国家接续教育与技术火种"
    typewriter(f"这让我认识到教授的责任，不仅是研究，还要 {Fore.GREEN}{mission}{Style.RESET_ALL}。\n", delay=3*delay_t)

# Adjustments
def adjustments():
    typewriter("评价维度: 动态调整能力\n", color=Fore.BLUE, delay=4*delay_t)
    time.sleep(2*t)
    research_adjustment = {
        "背景": "从NUS回国后，发现自己需要增强情感感知能力与人文关怀，",
        "关注点": "人文关怀与领导力",
        "案例": [
            "团队成员出错时，应鼓励而非直接批评",
            "不要那么顽固，应多多听取他人的意见"
        ]
    }
    typewriter("成长路径的动态调整:", color=Fore.MAGENTA, delay=4*delay_t)
    typewriter(f"- 背景: {Fore.YELLOW}{research_adjustment['背景']}{Style.RESET_ALL}", delay=3*delay_t)
    typewriter(f"- 关注点: {Fore.YELLOW}{research_adjustment['关注点']}{Style.RESET_ALL}", delay=3*delay_t)
    typewriter("- 案例:", delay=3*delay_t)
    for case in research_adjustment["案例"]:
        typewriter(f"  * {Fore.GREEN}{case}{Style.RESET_ALL}", delay=3*delay_t)

# Professional Requirements
def professional_requirements():
    typewriter("评价维度: 职业要求匹配度\n", color=Fore.BLUE, delay=4*delay_t)
    time.sleep(2*t)
    mit = ["talent", "ideas", "value diversity", "collaborative"]
    xjtu = ["政治素质过硬", "师德师风高尚", "育人成效显著", "学术成就卓越"]
    hkust = [
        "Excellence",
        "Integrity",
        "Academic Freedom",
        "Local Commitment",
        "Contribute to the nation as a leading university"
    ]
    typewriter(f"- 麻省理工大学: {Fore.YELLOW}{mit}{Style.RESET_ALL}", delay=3*delay_t)
    typewriter(f"- 西安交通大学: {Fore.YELLOW}{xjtu}{Style.RESET_ALL}", delay=3*delay_t)
    typewriter(f"- 香港科技大学: {Fore.YELLOW}{hkust}{Style.RESET_ALL}\n", delay=3*delay_t)

# Academic Foundation
def academic_foundation():
    typewriter("评价维度: 学科基础\n", color=Fore.BLUE, delay=4*delay_t)
    time.sleep(2*t)
    academic_details = [
        {"课程": "计算机程序设计（拔尖班）", "成绩": 100},
        {"课程": "线性代数与解析几何", "成绩": 100},
        {"课程": "大学物理实验", "成绩": "A+ [TOP 1%]"},
        {"课程": "数据结构与算法（拔尖班）", "成绩": 98},
        {"课程": "计算机科学技术导论", "成绩": 97},
        {"课程": "大学物理一", "成绩": 96},
        {"课程": "托福强化", "成绩": 95}
    ]
    for detail in academic_details:
        typewriter(f"- {Fore.YELLOW}{detail['课程']}{Style.RESET_ALL}: {Fore.GREEN}{detail['成绩']}{Style.RESET_ALL}", delay=3*delay_t)

# Evaluation
def evaluate_score():
    scores = {
        "学术基础": 95,
        "研究能力": 90,
        "领导力": 88,
        "动态调整": 93
    }
    typewriter("评估细节:", color=Fore.BLUE, delay=4*delay_t)
    for key, value in scores.items():
        typewriter(f"- {Fore.YELLOW}{key}{Style.RESET_ALL}: {Fore.GREEN}{value}{Style.RESET_ALL}", delay=3*delay_t)
    overall_score = sum(scores.values()) // len(scores)
    typewriter(f"\n最终评估得分: {Fore.YELLOW}{overall_score}/100{Style.RESET_ALL}\n", delay=4*delay_t)
    return overall_score

# Conclusion
def conclusion():
    typewriter("总结:\n", color=Fore.GREEN, delay=4*delay_t)
    typewriter("您的目标不仅是成为一名教授，更是通过知识报效祖国，为国铸剑，为国育人，为人类科技发展贡献力量。\n", delay=3*delay_t)
    overall_score = evaluate_score()
    if overall_score >= 90:
        typewriter("系统评估: 您具备成为一名计算机教授的潜力，并能够为国家与民族贡献力量！", color=Fore.CYAN, delay=4*delay_t)
    else:
        typewriter("系统评估: 您的潜力尚可，但需要进一步提升科研与教育能力。", color=Fore.RED, delay=4*delay_t)

# Run the full program
def main():
    evaluation_start()
    title()
    story_1()
    story_2()
    professional_requirements()
    academic_foundation()
    adjustments()
    conclusion()

# Execute the program
main()
