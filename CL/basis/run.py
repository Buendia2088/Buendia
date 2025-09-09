import subprocess, sys, ast, itertools, csv, os, re
import numpy as np

# ---------------- Configurable grid ----------------
scens = ["CIL", "TIL", "DIL"]
methods = {
  "baseline":          "--no-replay --no-regularization --no-distill --no-proj",
  "replay":            "--replay --no-regularization --no-distill --no-proj",
  "ewc":               "--no-replay --regularization --no-distill --no-proj",
  "kd":                "--no-replay --no-regularization --distill --no-proj",
  "proj":              "--no-replay --no-regularization --no-distill --proj",
  "replay_ewc":        "--replay --regularization --no-distill --no-proj",
  "replay_kd":         "--replay --no-regularization --distill --no-proj",
  "replay_proj":       "--replay --no-regularization --no-distill --proj",
  "ewc_kd":            "--no-replay --regularization --distill --no-proj",
  "replay_ewc_kd":     "--replay --regularization --distill --no-proj",
}
# 你可以按需改这里
EPOCHS, TASKS, BATCH = 3, 5, 128
PY = sys.executable or "python"

os.makedirs("logs", exist_ok=True)

# ---------------- Helpers ----------------
def parse_metrics(stdout: str):
    """Extract {'avg_acc': ..., 'BWT': ...} from a line starting with 'Metrics'."""
    for line in stdout.splitlines():
        if "Metrics" in line:
            m = re.search(r"Metrics\s*(\{.*\})\s*$", line.strip())
            if m:
                try:
                    return ast.literal_eval(m.group(1))
                except Exception:
                    # 容忍尾随逗号/空格
                    try:
                        text = m.group(1).rstrip(", ")
                        return ast.literal_eval(text)
                    except Exception:
                        pass
    return {}

def _first_matrix_from_block(block_text: str, prefer_n: int | None = None):
    """
    从给定文本块里提取一个近似 N×N 的数字矩阵。
    优先用 prefer_n（通常等于 tasks）；否则自动探测最大正方形。
    返回 (np.ndarray or None, N used or None)
    """
    nums = re.findall(r"-?\d+(?:\.\d+)?", block_text)
    if not nums:
        return None, None

    # 优先使用 prefer_n
    if prefer_n is not None and len(nums) >= prefer_n * prefer_n:
        vals = list(map(float, nums[: prefer_n * prefer_n]))
        try:
            return np.array(vals, dtype=float).reshape(prefer_n, prefer_n), prefer_n
        except Exception:
            pass

    # 自动探测：从较大 N 往下试
    max_try = min(20, int(len(nums) ** 0.5) + 1)  # 容错上限
    for N in range(max_try, 1, -1):
        if len(nums) >= N * N:
            vals = list(map(float, nums[: N * N]))
            try:
                return np.array(vals, dtype=float).reshape(N, N), N
            except Exception:
                continue
    return None, None

def parse_acc_matrix(stdout: str, tasks: int):
    """
    鲁棒解析：
    1) 先大小写不敏感匹配含 "acc" 与 "matrix" 的行作为起点，抓到 'Metrics' 之前的块；
    2) 无起点则退化为扫描输出末尾 200 行；
    3) 允许 prefer_n=tasks，失败则自动探测 N。
    """
    lines = stdout.splitlines()

    # 1) 大小写不敏感地找 'acc.*matrix'
    start = None
    for i, ln in enumerate(lines):
        if re.search(r"acc.*matrix", ln, flags=re.I):
            start = i
            break

    candidates = []
    if start is not None:
        block = []
        gap = 0
        for ln in lines[start + 1 :]:
            if "Metrics" in ln:
                break
            if ln.strip() == "":
                gap += 1
                # 避免把后面的无关段落全部吞进去
                if gap > 2:
                    break
                block.append(ln)
                continue
            block.append(ln)
            gap = 0
        cand = "\n".join(block).strip()
        if cand:
            candidates.append(cand)

    # 2) 追加尾部兜底（最后 200 行）
    tail = "\n".join(lines[-200:])
    candidates.append(tail)

    # 3) 尝试从候选块里提取矩阵
    for blk in candidates:
        # 去除可能的包装如 tensor( / array( / np.array(
        cleaned = re.sub(r"\btensor\s*\(|\barray\s*\(|\bnp\.array\s*\(", "(", blk)
        # 容忍引号/方括号/逗号混杂；直接抓数字更稳
        mat, n_used = _first_matrix_from_block(cleaned, prefer_n=tasks)
        if mat is not None:
            return mat
    return None

def save_matrix_csv(matrix: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["task\\task"] + [f"T{j+1}" for j in range(matrix.shape[1])]
        w.writerow(header)
        for i in range(matrix.shape[0]):
            w.writerow([f"T{i+1}"] + [f"{matrix[i, j]:.6f}" for j in range(matrix.shape[1])])

# ---------------- Run grid ----------------
rows = [("method","scenario","avg_acc","BWT","matrix_csv")]
for s in scens:
    for name, flags in methods.items():
        cmd = f'{PY} continual_minimal.py --scenario {s} --tasks {TASKS} --epochs {EPOCHS} --batch {BATCH} {flags}'
        print(">>>", cmd, flush=True)
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        stdout, stderr = p.stdout, p.stderr
        # 保存原始日志，便于诊断
        out_path = os.path.join("logs", f"{s}_{name}.out")
        err_path = os.path.join("logs", f"{s}_{name}.err")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write(stdout)
        if stderr:
            with open(err_path, "w", encoding="utf-8", newline="") as f:
                f.write(stderr)

        if p.returncode != 0:
            print(f"[WARN] Return code {p.returncode} for {name}/{s}. See {err_path}", file=sys.stderr)

        metrics = parse_metrics(stdout)
        avg = metrics.get("avg_acc", "NA")
        bwt = metrics.get("BWT", "NA")

        mat = parse_acc_matrix(stdout, TASKS)
        mat_path = ""
        if mat is not None:
            mat_path = os.path.join("matrices", f"matrix_{s}_{name}.csv")
            save_matrix_csv(mat, mat_path)
            print(f"[OK] Saved matrix -> {mat_path}")
        else:
            print(f"[WARN] Acc matrix not found for {name}/{s}. See {out_path} for stdout")

        rows.append((name, s, avg, bwt, mat_path))

with open("results.csv","w",newline="") as f:
    csv.writer(f).writerows(rows)

print("\nSaved to results.csv, matrices/*.csv, and logs/*.out/.err")
