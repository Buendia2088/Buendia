#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# ----------- 正则：更宽松，考虑键名后的引号、负号、科学计数法 -----------
FLOAT = r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'

AVG_PATTERNS = [
    rf"\bavg[_\s-]*acc(?:uracy)?\b[\'\"]?\s*[:=]\s*({FLOAT})",
    rf"\bAverage\s+Accuracy\b[\'\"]?\s*[:=]\s*({FLOAT})",
    rf"\bAvgAcc\b[\'\"]?\s*[:=]\s*({FLOAT})",
    rf"\bACC\b[\'\"]?\s*[:=]\s*({FLOAT})",
]
BWT_PATTERNS = [
    rf"\bBWT\b[\'\"]?\s*[:=]\s*({FLOAT})",
    rf"\bBackward\W*Transfer\b[\'\"]?\s*[:=]\s*({FLOAT})",
]

FILE_PAT = re.compile(r'(?P<scenario>[^_/]+)_(?P<method>.+)$', re.IGNORECASE)

# Acc matrix 区块（尽量兼容多行）
MATRIX_BLOCK_PAT = re.compile(
    r"Acc\s*matrix.*?tensor\(\s*\[([\s\S]*?)\]\s*\)", re.IGNORECASE
)
ROW_PAT = re.compile(r"\[([^\]]+)\]")  # 抓每一行的逗号分隔元素

def find_last_float(text: str, patterns: List[str]) -> Optional[float]:
    val = None
    for pat in patterns:
        ms = list(re.finditer(pat, text, flags=re.IGNORECASE))
        if ms:
            try:
                val = float(ms[-1].group(1))
            except Exception:
                pass
    return val

def parse_matrix_lastrow_mean(text: str) -> Optional[float]:
    """
    从 Acc matrix 的 tensor 区块中，计算**最后一行**的均值（忽略 NaN）。
    作为 avg_acc 的回退方案。
    """
    m = MATRIX_BLOCK_PAT.search(text)
    if not m:
        return None
    block = m.group(1)  # 多行的元素区块
    rows = ROW_PAT.findall(block)
    if not rows:
        return None
    last = rows[-1]
    elems = []
    for tok in last.split(","):
        t = tok.strip()
        if t.lower() == "nan":
            elems.append(np.nan)
        else:
            try:
                elems.append(float(t))
            except Exception:
                elems.append(np.nan)
    arr = np.array(elems, dtype=float)
    if arr.size == 0:
        return None
    mean_val = np.nanmean(arr)
    if np.isnan(mean_val):
        return None
    return float(mean_val)

def parse_metrics_from_text(text: str) -> Dict[str, Optional[float]]:
    avg = find_last_float(text, AVG_PATTERNS)
    bwt = find_last_float(text, BWT_PATTERNS)
    if avg is None:
        # 回退：用 Acc matrix 的最后一行均值
        avg = parse_matrix_lastrow_mean(text)
    return {"avg_acc": avg, "BWT": bwt}

def parse_name_from_path(p: Path) -> Optional[Tuple[str, str]]:
    m = FILE_PAT.match(p.stem)  # 例如 CIL_replay_proj
    if not m:
        return None
    return m.group("scenario"), m.group("method")

def save_bar_by_scenario(df, scenario_col, method_col, value_col, outdir: Path, title_prefix: str):
    grouped = (
        df.groupby([scenario_col, method_col])[value_col]
          .agg(['mean', 'std', 'count'])
          .reset_index()
    )
    for scen, sub in grouped.groupby(scenario_col):
        sub = sub.sort_values('mean', ascending=False).reset_index(drop=True)
        fig = plt.figure()
        plt.bar(sub[method_col].astype(str).values, sub['mean'].values,
                yerr=sub['std'].fillna(0.0).values)
        plt.ylabel(value_col)
        plt.xlabel(method_col)
        plt.title(f"{title_prefix} — {scen}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(outdir / f"{value_col}_by_method__{scen}.png", dpi=180)
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser("Visualize continual learning results from logs/*.out")
    ap.add_argument("--logs", default="logs", help="Directory containing *.out/*.txt/*.log")
    ap.add_argument("--outdir", default="viz_out", help="Output directory for charts and CSVs")
    args = ap.parse_args()

    logs_dir = Path(args.logs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not logs_dir.exists():
        print(f"[ERROR] Logs directory not found: {logs_dir}", file=sys.stderr)
        sys.exit(1)

    files = []
    for ext in ("*.out", "*.txt", "*.log"):
        files += list(logs_dir.glob(ext))
    if not files:
        print(f"[ERROR] No .out/.txt/.log in {logs_dir}", file=sys.stderr)
        sys.exit(2)

    rows = []
    for f in files:
        nm = parse_name_from_path(f)
        if nm is None:
            continue
        scen, meth = nm
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = f.read_text(errors="ignore")
        m = parse_metrics_from_text(text)
        rows.append({
            "scenario": scen,
            "method": meth,
            "avg_acc": m["avg_acc"],
            "BWT": m["BWT"],
            "file": f.name,
        })

    df = pd.DataFrame(rows)
    # 过滤至少要有 avg_acc
    df = df[~df["avg_acc"].isna()].copy()
    if df.empty:
        print("[ERROR] All rows missing avg_acc; check regex or log format.", file=sys.stderr)
        sys.exit(3)

    print(f"[INFO] Parsed rows: {len(df)}")
    print(df.head(10).to_string(index=False))

    metrics = ["avg_acc"] + (["BWT"] if "BWT" in df.columns else [])
    # 汇总
    agg = (
        df.groupby(["scenario","method"])[metrics]
          .agg(['mean','std','count'])
          .reset_index()
    )
    agg.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c for c in agg.columns]
    agg_out = outdir / "summary_results.csv"
    agg.to_csv(agg_out, index=False)
    print(f"[OK] Saved summary -> {agg_out}")

    # 总榜
    overall = (
        df.groupby("method")[metrics]
          .mean(numeric_only=True)
          .reset_index()
          .sort_values(["avg_acc","BWT"] if "BWT" in df.columns else ["avg_acc"],
                       ascending=[False, False] if "BWT" in df.columns else [False])
          .reset_index(drop=True)
    )
    overall_out = outdir / "leaderboard_overall.csv"
    overall.to_csv(overall_out, index=False)
    print(f"[OK] Saved overall leaderboard -> {overall_out}")

    # 每场景 Top-3
    grouped = (
        df.groupby(["scenario","method"])[metrics]
          .mean(numeric_only=True)
          .reset_index()
          .sort_values(["scenario","avg_acc"] + (["BWT"] if "BWT" in df.columns else []),
                       ascending=[True, False] + ([False] if "BWT" in df.columns else []))
    )
    top3 = grouped.groupby("scenario").head(3).reset_index(drop=True)
    top3_out = outdir / "leaderboard_top3_per_scenario.csv"
    top3.to_csv(top3_out, index=False)
    print(f"[OK] Saved per-scenario top-3 -> {top3_out}")

    # 图
    save_bar_by_scenario(df, "scenario", "method", "avg_acc", outdir, "Avg Accuracy by Method")
    if "BWT" in df.columns and not df["BWT"].isna().all():
        save_bar_by_scenario(df, "scenario", "method", "BWT", outdir, "BWT by Method")

    # 相关性（可选）
    if "BWT" in df.columns and not df["BWT"].isna().all():
        r_all = df[["avg_acc","BWT"]].dropna().corr().iloc[0,1]
        print(f"[INFO] Overall corr(avg_acc, BWT) = {r_all:.3f}")
        for scen, sub in df.groupby("scenario"):
            sub2 = sub[["avg_acc","BWT"]].dropna()
            if len(sub2) >= 3:
                r = sub2.corr().iloc[0,1]
                print(f"[INFO] {scen}: corr(avg_acc, BWT) = {r:.3f}")

    print(f"[DONE] Charts saved in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
