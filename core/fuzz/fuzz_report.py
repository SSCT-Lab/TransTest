# core/fuzz_report.py
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------
# 判定与统计工具
# -----------------------

def is_pass(max_abs_err: float, max_rel_err: float, atol: float, rtol: float) -> bool:
    """混合容差通过规则：绝对误差<=atol 或 相对误差<=rtol"""
    return (max_abs_err <= atol) or (max_rel_err <= rtol)

def percentiles(values: np.ndarray, ps=(50, 95, 99)) -> Dict[str, float]:
    if len(values) == 0:
        return {f"P{p}": float("nan") for p in ps}
    return {f"P{p}": float(np.percentile(values, p)) for p in ps}

def safe_float_col(df: pd.DataFrame, col: str) -> pd.Series:
    """将字符串/科学计数等安全转为 float"""
    return pd.to_numeric(df[col], errors="coerce")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -----------------------
# 图像生成（单图/不设颜色）
# -----------------------

def plot_pass_rate_bar(per_family_df: pd.DataFrame, out_png: Path, title="Pass Rate by Family"):
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    x = np.arange(len(per_family_df))
    ax.bar(x, per_family_df["pass_rate"].values)
    ax.set_xticks(x)
    ax.set_xticklabels(per_family_df["family"].tolist(), rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Pass Rate")
    ax.set_title(title)
    for i, v in enumerate(per_family_df["pass_rate"].values):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def plot_box(values_by_family: Dict[str, np.ndarray], out_png: Path, ylabel: str, title: str):
    families = list(values_by_family.keys())
    data = [values_by_family[f] for f in families]

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(np.arange(1, len(families) + 1))
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def plot_abs_rel_scatter(df: pd.DataFrame, out_png: Path, title="Abs vs Rel Error (log scale)"):
    # 只画通过计算的数值
    x = df["max_abs_err"].values
    y = df["max_rel_err"].values
    # 避免 log(0)
    x = np.clip(x, 1e-12, None)
    y = np.clip(y, 1e-12, None)

    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("max_abs_err (log)")
    ax.set_ylabel("max_rel_err (log)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# -----------------------
# 主流程
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="fuzz 结果 CSV 路径")
    ap.add_argument("--out-dir", default="reports/fuzzing", help="输出目录")
    ap.add_argument("--atol", type=float, default=3e-5, help="绝对误差阈值 (default: 3e-5)")
    ap.add_argument("--rtol", type=float, default=1e-3, help="相对误差阈值 (default: 1e-3)")
    ap.add_argument("--topk", type=int, default=20, help="导出 Top-K outliers")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(args.csv)
    # 规范列名
    for c in ["max_abs_err", "max_rel_err"]:
        df[c] = safe_float_col(df, c)

    # 判定通过
    df["pass"] = df.apply(
        lambda r: is_pass(r["max_abs_err"], r["max_rel_err"], args.atol, args.rtol),
        axis=1,
    )

    # ------- 汇总（每个 family）-------
    rows = []
    for fam, g in df.groupby("family"):
        n = len(g)
        n_pass = int(g["pass"].sum())
        pass_rate = n_pass / n if n else float("nan")

        abs_ps = percentiles(g["max_abs_err"].values)
        rel_ps = percentiles(g["max_rel_err"].values)

        rows.append({
            "family": fam,
            "n": n,
            "pass": n_pass,
            "pass_rate": pass_rate,
            "abs_P50": abs_ps["P50"],
            "abs_P95": abs_ps["P95"],
            "abs_P99": abs_ps["P99"],
            "rel_P50": rel_ps["P50"],
            "rel_P95": rel_ps["P95"],
            "rel_P99": rel_ps["P99"],
        })

    per_family = pd.DataFrame(rows).sort_values(by="pass_rate", ascending=False)
    per_family_csv = out_dir / "per_family_summary.csv"
    per_family.to_csv(per_family_csv, index=False)

    # ------- Top-K outliers -------
    top_abs = df.sort_values(by="max_abs_err", ascending=False).head(args.topk)
    top_rel = df.sort_values(by="max_rel_err", ascending=False).head(args.topk)

    # 额外导出 params 为可读 JSON
    def parse_params(s):
        try:
            return json.loads(s.replace("'", "\""))
        except Exception:
            return s

    for sub in (top_abs, top_rel):
        sub["params_json"] = sub["params"].apply(parse_params)

    top_abs_csv = out_dir / f"top{args.topk}_abs_err.csv"
    top_rel_csv = out_dir / f"top{args.topk}_rel_err.csv"
    top_abs.to_csv(top_abs_csv, index=False)
    top_rel.to_csv(top_rel_csv, index=False)

    # ------- 图像 -------
    # 1) 通过率条形图
    pass_png = out_dir / "pass_rate_bar.png"
    plot_pass_rate_bar(per_family, pass_png)

    # 2) 绝对误差箱线图
    abs_by_fam = {fam: g["max_abs_err"].values for fam, g in df.groupby("family")}
    abs_box_png = out_dir / "abs_err_box.png"
    plot_box(abs_by_fam, abs_box_png, ylabel="max_abs_err", title="Absolute Error by Family")

    # 3) 相对误差箱线图
    rel_by_fam = {fam: g["max_rel_err"].values for fam, g in df.groupby("family")}
    rel_box_png = out_dir / "rel_err_box.png"
    plot_box(rel_by_fam, rel_box_png, ylabel="max_rel_err", title="Relative Error by Family")

    # 4) 误差散点图（log）
    scatter_png = out_dir / "abs_vs_rel_scatter.png"
    plot_abs_rel_scatter(df, scatter_png)

    # ------- Markdown 小报告（PPT 可抄）-------
    md = []
    md.append("# Fuzzing 对齐结果小结\n")
    md.append(f"- 输入文件：`{Path(args.csv).name}`")
    md.append(f"- 判定规则：`abs<= {args.atol:g}` **或** `rel<= {args.rtol:g}`")
    md.append("")
    md.append("## 1) 每家族通过率与误差分位\n")
    md.append(per_family.to_markdown(index=False))
    md.append("")
    md.append("## 2) 图表（PPT 可直接贴）\n")
    md.append(f"![pass rate]({pass_png.name})")
    md.append(f"![abs box]({abs_box_png.name})")
    md.append(f"![rel box]({rel_box_png.name})")
    md.append(f"![scatter]({scatter_png.name})")
    md.append("")
    md.append("## 3) Top 异常样本（导出 CSV）\n")
    md.append(f"- 绝对误差 Top-{args.topk}：`{top_abs_csv.name}`")
    md.append(f"- 相对误差 Top-{args.topk}：`{top_rel_csv.name}`")
    md.append("")
    md.append("> 注：相对误差异常但绝对误差极小的样本，多属于“零附近放大”现象，可结合散点图（log-log）辨析。")
    report_md = out_dir / "fuzz_report.md"
    report_md.write_text("\n".join(md), encoding="utf-8")

    # 控制台友好输出
    print(f"[OK] Per-family summary  → {per_family_csv}")
    print(f"[OK] Top-K abs errors    → {top_abs_csv}")
    print(f"[OK] Top-K rel errors    → {top_rel_csv}")
    print(f"[OK] Charts saved        → {pass_png}, {abs_box_png}, {rel_box_png}, {scatter_png}")
    print(f"[OK] Markdown report     → {report_md}")

if __name__ == "__main__":
    main()
