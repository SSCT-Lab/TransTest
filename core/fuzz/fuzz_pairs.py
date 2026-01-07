# core/fuzzing.py
import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
# 可选：在无 GPU 环境下也可以跑
os_environ = {}
try:
    import torch
    import torch.nn.functional as F
except Exception as e:
    print("[ERROR] import torch 失败：", e)
    sys.exit(1)

try:
    import tensorflow as tf
    # 禁止 GPU 以减少环境差异（如需用 GPU，可注释掉）
    tf.config.set_visible_devices([], "GPU")
except Exception as e:
    print("[ERROR] import tensorflow 失败：", e)
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True, parents=True)


# ---------------------------- 工具 & 数据结构 ----------------------------
# SUPPORTED_FAMILIES = ["conv2d", "dense", "relu", "pool2d", "batchnorm"]
SUPPORTED_FAMILIES = ["conv2d", "dense", "relu", "pool2d", "batchnorm", "linear", "attention"]


@dataclass
class TrialResult:
    family: str
    params: Dict
    status: str              # "ok" | "value_mismatch" | "shape_mismatch" | "exception_mismatch" | "both_exception"
    atol: float
    rtol: float
    tf_exception: str = ""
    pt_exception: str = ""
    pt_shape: Tuple = ()
    tf_shape: Tuple = ()
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0


def np_allclose(a: np.ndarray, b: np.ndarray, rtol=1e-4, atol=1e-5) -> Tuple[bool, float, float]:
    if a.shape != b.shape:
        return False, float("inf"), float("inf")
    diff = np.abs(a - b)
    max_abs = float(diff.max()) if diff.size > 0 else 0.0
    # 避免除零
    denom = np.maximum(np.abs(b), np.finfo(a.dtype).eps)
    max_rel = float((diff / denom).max()) if diff.size > 0 else 0.0
    ok = np.all(diff <= (atol + rtol * np.abs(b)))
    return bool(ok), max_abs, max_rel


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return np.array(x)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


# ---------------------------- 形状与参数生成 ----------------------------
def rand_int(lo, hi):
    return int(np.random.randint(lo, hi + 1))


def gen_conv2d_case():
    # NCHW (torch), NHWC (tf)
    N = rand_int(1, 3)
    C_in = rand_int(1, 8)
    C_out = rand_int(1, 8)
    H = rand_int(8, 32)
    W = rand_int(8, 32)
    k = rand_int(1, 5)       # kernel size
    s = rand_int(1, 2)       # stride
    # 统一使用 VALID padding，避免 SAME 的边界差异
    return dict(N=N, C_in=C_in, C_out=C_out, H=H, W=W, k=k, s=s, padding="VALID")


def gen_dense_case():
    N = rand_int(1, 8)
    IN = rand_int(1, 64)
    OUT = rand_int(1, 64)
    return dict(N=N, IN=IN, OUT=OUT)


def gen_relu_case():
    N = rand_int(1, 8)
    C = rand_int(1, 16)
    H = rand_int(4, 32)
    W = rand_int(4, 32)
    return dict(shape=(N, C, H, W))


def gen_pool2d_case():
    N = rand_int(1, 3)
    C = rand_int(1, 8)
    H = rand_int(8, 32)
    W = rand_int(8, 32)
    k = rand_int(2, 4)
    s = k  # 常用设置：stride=kernel_size
    return dict(N=N, C=C, H=H, W=W, k=k, s=s, padding="VALID", mode="max")


def gen_batchnorm_case():
    N = rand_int(1, 4)
    C = rand_int(1, 16)
    H = rand_int(4, 16)
    W = rand_int(4, 16)
    eps = 1e-5
    return dict(N=N, C=C, H=H, W=W, eps=eps)

def gen_linear_case():
    N = rand_int(1, 8)
    IN = rand_int(1, 128)
    OUT = rand_int(1, 128)
    use_bias = random.choice([True, False])
    return dict(N=N, IN=IN, OUT=OUT, use_bias=use_bias)


def gen_attention_case():
    B = rand_int(1, 2)   # batch size
    H = rand_int(1, 4)   # num_heads
    S = rand_int(4, 16)  # seq_len
    D = rand_int(4, 32)  # head_dim
    use_mask = random.choice([True, False])
    return dict(B=B, H=H, S=S, D=D, use_mask=use_mask)

def run_linear(params, rtol, atol) -> TrialResult:
    N, IN, OUT, use_bias = params["N"], params["IN"], params["OUT"], params["use_bias"]
    x_pt = torch.randn(N, IN, dtype=torch.float32)
    W_pt = torch.randn(OUT, IN, dtype=torch.float32)
    b_pt = torch.randn(OUT, dtype=torch.float32) if use_bias else torch.zeros(OUT, dtype=torch.float32)

    x_tf = tf.convert_to_tensor(x_pt.numpy(), dtype=tf.float32)
    W_tf = tf.convert_to_tensor(W_pt.numpy().T, dtype=tf.float32)
    b_tf = tf.convert_to_tensor(b_pt.numpy(), dtype=tf.float32)

    tr = TrialResult(family="linear", params=params, status="ok", rtol=rtol, atol=atol)
    try:
        y_pt = F.linear(x_pt, W_pt, bias=b_pt)
        y_pt_np = to_numpy(y_pt)
        tr.pt_shape = tuple(y_pt_np.shape)
    except Exception as e:
        tr.pt_exception = str(e)

    try:
        y_tf = tf.linalg.matmul(x_tf, W_tf)
        if use_bias:
            y_tf = tf.nn.bias_add(y_tf, b_tf)
        y_tf_np = to_numpy(y_tf)
        tr.tf_shape = tuple(y_tf_np.shape)
    except Exception as e:
        tr.tf_exception = str(e)

    if tr.tf_exception and tr.pt_exception:
        tr.status = "both_exception"; return tr
    if tr.tf_exception or tr.pt_exception:
        tr.status = "exception_mismatch"; return tr
    if tr.pt_shape != tr.tf_shape:
        tr.status = "shape_mismatch"; return tr

    ok, max_abs, max_rel = np_allclose(y_pt_np, y_tf_np, rtol=rtol, atol=atol)
    tr.max_abs_err, tr.max_rel_err = max_abs, max_rel
    tr.status = "ok" if ok else "value_mismatch"
    return tr


def run_attention(params, rtol, atol) -> TrialResult:
    B, H, S, D, use_mask = params["B"], params["H"], params["S"], params["D"], params["use_mask"]
    scale = 1.0 / math.sqrt(D)
    tr = TrialResult(family="attention", params=params, status="ok", rtol=rtol, atol=atol)

    q_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    k_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    v_pt = torch.randn(B, H, S, D, dtype=torch.float32)

    q_tf = tf.convert_to_tensor(q_pt.numpy(), dtype=tf.float32)
    k_tf = tf.convert_to_tensor(k_pt.numpy(), dtype=tf.float32)
    v_tf = tf.convert_to_tensor(v_pt.numpy(), dtype=tf.float32)

    # Optional mask
    mask_np = None
    if use_mask:
        mask_np = (np.random.rand(B, 1, S, S) > 0.7).astype(np.float32) * -1e9
        mask_pt = torch.from_numpy(mask_np)
        mask_tf = tf.convert_to_tensor(mask_np, dtype=tf.float32)
    else:
        mask_pt = mask_tf = None

    try:
        # PyTorch
        scores_pt = torch.matmul(q_pt, k_pt.transpose(-2, -1)) * scale
        if mask_pt is not None:
            scores_pt += mask_pt
        weights_pt = F.softmax(scores_pt, dim=-1)
        y_pt = torch.matmul(weights_pt, v_pt)
        y_pt_np = to_numpy(y_pt)
        tr.pt_shape = tuple(y_pt_np.shape)
    except Exception as e:
        tr.pt_exception = str(e)

    try:
        # TensorFlow
        scores_tf = tf.matmul(q_tf, k_tf, transpose_b=True) * scale
        if mask_tf is not None:
            scores_tf += mask_tf
        weights_tf = tf.nn.softmax(scores_tf, axis=-1)
        y_tf = tf.matmul(weights_tf, v_tf)
        y_tf_np = to_numpy(y_tf)
        tr.tf_shape = tuple(y_tf_np.shape)
    except Exception as e:
        tr.tf_exception = str(e)

    if tr.tf_exception and tr.pt_exception:
        tr.status = "both_exception"; return tr
    if tr.tf_exception or tr.pt_exception:
        tr.status = "exception_mismatch"; return tr
    if tr.pt_shape != tr.tf_shape:
        tr.status = "shape_mismatch"; return tr

    ok, max_abs, max_rel = np_allclose(y_pt_np, y_tf_np, rtol=rtol, atol=atol)
    tr.max_abs_err, tr.max_rel_err = max_abs, max_rel
    tr.status = "ok" if ok else "value_mismatch"
    return tr

# ---------------------------- 家族执行器 ----------------------------
def run_conv2d(params, rtol, atol) -> TrialResult:
    N, C_in, C_out, H, W, k, s = params["N"], params["C_in"], params["C_out"], params["H"], params["W"], params["k"], params["s"]
    padding = params["padding"]  # VALID

    # 生成相同的随机输入 & 权重
    x_pt = torch.randn(N, C_in, H, W, dtype=torch.float32)
    w_pt = torch.randn(C_out, C_in, k, k, dtype=torch.float32)
    b_pt = torch.zeros(C_out, dtype=torch.float32)  # 统一不使用偏置，或置0

    # TF 需要 NHWC 输入，权重 [kH, kW, inC, outC]
    x_tf = tf.convert_to_tensor(np.transpose(x_pt.numpy(), (0, 2, 3, 1)), dtype=tf.float32)
    w_tf = tf.convert_to_tensor(np.transpose(w_pt.numpy(), (2, 3, 1, 0)), dtype=tf.float32)
    b_tf = tf.zeros([C_out], dtype=tf.float32)

    # 运行
    tr = TrialResult(family="conv2d", params=params, status="ok", rtol=rtol, atol=atol)

    # Torch
    try:
        y_pt = F.conv2d(x_pt, w_pt, bias=None, stride=s, padding=0)  # VALID
        y_pt_np = to_numpy(y_pt)
        tr.pt_shape = tuple(y_pt_np.shape)
    except Exception as e:
        tr.pt_exception = str(e)

    # TF
    try:
        y_tf = tf.nn.conv2d(x_tf, w_tf, strides=[1, s, s, 1], padding=padding)  # VALID
        y_tf_np = to_numpy(y_tf)
        # 转回 NCHW 以便比较
        y_tf_np = np.transpose(y_tf_np, (0, 3, 1, 2))
        tr.tf_shape = tuple(y_tf_np.shape)
    except Exception as e:
        tr.tf_exception = str(e)

    # 结果判断
    if tr.tf_exception and tr.pt_exception:
        tr.status = "both_exception"
        return tr
    if tr.tf_exception or tr.pt_exception:
        tr.status = "exception_mismatch"
        return tr
    if tr.pt_shape != tr.tf_shape:
        tr.status = "shape_mismatch"
        return tr

    ok, max_abs, max_rel = np_allclose(y_pt_np, y_tf_np, rtol=rtol, atol=atol)
    tr.max_abs_err, tr.max_rel_err = max_abs, max_rel
    tr.status = "ok" if ok else "value_mismatch"
    return tr


def run_dense(params, rtol, atol) -> TrialResult:
    N, IN, OUT = params["N"], params["IN"], params["OUT"]
    x_pt = torch.randn(N, IN, dtype=torch.float32)
    W_pt = torch.randn(OUT, IN, dtype=torch.float32)  # torch: (out, in)
    b_pt = torch.zeros(OUT, dtype=torch.float32)

    x_tf = tf.convert_to_tensor(x_pt.numpy(), dtype=tf.float32)
    W_tf = tf.convert_to_tensor(W_pt.numpy().T, dtype=tf.float32)  # tf: (in, out)
    b_tf = tf.zeros([OUT], dtype=tf.float32)

    tr = TrialResult(family="dense", params=params, status="ok", rtol=rtol, atol=atol)
    try:
        y_pt = F.linear(x_pt, W_pt, bias=None)
        y_pt_np = to_numpy(y_pt)
        tr.pt_shape = tuple(y_pt_np.shape)
    except Exception as e:
        tr.pt_exception = str(e)

    try:
        y_tf = tf.matmul(x_tf, W_tf)
        y_tf_np = to_numpy(y_tf)
        tr.tf_shape = tuple(y_tf_np.shape)
    except Exception as e:
        tr.tf_exception = str(e)

    if tr.tf_exception and tr.pt_exception:
        tr.status = "both_exception"; return tr
    if tr.tf_exception or tr.pt_exception:
        tr.status = "exception_mismatch"; return tr
    if tr.pt_shape != tr.tf_shape:
        tr.status = "shape_mismatch"; return tr

    ok, max_abs, max_rel = np_allclose(y_pt_np, y_tf_np, rtol=rtol, atol=atol)
    tr.max_abs_err, tr.max_rel_err = max_abs, max_rel
    tr.status = "ok" if ok else "value_mismatch"
    return tr


def run_relu(params, rtol, atol) -> TrialResult:
    N, C, H, W = params["shape"]
    x_pt = torch.randn(N, C, H, W, dtype=torch.float32)
    x_tf = tf.convert_to_tensor(np.transpose(x_pt.numpy(), (0, 2, 3, 1)), dtype=tf.float32)

    tr = TrialResult(family="relu", params=params, status="ok", rtol=rtol, atol=atol)
    try:
        y_pt = F.relu(x_pt)
        y_pt_np = to_numpy(y_pt)
        tr.pt_shape = tuple(y_pt_np.shape)
    except Exception as e:
        tr.pt_exception = str(e)

    try:
        y_tf = tf.nn.relu(x_tf)
        y_tf_np = to_numpy(y_tf)
        y_tf_np = np.transpose(y_tf_np, (0, 3, 1, 2))
        tr.tf_shape = tuple(y_tf_np.shape)
    except Exception as e:
        tr.tf_exception = str(e)

    if tr.tf_exception and tr.pt_exception:
        tr.status = "both_exception"; return tr
    if tr.tf_exception or tr.pt_exception:
        tr.status = "exception_mismatch"; return tr
    if tr.pt_shape != tr.tf_shape:
        tr.status = "shape_mismatch"; return tr

    ok, max_abs, max_rel = np_allclose(y_pt_np, y_tf_np, rtol=rtol, atol=atol)
    tr.max_abs_err, tr.max_rel_err = max_abs, max_rel
    tr.status = "ok" if ok else "value_mismatch"
    return tr


def run_pool2d(params, rtol, atol) -> TrialResult:
    N, C, H, W, k, s = params["N"], params["C"], params["H"], params["W"], params["k"], params["s"]
    x_pt = torch.randn(N, C, H, W, dtype=torch.float32)
    x_tf = tf.convert_to_tensor(np.transpose(x_pt.numpy(), (0, 2, 3, 1)), dtype=tf.float32)

    tr = TrialResult(family="pool2d", params=params, status="ok", rtol=rtol, atol=atol)
    try:
        y_pt = F.max_pool2d(x_pt, kernel_size=k, stride=s, padding=0)
        y_pt_np = to_numpy(y_pt)
        tr.pt_shape = tuple(y_pt_np.shape)
    except Exception as e:
        tr.pt_exception = str(e)

    try:
        y_tf = tf.nn.max_pool2d(x_tf, ksize=k, strides=s, padding="VALID")
        y_tf_np = to_numpy(y_tf)
        y_tf_np = np.transpose(y_tf_np, (0, 3, 1, 2))
        tr.tf_shape = tuple(y_tf_np.shape)
    except Exception as e:
        tr.tf_exception = str(e)

    if tr.tf_exception and tr.pt_exception:
        tr.status = "both_exception"; return tr
    if tr.tf_exception or tr.pt_exception:
        tr.status = "exception_mismatch"; return tr
    if tr.pt_shape != tr.tf_shape:
        tr.status = "shape_mismatch"; return tr

    ok, max_abs, max_rel = np_allclose(y_pt_np, y_tf_np, rtol=rtol, atol=atol)
    tr.max_abs_err, tr.max_rel_err = max_abs, max_rel
    tr.status = "ok" if ok else "value_mismatch"
    return tr


def run_batchnorm(params, rtol, atol) -> TrialResult:
    # 两边都走 inference 路径，并使用相同的 mean/var, gamma=1, beta=0
    N, C, H, W, eps = params["N"], params["C"], params["H"], params["W"], params["eps"]
    x_pt = torch.randn(N, C, H, W, dtype=torch.float32)
    x_tf = tf.convert_to_tensor(np.transpose(x_pt.numpy(), (0, 2, 3, 1)), dtype=tf.float32)

    # 统计沿着 N,H,W 维度的通道均值方差（与 batchnorm 计算一致）
    x_np = x_pt.numpy()
    mean = x_np.mean(axis=(0, 2, 3), keepdims=False)      # (C,)
    var  = x_np.var(axis=(0, 2, 3), keepdims=False)       # (C,)

    gamma = np.ones((C,), dtype=np.float32)
    beta  = np.zeros((C,), dtype=np.float32)

    tr = TrialResult(family="batchnorm", params=params, status="ok", rtol=rtol, atol=atol)

    # Torch: F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    try:
        y_pt = F.batch_norm(
            x_pt, torch.from_numpy(mean), torch.from_numpy(var),
            weight=torch.from_numpy(gamma), bias=torch.from_numpy(beta),
            training=False, momentum=0.1, eps=eps
        )
        y_pt_np = to_numpy(y_pt)
        tr.pt_shape = tuple(y_pt_np.shape)
    except Exception as e:
        tr.pt_exception = str(e)

    # TF: tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon)
    try:
        y_tf = tf.nn.batch_normalization(
            x_tf, mean=mean, variance=var,
            offset=beta, scale=gamma, variance_epsilon=eps
        )
        y_tf_np = to_numpy(y_tf)
        y_tf_np = np.transpose(y_tf_np, (0, 3, 1, 2))
        tr.tf_shape = tuple(y_tf_np.shape)
    except Exception as e:
        tr.tf_exception = str(e)

    if tr.tf_exception and tr.pt_exception:
        tr.status = "both_exception"; return tr
    if tr.tf_exception or tr.pt_exception:
        tr.status = "exception_mismatch"; return tr
    if tr.pt_shape != tr.tf_shape:
        tr.status = "shape_mismatch"; return tr

    ok, max_abs, max_rel = np_allclose(y_pt_np, y_tf_np, rtol=rtol, atol=atol)
    tr.max_abs_err, tr.max_rel_err = max_abs, max_rel
    tr.status = "ok" if ok else "value_mismatch"
    return tr


FAMILY_GEN = {
    "conv2d": gen_conv2d_case,
    "dense": gen_dense_case,
    "relu": gen_relu_case,
    "pool2d": gen_pool2d_case,
    "batchnorm": gen_batchnorm_case,
}
FAMILY_RUN = {
    "conv2d": run_conv2d,
    "dense": run_dense,
    "relu": run_relu,
    "pool2d": run_pool2d,
    "batchnorm": run_batchnorm,
}

FAMILY_GEN.update({
    "linear": gen_linear_case,
    "attention": gen_attention_case,
})

FAMILY_RUN.update({
    "linear": run_linear,
    "attention": run_attention,
})

# ---------------------------- 可选：从对齐结果自动选择家族 ----------------------------
def infer_families_from_pairs(pairs_path: Path, threshold=0.6) -> List[str]:
    fams = set()
    if not pairs_path.exists():
        return list(SUPPORTED_FAMILIES)
    from collections import Counter
    cnt = Counter()
    for line in open(pairs_path, "r"):
        p = json.loads(line)
        if p.get("final_score", 0.0) >= threshold:
            # 简易：看 apis_mapped 里面包含的关键词
            apis = (p.get("apis_mapped") or []) + (p.get("tf_apis") or []) + (p.get("pt_apis") or [])
            s = " ".join(map(str, apis)).lower()
            if "conv" in s: cnt["conv2d"] += 1
            if "dense" in s or "linear" in s: cnt["dense"] += 1
            if "relu" in s: cnt["relu"] += 1
            if "pool" in s: cnt["pool2d"] += 1
            if "norm" in s or "batchnorm" in s: cnt["batchnorm"] += 1
    if cnt:
        fams = {k for k, _ in cnt.most_common()}
    return [f for f in SUPPORTED_FAMILIES if f in fams] or list(SUPPORTED_FAMILIES)


# ---------------------------- 主流程 ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=str, default="auto",
                        help="逗号分隔家族: conv2d,dense,relu,pool2d,batchnorm 或 'auto'")
    parser.add_argument("--trials", type=int, default=50, help="每个家族随机样本数")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--pairs", type=str, default=str(DATA/"recall_pairs_fused.jsonl"),
                        help="当 families=auto 时，从该文件推断要测的家族")
    parser.add_argument("--report", type=str, default=str(DATA/"fuzz_report.jsonl"))
    parser.add_argument("--summary", type=str, default=str(DATA/"fuzz_summary.csv"))
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.families.strip().lower() == "auto":
        fams = infer_families_from_pairs(Path(args.pairs))
        print(f"[INFO] auto 选择家族: {fams}")
    else:
        fams = [f.strip().lower() for f in args.families.split(",") if f.strip()]
        for f in fams:
            if f not in SUPPORTED_FAMILIES:
                print(f"[WARN] 不支持的家族：{f}，已忽略。")
        fams = [f for f in fams if f in SUPPORTED_FAMILIES]
        if not fams:
            fams = SUPPORTED_FAMILIES

    report_path = Path(args.report)
    summary_path = Path(args.summary)

    # 统计桶
    from collections import Counter
    status_cnt = Counter()
    family_cnt = Counter()
    rows_for_csv = []

    with open(report_path, "w") as fout, tqdm(total=len(fams), desc="Families", unit="fam") as fam_bar:
        for fam in fams:
            gen = FAMILY_GEN[fam]
            run = FAMILY_RUN[fam]
            with tqdm(total=args.trials, desc=f"{fam}", unit="case", leave=False) as case_bar:
                for t in range(args.trials):
                    params = gen()
                    try:
                        res: TrialResult = run(params, args.rtol, args.atol)
                    except Exception as e:
                        res = TrialResult(family=fam, params=params, status="unexpected_error",
                                          rtol=args.rtol, atol=args.atol,
                                          tf_exception="", pt_exception=str(e))
                    status_cnt[res.status] += 1
                    family_cnt[fam] += 1
                    fout.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")

                    rows_for_csv.append([
                        fam, res.status, res.params, res.pt_shape, res.tf_shape,
                        res.max_abs_err, res.max_rel_err
                    ])
                    case_bar.update(1)
            fam_bar.update(1) # 更新进度条
    # 写 CSV 汇总
    import csv
    with open(summary_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["family", "status", "params", "pt_shape", "tf_shape", "max_abs_err", "max_rel_err"])
        w.writerows(rows_for_csv)

    # 终端输出简报
    total = sum(family_cnt.values())
    print("\n==== FUZZ SUMMARY ====")
    print(f"总样本: {total}  | 家族分布: {dict(family_cnt)}")
    for k, v in status_cnt.most_common():
        ratio = v / max(total, 1)
        print(f"  - {k:20s}: {v:5d} ({ratio:.2%})")
    print(f"\n报告: {report_path}\n汇总: {summary_path}\n")


if __name__ == "__main__":
    main()
