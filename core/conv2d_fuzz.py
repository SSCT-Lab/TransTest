# ==================【 macOS 全稳定模式 Patch 】==================
import os

# 关闭所有加速线程（PyTorch / TensorFlow / MKL / OpenMP / Eigen）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 禁用全部设备加速（macOS 的 MPS 也要关！）
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 静默 TF

import tensorflow as tf
import torch

# PyTorch 全局单线程
torch.set_num_threads(1)

# TensorFlow 关闭 GPU/MPS
try:
    tf.config.set_visible_devices([], "GPU")
except:
    pass

try:
    tf.config.set_visible_devices([], "MPS")
except:
    pass

# TensorFlow 单线程运行
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

print("[INIT] macOS 全稳定模式已启用：Torch/TF 均为单线程、无 GPU/MPS，加速库关闭。")

# ===============================================================

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # 强制 CPU

# ---------------- utils ----------------

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return np.array(x)

def np_allclose(a, b, rtol=1e-4, atol=1e-5):
    if a.shape != b.shape:
        return False, float("inf"), float("inf")
    diff = np.abs(a - b)
    max_abs = float(diff.max())
    denom = np.maximum(np.abs(b), np.finfo(a.dtype).eps)
    max_rel = float((diff / denom).max())
    ok = np.all(diff <= (atol + rtol * np.abs(b)))
    return bool(ok), max_abs, max_rel

# ---------------- generate random conv2d case ----------------

def gen_case():
    N = np.random.randint(1, 4)
    C_in = np.random.randint(1, 9)
    C_out = np.random.randint(1, 9)
    H = np.random.randint(8, 33)
    W = np.random.randint(8, 33)
    k = np.random.randint(1, 6)
    s = np.random.randint(1, 3)
    return dict(N=N, C_in=C_in, C_out=C_out, H=H, W=W, k=k, s=s, padding="VALID")

# ---------------- run single conv2d test ----------------

def run_conv2d(p, rtol=1e-4, atol=1e-5):
    N, Cin, Cout, H, W = p["N"], p["C_in"], p["C_out"], p["H"], p["W"]
    k, s = p["k"], p["s"]

    # Input
    x_pt = torch.randn(N, Cin, H, W, dtype=torch.float32)
    w_pt = torch.randn(Cout, Cin, k, k, dtype=torch.float32)

    # Convert to TF format
    x_tf = tf.convert_to_tensor(np.transpose(x_pt.numpy(), (0, 2, 3, 1)), dtype=tf.float32)
    w_tf = tf.convert_to_tensor(np.transpose(w_pt.numpy(), (2, 3, 1, 0)), dtype=tf.float32)

    # Running
    try:
        y_pt = F.conv2d(x_pt, w_pt, stride=s, padding=0)
        y_pt_np = to_numpy(y_pt)
    except Exception as e:
        return "pt_exception", str(e), None, None, None, None

    try:
        y_tf = tf.nn.conv2d(x_tf, w_tf, strides=[1, s, s, 1], padding="VALID")
        y_tf_np = to_numpy(y_tf)
        y_tf_np = np.transpose(y_tf_np, (0, 3, 1, 2))
    except Exception as e:
        return "tf_exception", str(e), None, None, None, None

    # Shape check
    if y_pt_np.shape != y_tf_np.shape:
        return "shape_mismatch", None, tuple(y_pt_np.shape), tuple(y_tf_np.shape), None, None

    ok, abs_err, rel_err = np_allclose(y_pt_np, y_tf_np, rtol=rtol, atol=atol)
    if ok:
        return "ok", None, tuple(y_pt_np.shape), tuple(y_tf_np.shape), abs_err, rel_err
    else:
        return "value_mismatch", None, tuple(y_pt_np.shape), tuple(y_tf_np.shape), abs_err, rel_err

# ---------------- main fuzz runner ----------------

def main():
    trials = 100000
    out = Path("data/conv2d_mismatch.jsonl")
    out.parent.mkdir(exist_ok=True)

    mismatch = 0

    with open(out, "w") as f:
        for _ in tqdm(range(trials)):
            p = gen_case()
            status, err, pt_shape, tf_shape, abs_err, rel_err = run_conv2d(p)

            if status != "ok":  # Only save mismatches
                mismatch += 1
                f.write(json.dumps({
                    "params": p,
                    "status": status,
                    "error": err,
                    "pt_shape": pt_shape,
                    "tf_shape": tf_shape,
                    "max_abs_err": abs_err,
                    "max_rel_err": rel_err
                }) + "\n")

    print("=====================================")
    print("TOTAL:", trials)
    print("MISMATCH:", mismatch)
    print("RATE:", mismatch / trials)
    print("OUTPUT:", out)

if __name__ == "__main__":
    main()