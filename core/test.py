import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F

# ---- Parameters ----
N, C_in, C_out = 2, 7, 8
H, W = 9, 18
k = 5
stride = 1
padding = "VALID"

# Generate identical FP32 inputs
x = np.random.randn(N, C_in, H, W).astype(np.float32)
w = np.random.randn(C_out, C_in, k, k).astype(np.float32)

# ---- TensorFlow (NHWC / HWIO) ----
x_tf = tf.convert_to_tensor(np.transpose(x, (0, 2, 3, 1)))
w_tf = tf.convert_to_tensor(np.transpose(w, (2, 3, 1, 0)))

y_tf = tf.nn.conv2d(
    x_tf,
    w_tf,
    strides=[1, stride, stride, 1],
    padding=padding
)
y_tf_np = tf.transpose(y_tf, (0, 3, 1, 2)).numpy()

# ---- Reference (PyTorch, NCHW / OIHW) ----
x_pt = torch.from_numpy(x)
w_pt = torch.from_numpy(w)

y_ref = F.conv2d(x_pt, w_pt, stride=stride, padding=0)
y_ref_np = y_ref.detach().cpu().numpy()

# ---- Comparison ----
abs_diff = np.abs(y_tf_np - y_ref_np)
rel_diff = abs_diff / (np.abs(y_ref_np) + 1e-12)

print("Output shapes:", y_tf_np.shape, y_ref_np.shape)
print("Max abs error:", abs_diff.max())
print("Max rel error:", rel_diff.max())