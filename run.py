import os, sys, subprocess, tempfile, requests, re


SAVE_DIR = "outputs"
tf_path = os.path.join(SAVE_DIR, "test_tf_original.py")
pt_path = os.path.join(SAVE_DIR, "migrated_test_torch.py")

def run_py(path):
    res = subprocess.run(['python', path], capture_output=True, text=True)
    return res.stdout + res.stderr

tf_output = run_py(tf_path)
pt_output = run_py(pt_path)

# 保存运行结果
with open(os.path.join(SAVE_DIR, "run_tf.log"), "w") as f: f.write(tf_output)
with open(os.path.join(SAVE_DIR, "run_torch.log"), "w") as f: f.write(pt_output)

# 打印摘要
print("=== PyTorch 测试已保存至:", pt_path)
print("=== 执行结果如下：")
print(">>> TF 输出:\n", tf_output)
print(">>> Torch 输出:\n", pt_output)