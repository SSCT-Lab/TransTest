import os, sys, subprocess, tempfile, requests, re

# 常量配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_key(path="key.config"):
    with open(path) as f: return f.read().strip()
API_TOKEN = load_key()

PROMPT = """请将以下 TensorFlow 单元测试代码迁移为等价的 PyTorch 单元测试，
保持测试结构，替换 tf.* API 为 torch.*，用 torch.allclose 代替 assertAllClose，
去掉 pytest 装饰，只用 unittest：
---
{code}
"""

def migrate(code: str) -> str:
    payload = {
        "model":"Qwen/QwQ-32B",
        "messages":[{"role":"user","content": PROMPT.format(code=code)}],
        "stream": False, "max_tokens":1024, "temperature":0.2, "top_p":0.9,
        "top_k":50, "n":1, "response_format": {"type":"text"}, "tools":[]
    }
    r = requests.post(API_URL, headers={"Authorization":f"Bearer {API_TOKEN}"}, json=payload)
    full_text = r.json()["choices"][0]["message"]["content"]
    match = re.search(r"```python\n(.*?)```", full_text, re.DOTALL)
    return match.group(1) if match else full_text

def run_py(path):
    res = subprocess.run(['python', path], capture_output=True, text=True)
    return res.stdout + res.stderr

def main():
    tf_code = '''import tensorflow as tf

class TFTScaleTest(tf.test.TestCase):
    def setUp(self):
        super(TFTScaleTest, self).setUp()
        self.input = tf.linspace(start=0, stop=10, num=10)
        self.expected_result = tf.linspace(start=0, stop=1, num=10)

    def test_tft_scale(self):
        tensor = tf.cast(self.input, dtype=tf.float64)
        tensor = tensor / tf.norm(tensor)
        self.assertAllClose(tensor, self.expected_result)


if __name__ == "__main__":
    tf.test.main()

'''
    pt_code = migrate(tf_code)

    # 保存源/目标代码
    tf_path = os.path.join(SAVE_DIR, "test_tf_original.py")
    pt_path = os.path.join(SAVE_DIR, "migrated_test_torch.py")
    with open(tf_path, "w") as f: f.write(tf_code)
    with open(pt_path, "w") as f: f.write(pt_code)

    # 执行测试
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

if __name__ == "__main__":
    main()
