import json
from pathlib import Path

# 初始人工映射表（可以不断扩展）
API_EQUIVALENTS = {
    "torch.nn.Conv2d": "API_CONV2D",
    "tf.keras.layers.Conv2D": "API_CONV2D",

    "torch.nn.Linear": "API_DENSE",
    "tf.keras.layers.Dense": "API_DENSE",

    "torch.nn.ReLU": "API_RELU",
    "tf.keras.layers.ReLU": "API_RELU",
    "tf.nn.relu": "API_RELU",

    "torch.nn.BatchNorm2d": "API_BATCHNORM",
    "tf.keras.layers.BatchNormalization": "API_BATCHNORM",

    "torch.optim.Adam": "API_OPT_ADAM",
    "tf.keras.optimizers.Adam": "API_OPT_ADAM",

    "torch.nn.CrossEntropyLoss": "API_LOSS_CE",
    "tf.keras.losses.CategoricalCrossentropy": "API_LOSS_CE",
}

def normalize_api_name(api):
    """统一 API 名字格式，便于匹配"""
    return api.lower().replace("_", "").replace(".", "")

def build_api_map(tf_tests, pt_tests):
    all_apis = set()
    for t in tf_tests + pt_tests:
        all_apis.update(t.get("apis", []))

    mapping = {}
    for api in all_apis:
        # 先查手工映射表
        if api in API_EQUIVALENTS:
            mapping[api] = API_EQUIVALENTS[api]
        else:
            # 自动规则：同名/大小写忽略
            norm = normalize_api_name(api)
            if "conv2d" in norm:
                mapping[api] = "API_CONV2D"
            elif "dense" in norm or "linear" in norm:
                mapping[api] = "API_DENSE"
            elif "relu" in norm:
                mapping[api] = "API_RELU"
            elif "batchnorm" in norm:
                mapping[api] = "API_BATCHNORM"
            elif "adam" in norm:
                mapping[api] = "API_OPT_ADAM"
            elif "crossentropy" in norm:
                mapping[api] = "API_LOSS_CE"
            else:
                mapping[api] = f"API_{api.upper()}"  # fallback
    return mapping

def apply_api_mapping(tests, mapping):
    for t in tests:
        mapped = [mapping.get(api, api) for api in t.get("apis", [])]
        t["apis_mapped"] = mapped
    return tests

if __name__ == "__main__":
    Path("../data/parsing").mkdir(parents=True, exist_ok=True)
    Path("../data/mapping").mkdir(parents=True, exist_ok=True)
    
    tf_tests = [json.loads(line) for line in open("../data/parsing/tests_tf.parsed.jsonl")]
    pt_tests = [json.loads(line) for line in open("../data/parsing/tests_pt.parsed.jsonl")]

    mapping = build_api_map(tf_tests, pt_tests)

    # 保存映射表
    with open("../data/mapping/api_map.json", "w") as f:
        json.dump(mapping, f, indent=2)

    # 应用映射
    tf_mapped = apply_api_mapping(tf_tests, mapping)
    pt_mapped = apply_api_mapping(pt_tests, mapping)

    with open("../data/mapping/tests_tf.mapped.jsonl", "w") as f:
        for t in tf_mapped:
            f.write(json.dumps(t) + "\n")

    with open("../data/mapping/tests_pt.mapped.jsonl", "w") as f:
        for t in pt_mapped:
            f.write(json.dumps(t) + "\n")
