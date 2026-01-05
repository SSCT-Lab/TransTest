import json

pairs = [json.loads(l) for l in open("data/component_pairs.jsonl")]
tf_test = [json.loads(l) for l in open("data/tf_test_api_usage.jsonl")]

tf_apis_in_pairs = set(p["tf_api"] for p in pairs)
tf_apis_in_tests = set(a for item in tf_test for a in item["apis"])

# 看真正有交集的 API 有多少
intersection = tf_apis_in_pairs & tf_apis_in_tests
print("交集数量：", len(intersection))
print(list(intersection)[:20])
