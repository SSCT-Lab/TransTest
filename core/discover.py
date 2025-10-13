import os
import fnmatch
import hashlib
from pathlib import Path

def sha1_of_file(path: Path, block_size=65536) -> str:
    """计算文件的 sha1 哈希，便于唯一标识和缓存"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def discover_test_files(root_dirs, include_globs, exclude_globs):
    """
    遍历测试目录，返回符合条件的文件列表

    Args:
        root_dirs (list[str]): 根目录列表
        include_globs (list[str]): 匹配测试文件的 glob 模式
        exclude_globs (list[str]): 需要排除的 glob 模式

    Returns:
        list[dict]: 每个文件的元信息 (rel_path, abs_path, sha1, file_size)
    """
    results = []
    for root in root_dirs:
        root = Path(root)
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                full_path = Path(dirpath) / filename
                rel_path = str(full_path.relative_to(root))

                # include 过滤
                if not any(fnmatch.fnmatch(rel_path, pat) for pat in include_globs):
                    continue
                # exclude 过滤
                if any(fnmatch.fnmatch(rel_path, pat) for pat in exclude_globs):
                    continue

                results.append({
                    "rel_path": rel_path,
                    "abs_path": str(full_path),
                    "file_size": full_path.stat().st_size,
                    "sha1": sha1_of_file(full_path),
                })
    return results
