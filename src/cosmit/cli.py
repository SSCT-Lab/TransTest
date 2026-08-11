from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from cosmit.config import load_config
from cosmit.pipeline.stages import PIPELINE_STAGES, validate_pipeline_contract
from cosmit.runner import run_pipeline


def _show_pipeline(output_format: str) -> int:
    validate_pipeline_contract()
    if output_format == "json":
        print(json.dumps([stage.as_dict() for stage in PIPELINE_STAGES], ensure_ascii=False, indent=2))
        return 0

    for stage in PIPELINE_STAGES:
        print(f"{stage.order}. {stage.name.value}")
        print(f"   目标：{stage.goal}")
        print(f"   输出：{', '.join(stage.outputs)}")
        print(f"   验收：{stage.acceptance}")
    return 0


def _validate_config(path: Path) -> int:
    config = load_config(path)
    print("配置有效")
    print(f"迁移方向：{', '.join(direction.slug for direction in config.directions)}")
    print(f"相似组件 Top-K：{config.top_k_similar_components}")
    print(f"最大修复轮次：{config.max_repair_rounds}")
    print(f"动态验证：{'开启' if config.dynamic_validation else '关闭'}")
    return 0


def _run(config_path: Path, run_id: str | None, artifacts_dir: Path | None) -> int:
    config = load_config(config_path)
    active_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = run_pipeline(config, active_run_id, artifacts_dir=artifacts_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cosmit",
        description="CoSMiT 深度学习框架测试知识迁移工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show-pipeline", help="显示六阶段方法及 artifact 契约")
    show.add_argument("--format", choices=("text", "json"), default="text")

    validate = subparsers.add_parser("validate-config", help="校验 CoSMiT 配置")
    validate.add_argument("path", type=Path)

    run = subparsers.add_parser("run", help="执行 TensorFlow ↔ PyTorch 六阶段流水线")
    run.add_argument("config", type=Path)
    run.add_argument("--run-id", help="指定可重复的运行 ID；默认使用 UTC 时间")
    run.add_argument("--artifacts-dir", type=Path, help="覆盖配置中的 artifact 根目录")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "show-pipeline":
        return _show_pipeline(args.format)
    if args.command == "validate-config":
        return _validate_config(args.path)
    if args.command == "run":
        return _run(args.config, args.run_id, args.artifacts_dir)
    raise AssertionError(f"unhandled command: {args.command}")
