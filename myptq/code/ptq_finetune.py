"""按指定浮点精度微调 FEMBA；复用原 run_train.py，不在训练中量化。"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PRECISIONS = {"fp32": "32-true", "fp16": "16-mixed", "bf16": "bf16-mixed"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--pretrained-weights", type=Path, required=True,
                   help="已加 model. 前缀的 FEMBA safetensors")
    p.add_argument("--data-dir", type=Path, default=os.getenv("DATA_PATH"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--precision", choices=PRECISIONS, default="fp32",
                   help="fp16 / bf16 使用 AMP 混合精度，主权重保持 FP32")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--dry-run", action="store_true", help="仅打印训练命令")
    return p


def training_command(args):
    # 双引号供 Hydra 解析带空格、等号的路径；subprocess 使用 argv，不经过 shell。
    weights = json.dumps(args.pretrained_weights.expanduser().resolve().as_posix(),
                         ensure_ascii=False)
    return [sys.executable, "-u", str(args.repo.expanduser().resolve() / "run_train.py"),
            "+experiment=FEMBA_tiny_finetune",
            f"pretrained_safetensors_path={weights}",
            f"++trainer.precision={PRECISIONS[args.precision]}",
            "trainer.accelerator=gpu", "trainer.devices=1",
            f"trainer.max_epochs={args.epochs}", f"batch_size={args.batch_size}",
            f"num_workers={args.num_workers}", f"seed={args.seed}",
            f"optimizer.lr={args.learning_rate}",
            f"tag=FEMBA_tiny_finetune_{args.precision}",
            "resume=False", "final_test=True"]


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.data_dir is None:
        p.error("请提供 --data-dir 或设置 DATA_PATH。")
    if min(args.epochs, args.batch_size) <= 0 or min(args.num_workers, args.seed) < 0:
        p.error("epochs / batch-size 应为正数，num-workers / seed 应非负。")
    args.repo = args.repo.expanduser().resolve()
    command = training_command(args)
    print(shlex.join(command))
    if args.dry_run:
        return command

    from ptq_runtime import require_cuda
    require_cuda("cuda")
    # 原训练脚本会请求 high matmul precision；关闭 TF32，使浮点对照更清楚。
    env = dict(os.environ, DATA_PATH=str(args.data_dir.expanduser().resolve()),
               CHECKPOINT_DIR=str(args.output_dir.expanduser().resolve()),
               NVIDIA_TF32_OVERRIDE="0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = {"precision": args.precision, "trainer_precision": PRECISIONS[args.precision],
              "command": command, "data_dir": env["DATA_PATH"]}
    (args.output_dir / "myptq_finetune.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run(command, cwd=args.repo, env=env, check=True)
    return record


if __name__ == "__main__":
    main()
