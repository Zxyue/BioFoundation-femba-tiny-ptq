# FEMBA-tiny PTQ实现及说明

使用现有的 FEMBA 统一 PTQ 框架，目标环境：CUDA/12.4.0、torch2.5.1、mamba-ssm2.2.6.post3

代码文件放在 code目录下，以下 Bash 命令在 BioFoundation 根目录执行

## 1. 浮点微调精度

已有微调 checkpoint 可直接做 PTQ。FP32 和 BF16 的最佳文件可能同名，使用完整路径区分。需要比较不同微调精度时，使用同一份预训练权重、数据划分和 seed 分别训练：

```bash
python myptq/code/ptq_finetune.py \
  --pretrained-weights "$PWD/checkpoints/femba/TUAR/FEMBA_tiny_prefixed.safetensors" \
  --data-dir /path/to/hdf5 --output-dir results/finetune/fp16 \
  --precision fp16 --epochs 30 --batch-size 128 --seed 42
```

`--precision` 支持 `fp32`、`fp16`、`bf16`，分别传给原 `run_train.py` 的 Lightning `32-true`、`16-mixed`、`bf16-mixed`。FP16/BF16 使用 AMP，主权重仍为 FP32；FP16 使用梯度缩放，BF16 需对应 GPU 支持。

可选 `--learning-rate`（默认 5e-4）、`--num-workers`（默认 4）；加 `--dry-run` 只打印命令。训练沿用原实验配置，关闭自动续训并执行最终测试；在日志的 `Best checkpoint path` 找到验证集选出的 checkpoint。

**微调精度与 PTQ 位宽独立。** PTQ 不更新梯度，对上述任一 checkpoint，均用 FP32 执行 fake quant。旧 checkpoint 未记录训练精度时，元数据标为 `not_recorded`。

## 2. 单次 PTQ

```bash
CKPT=/path/to/epoch=2-step=1299.ckpt
DATA=/path/to/hdf5

python myptq/code/ptq.py --checkpoint "$CKPT" --data-dir "$DATA" \
  --quant-format W4A8 --weight-scope all-weights \
  --group-size 64 --quantization partial-rotation \
  --rotation-transform hadamard --rotation-block-size 64 \
  --weight-init mse --calibration-method sequential --activation-percentile 95 \
  --batch-size 32 --output-dir results/myptq/w4a8_g64_r64
```

| 参数                                     | 可选值／含义                                                                      |
| ---------------------------------------- | --------------------------------------------------------------------------------- |
| `--quant-format`                       | W8A32 / W4A32 / W8A8 / W4A8 / W4A4；默认 W4A32                                    |
| `--weight-bits`、`--activation-bits` | 也可分别指定 W=4/8、A=4/8/32；weight-bits 0 保持浮点                              |
| `--group-size`                         | 32 / 64 / 128 / none；默认 64，沿每个输出通道的输入列分组                         |
| `--granularity`                        | 默认 channel；group-size none 时每输出通道一个尺度；tensor 配 none 为整层一个尺度 |
| `--quantization`                       | symmetric 对称量化；partial-rotation 分块旋转；full-rotation 全输入维旋转         |
| `--rotation-transform`                 | qr / hadamard，默认 qr                                                            |
| `--rotation-block-size`                | 32 / 64 / 128，默认 64；仅部分旋转使用，与 group-size 独立                        |
| `--rotation-seed`                      | 默认 2025，按层名派生，跨分组与位宽保持一致                                       |
| `--weight-init`                        | max（默认，absmax）或 mse（36 档截断范围搜索）                                    |
| `--clip-percentile`                    | 默认 100；max 可用百分位截断，mse 固定 100                                        |
| `--weight-scope`                       | encoder-linear（默认，仅编码器 Linear）/ all-linear / all-weights                 |
| `--classifier-bits`                    | all-* 范围内分类头的独立位宽；0 保持浮点，未指定则跟随 W                          |
| `--include`、`--exclude`             | 范围内按层名 glob 筛选；例如 `--exclude '*.conv1d'`                             |

**W/A 仅描述选中权重和指定激活位置的模拟位宽。** 默认范围不包含分类头；上例显式使用 all-weights，标准 Tiny 覆盖 28 个权重张量。bias、LayerNorm、位置嵌入、A_log/D 等保持浮点。卷积始终对称量化，旋转只作用于 Linear。

旋转实现为缓存有效权重 `Q(W R) Rᵀ`，激活仍在原坐标量化；不是完整的 QuaRot 权重／激活联合旋转。Hadamard 补齐维度后，先完整逆旋转再裁剪。输出 checkpoint 为浮点 fake-quant 推理文件，不是 INT4 压缩文件或整数 CUDA 内核。

浮点基准使用 `--weight-bits 0 --activation-bits 32`；加 `--print-config` 只查看配置。旧参数 `--bits`、`--method block/full`、`--rotation-size`、`--scope` 仍可用。

## 3. 激活校准

A32 不做激活量化；A8/A4 从 **train.h5** 抽取无放回窗口，随后冻结范围。

| `--calibration-method` | 默认窗口／batch／seed／校准权重 | 实现                                                             |
| ------------------------ | ------------------------------- | ---------------------------------------------------------------- |
| observer                 | 512 / 32 / 42 / float           | 42 个位置，Brevitas 仅观测后冻结                                 |
| sequential               | 1024 / 128 / 0 / quantized      | 32 个位置、24 个阶段，依次开启前序量化；SiLU/GELU 使用非对称范围 |

可通过 `--calibration-windows`、`--calibration-batch-size`、`--calibration-seed`、`--calibration-weight-state float/quantized` 覆盖默认值。`--activation-percentile` 默认 99.999；sequential 支持 95/99/99.9/99.99/99.999/100。两种方法统计位置、共享规则和范围聚合均不同。

校准文件默认 `$DATA/train.h5`，可用 `--calibration-file` 指定。统计按批聚合，因此校准 batch 会影响结果；实际抽样数和索引均保存。数据窗口、通道、归一化和划分应与父模型一致。A32 可仅传 `--data-file /path/to/val.h5`。

observer 额外安装依赖；A32 和 sequential 不需要该步骤：

```bash
python -m pip install --no-deps -r myptq/code/ptq_requirements.txt
```

## 4. 扫描与测试

```bash
# 5 种格式 × 4 种分组 × 3 种方法，共 60 个配置；默认只写计划，加 --run 执行
python myptq/code/ptq_sweep.py --checkpoint "$CKPT" --data-dir "$DATA" \
  --quant-formats W8A32 W4A32 W8A8 W4A8 W4A4 \
  --group-sizes 32 64 128 none --weight-scope all-weights \
  --quantizations symmetric partial-rotation full-rotation \
  --rotation-transforms hadamard --rotation-block-sizes 64 \
  --weight-inits mse --calibration-methods sequential --activation-percentiles 99.999 \
  --calibration-windows 1024 --calibration-batch-size 32 --calibration-seed 42 \
  --batch-size 32 --output-dir results/myptq/sweep --run

# 验证集选定后，用保存的有效权重与冻结尺度评估 test，不再校准
python myptq/code/ptq.py --checkpoint /path/to/selected_case/myptq.ckpt --load-calibrated \
  --data-dir "$DATA" --split test --batch-size 32 --output-dir results/myptq/test
```

扫描可使用 `--weight-inits max mse`、`--calibration-methods observer sequential` 等组合；无效维度自动去重。

结果含 metrics JSON、predictions NPZ、metadata JSON、完成记录及量化后的 myptq.ckpt；扫描另存 CSV 和完整验证集 AUROC 优先的最佳配置。二分类 AUROC/AP 用 `logit1-logit0`，accuracy/BAcc/F1 用零 margin（概率 0.5）阈值。`--max-batches` 为子集诊断，不参与正式最佳配置选择。微调精度、源权重 dtype、PTQ 配置和运行环境分别记录。

## 5. 检查与文件

```bash
# 真实 CUDA 前向检查；可加 --checkpoint "$CKPT"，结果不代表数据集精度
python myptq/code/ptq_cuda_check.py --quant-format W4A8 --weight-scope all-weights \
  --quantization partial-rotation --rotation-transform hadamard --rotation-block-size 64

python -m unittest discover -s myptq/code -p 'ptq*test.py' -v
```

结果按 `result/TUAR/origin-dataset`、`result/TUAR/tuar` 和 `result/TUSL` 整理。TUAR 两种划分均包含 HF 与随机初始化基线，受试者划分还完成两种父模型的 PTQ 测试。TUSL 使用三分类四折，包含 HF 微调、共同配置测试和随机初始化 FP32 对照。共完成 480 个验证扫描配置。详细比较、INT4 配方、远程文件路径和哈希见 [result.md](result.md)。

按同一父模型计算量化损失。可先用 `--weight-bits 0 --activation-bits 32` 分别跑 val 和 test 的浮点基准。扫参只在 val 进行，选定配置后再评估 test。受试者划分 TUAR 的 A8/A4 出现近单类预测，TUSL 的 A8 保留部分排序能力但宏 F1 明显下降，分析时同时查看平衡准确率和混淆矩阵。TUAR 当前清单按验证选择 FP32 的 W4A32，BF16 候选测试表现更好，二者在结果文档中分别标记。

入口：`ptq_finetune.py` 微调、`ptq.py` PTQ、`ptq_sweep.py` 扫描。算法在 `ptq_quant.py`、`ptq_calibration.py` 及两个 activation 文件；`ptq_cuda_check.py` 用于远程检查。来源和许可证见 `ptq_sources.json`、`ptq_LICENSE`、`ptq_mamba_NOTICE.txt`。
