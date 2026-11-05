# TUSL QAP 六种精度下执行预训练、微调、测试

tusl_qap_code目录下代码支持W4A32、W4A8、W8A32、W8A8、W4A4 和 W32A32 浮点对照训练

和同组同学使用同一份数据 `tusl-events-20261026-v2`，TUSL 三分类（0=bckg、1=seiz、2=slow）、FEMBA-Tiny（embed_dim=35、2 blocks）、固定受试者折

本代码在xiaochen训练好的qap模型 [github.com/xiaochenwang860-png/femba-qap.git](https://github.com/xiaochenwang860-png/femba-qap.git) 上运行，详见第二节内容

## 1. 六种配置与统一预算

| 配置   | 权重量化开关 qap_weight_bits | 激活开关 qap_act_bits | 初始校准分位数 |
| ------ | ---------------------------: | --------------------: | -------------: |
| W32A32 |                    0（浮点） |             0（浮点） |         不使用 |
| W4A32  |                            4 |             0（浮点） |         不使用 |
| W4A8   |                            4 |                     8 |           99.9 |
| W8A32  |                            8 |             0（浮点） |         不使用 |
| W8A8   |                            8 |                     8 |           99.9 |
| W4A4   |                            4 |                     4 |             95 |

六种配置均执行：

```text
本折 train.h5 无标签预训练（随机初始化）
→ final.ckpt
→ 本折 train.h5 有标签微调、val.h5 选择最低 val_loss
→ 最佳分类 checkpoint
→ 本折 test.h5 测试，保留窗口级与事件级结果
```

统一为预训练 50 轮、batch 32；微调 10 轮、batch 16、累积 1 步、warmup 2 轮；默认 seed=42。预训练与微调保持相同 W/A 位宽，测试恢复训练学到的尺度和激活范围，不用测试数据重新校准。W32A32 的权重和激活量化参数均为 0，训练与测试使用浮点模型。其他配置在指定位置使用 fake quant，SSM 状态及部分计算仍为浮点。

默认四折 × 六种配置 × 一个种子 = **24 条正式通路**，即 **24 次预训练 + 24 次微调 + 24 次测试**。开始前还会对六种配置各跑一条短 smoke（各 1 轮预训练和微调，共 12 次短训练），smoke 结果单独保存，不进入正式精度汇总。采用单 GPU 串行执行。

## 2. 环境和目录

xiaochen 激活量化代码对应 `mamba-ssm==2.2.6`

使用 PyTorch 2.7.1/CUDA 12.6或PyTorch 2.5.1/CUDA 12.4环境均可

下载 xiaochen 完整仓库代码，检查提交号 `ac9a4c0e53c874d9f587ca762f1a9b1d6857b3fc`

将代码和已划分数据上传到 GPU 环境

示例：

```text
$HOME/femba-qap/
$HOME/tusl_qap_six_configs_20261105/
$HOME/data/tusl-events-20261026-v2/
$HOME/results/tusl_qap_six_v1/       新的总输出目录
```

`DATA` 下须保留 fold0...fold3、各 train/val/test.h5 及 data-manifest.json。直接使用已有划分，不重新处理原始 EDF。

新增 data_module 配置由本包的 Hydra 搜索路径加载，无须覆盖原仓库配置。旧工具包和旧实验目录保持不变；这次采用一个新的总输出目录。

## 3. 一条命令跑完

先激活正确的 CUDA 环境，在已分配 GPU 的节点或作业内设置四个路径：

```Shell
export TOOLS=$HOME/xiao/tusl_qap_six_configs_20261105
export REPO=$HOME/xiao/femba-qap
export DATA=$HOME/processed_eeg/tusl-events-20261026-v2
export OUT=$HOME/xiao/results/tusl_qap_six_v1

python "$TOOLS/run_all_tusl_qap.py" run \
  --repo "$REPO" --data-root "$DATA" --out "$OUT"
```

这一个入口自动完成：

1. 核对全部选中实验的数据 SHA-256、生成预训练和微调计划。
2. 检查/记录源代码、配置、工具代码、Python 和关键包版本。
3. 六种配置各做一次短 smoke；任何配置失败即停止，先检查对应日志。
4. 依次运行 W32A32、W4A32、W4A8、W8A32、W8A8、W4A4，每种配置按选定的 fold 和 seed 独立执行 的预训练、微调和测试。
5. 完成后生成各折明细、六种配置对比和合并事件结果。

训练阶段、输出路径、位宽和量化器状态均自动处理；不需要逐条调用旧工具，也不需要手动传入 checkpoint。不要把其他数据集或另一种精度的 checkpoint 放入某条通路。

如果只想先检查计划，不执行训练：

```bash
python "$TOOLS/run_all_tusl_qap.py" plan \
  --repo "$REPO" --data-root "$DATA" --out "$OUT"
```

正式 `run` 已包含计划检查，不必强制先运行 `plan`。

只执行 W32A32 时使用一个独立输出目录。保持与原五配置相同的仓库版本、环境、四折数据、训练预算和种子。

```bash
python "$TOOLS/run_all_tusl_qap.py" run \
  --repo "$REPO" --data-root "$DATA" \
  --out "$HOME/xiao/results/tusl_w32a32_baseline" \
  --configs W32A32 --folds 0 1 2 3 --seeds 42 43 44 \
  --pretrain-epochs 50 --finetune-epochs 10
```

此命令执行 12 条正式流程。单独运行的 W32A32 可按相同 fold、seed 和事件级指标与原结果比较。汇总表只包含本次选中的配置。

## 4. Slurm 提交方式（可选）

本包也提供 `run_all.sbatch`。在提交前按集群要求补充分区、账号、内存、时限等设置，并激活正确环境；模板不猜测你的集群账号或时限。

```bash
export TOOLS=$HOME/tusl_qap_six_configs_20261105
export REPO=$HOME/femba-qap
export DATA=$HOME/data/tusl-events-20261026-v2
export OUT=$HOME/results/tusl_qap_six_v1
export PYTHON_BIN=$(command -v python)

sbatch "$TOOLS/run_all.sbatch"
```

默认提交一张 GPU，在一个作业内顺序执行。48 次正式训练能否落在集群允许的 walltime 内需要实测；若被时限中断，使用下一节的继续方式。不要在没有分配 GPU 的登录节点运行。

## 5. 中断后如何继续

相同配置重复执行同一条 `run` 命令时，会核对已完成阶段的输出哈希和父 checkpoint，跳过有效完成的阶段，不重复训练。

如果某阶段中断，且你已经确认旧进程/旧作业不再运行，查看日志解决环境或资源问题后执行：

```bash
python "$TOOLS/run_all_tusl_qap.py" run \
  --repo "$REPO" --data-root "$DATA" --out "$OUT" \
  --retry-incomplete
```

该选项会将未完成阶段目录保存到同一 case 的 `interrupted/`，然后重新执行该阶段；不删除旧内容、不影响其他完成阶段。它不是从中断的优化步接着训练，而是从该阶段规定的起点重跑。

有 `complete.json` 的阶段会验证命令、输入 checkpoint 和输出哈希。若已完成文件被改动、父模型改变，或源代码/数据/预算变动，脚本拒绝静默复用。需要改变科学设置时使用新的 `--out`。

同一输出目录一次只运行一个总作业，不要在旧作业仍在执行时启动带 `--retry-incomplete` 的第二个进程。

## 6. 输出在哪里

```text
OUT/
├── batch_plan.json                 所有选中通路与命令
├── campaign.json                   固定方案、源码/工具哈希、环境版本
├── _plan_logs/
├── W32A32/
│   ├── smoke/fold0/seed42/...       短 smoke，单独存放
│   └── fold0/seed42/
│       ├── execution_plan.json
│       ├── pretrain/
│       │   ├── final.ckpt
│       │   ├── pretrain.log
│       │   └── complete.json
│       ├── finetune/
│       │   ├── checkpoints/.../epoch=*.ckpt
│       │   ├── finetune.log
│       │   └── complete.json
│       └── test/
│           ├── metrics.json
│           ├── predictions.npz
│           ├── test.log
│           └── complete.json
├── W4A32/ ... fold0...fold3
├── W4A8/ ... fold0...fold3
├── W8A32/ ... fold0...fold3
├── W8A8/ ... fold0...fold3
├── W4A4/ ... fold0...fold3
├── results_by_fold.csv             六种配置每折的窗口级/事件级指标
├── comparison.csv                  每配置、每种子的折间均值及样本标准差
├── comparison_across_seeds.csv     如有多种子，单独汇总种子间差异
└── summary.json                    含各配置合并的留出事件预测指标
```

优先查看 `comparison.csv` 的事件级 macro AUROC、macro AP、balanced accuracy、macro F1、accuracy。它的标准差是折间变化，不是种子标准差或置信区间；默认只有 seed=42，种子标准差留空。

每条通路都使用最低验证损失的分类 checkpoint；预训练使用完整预算的 `final.ckpt`。测试不选择 checkpoint、不调阈值、不校准激活。TUSL 必须是 3 类、mcc，原 TUAR 二分类评估脚本不适用。

监控当前阶段时，入口会打印对应的日志文件；可在另一终端执行例如：

```bash
tail -f "$OUT/W4A32/fold0/seed42/pretrain/pretrain.log"
```

## 7. 小规模执行或多种子（可选）

先只跑 fold0 的六种配置：

```bash
python "$TOOLS/run_all_tusl_qap.py" run \
  --repo "$REPO" --data-root "$DATA" --out "$OUT" --folds 0
```

之后去掉 `--folds 0` 用同一总输出目录运行，就会保留已完成的 fold0，补齐其他折，再生成四折汇总。

只补某些配置也可以，例如：

```bash
python "$TOOLS/run_all_tusl_qap.py" run \
  --repo "$REPO" --data-root "$DATA" --out "$OUT" --configs W8A32 W8A8 W4A4
```

此时对比表仅汇总该命令选中的配置；要生成完整六配置表，使用默认完整命令重跑，它会核对并跳过已完成工作。

多个种子：`--seeds 42 43 44`。这会扩展到 72 条正式通路、144 次正式训练和 72 次测试，增加计算量。种子间均值/标准差写入单独的表，不与折间波动混算。

如果相同环境下六种配置已经通过 smoke，可加 `--skip-smoke`；默认建议保留，尤其 W4A4 和带激活量化的 Mamba 自定义前向。

## 8. 数据与比较口径

已核对四折 train/val/test 没有受试者和录音重叠，详见随包 `split_audit.json`。14 名受试者每折固定用于训练，24 名受试者轮流测试；四折测试池共 153 个事件，不能称为全部 38 名受试者的完全留出测试。

每个事件有两个 5 秒窗口，事件结果用两个 softmax 概率向量的算术平均。窗口和事件指标都保存，跨方法比较时保持同一口径。六种配置、每折、每个种子都从各自随机初始化开始，不共享跨折预训练权重，以避免测试受试者进入预训练。

本包不自动导入以前单独目录里的训练结果，默认使用新的总输出目录。

已有五配置实验分析文档见[执行结果](result.md)，及结果数据目录result
