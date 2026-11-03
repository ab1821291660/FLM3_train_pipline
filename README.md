# LFM3 轻量训练流水线复现




本项目把用户原文中的 **FLM3** 统一按 Liquid AI 的 **LFM3** 处理。仓库提供两条路径：

- **默认离线路径**：不下载任何模型权重，从随机初始化的 3.8 万（smoke）或 29.5 万参数 tiny LFM3 开始，依次完成六个阶段的真实前向、反向和参数更新。
- **可选 1.7B LoRA 路径**：对齐 Liquid AI 官方 TRL Notebook 的目标模块和主要超参数；没有本地权重时默认拒绝运行，防止误下载大文件。

这不是 Liquid AI 内部训练的精确复刻。官方只公开了阶段、架构、部分 token 规模和微调示例，没有公开 22T token 数据、六个 RL 教师、奖励环境、采样策略、全部超参数和训练基础设施。本仓库的目标是用最少资源复现可观察的**流程与数据契约**。

## 一分钟跑通

环境要求：Python 3.10+、PyTorch 2.3+。

```powershell
# 1. 检查六组数据，每组必须恰好 20 条
python prepare_data.py --check

# 2. 六阶段离线烟测：每阶段 1 个优化器 step，不保存 checkpoint
python train_pipeline.py --smoke --device cpu --no-save

# 3. 单元测试
python -m unittest -v test_pipeline.py
```

完整 tiny 流程会使用每阶段 10 个 step，并把很小的教学 checkpoint 写入 `outputs/tiny-pipeline/`：

```powershell
python train_pipeline.py --device auto
```

只运行部分阶段：

```powershell
python train_pipeline.py --stages pretrain midtrain sft --no-save
```

## 六阶段映射

| 官方公开阶段 | 本仓库数据 | 本仓库可执行目标 | 复现边界 |
|---|---|---|---|
| Main pre-training | `data/pretrain.jsonl` | 全 token 因果语言模型 CE | 20 条合成文本，不是约 18T+ token 原语料 |
| Mid-training / 长上下文扩展 | `data/midtrain.jsonl` | 工具轨迹 ChatML 因果 CE | 最长 224 byte-token，不是 32K→128K→256K 扩展 |
| SFT | `data/sft.jsonl` | 只对 assistant 答案计算 CE | 20 条，不是官方 1.75T token |
| Mixed RL | `data/mixed_rl.jsonl` | 固定候选上的组相对优势 + KL 代理 | 离线教学代理，不是多域 on-policy RL |
| MOPD | `data/mopd.jsonl` | 六类 specialist response 的序列蒸馏 | 无官方教师 checkpoint，属于 response imitation |
| Agentic RL | `data/agentic_rl.jsonl` | 工具候选的组相对奖励 + KL 代理 | 不执行浏览器/办公环境，无真实副作用 |

官方模型卡还提供 CPT、DPO、GRPO 微调 Notebook。它们是对外可执行的专项适配方法，不等同于官方内部完整后训练算法；本项目的可选 1.7B 路径分别用 SFTTrainer、DPOTrainer、GRPOTrainer 作为简单近似。

## 数据文件

`prepare_data.py` 会确定性生成以下文件，全部位于 `data/`，每个文件 20 条 JSONL：

| 文件 | 主要字段 | 用途 |
|---|---|---|
| `pretrain.jsonl` | `text`, `domain` | 科学、数学、代码、知识、工具原则等原始文本 |
| `midtrain.jsonl` | `messages`, `tools` | agent/tool-call 轨迹和高质量规则 |
| `sft.jsonl` | `messages`, `category` | QA、JSON、SQL、工具、安全、拒答等监督样本 |
| `mixed_rl.jsonl` | `prompt`, `candidates[].reward` | 数学、代码、SQL、格式和安全偏好 |
| `mopd.jsonl` | `teacher_domain`, `teacher_response` | code/math/tool/long-context/agentic/enterprise 教师响应 |
| `agentic_rl.jsonl` | `tools`, `expected_tool`, `candidates` | 可验证工具选择与参数奖励 |

重新生成并立即检查：

```powershell
python prepare_data.py
```

数据仅用于流程验证。20 条样本不足以训练有实际能力的模型，也不应用于衡量泛化效果。

## Tiny LFM3 架构

`lfm3_tiny.py` 直接用 PyTorch 实现了公开结构中的核心设计：

- gated short convolution，卷积核宽度 7；
- sliding-window attention；
- global attention；
- grouped-query attention 和逐 head sigmoid gate；
- global attention 的 Partial RoPE；
- attention/conv 与 SwiGLU FFN 两侧的 sandwich RMSNorm；
- tied token embedding / LM head。

默认 6 层教学配置是 `C-C-F-C-S-C`。`--smoke` 缩为 `C-S-F`，但仍覆盖三类 sequence operator。离线 tokenizer 是 UTF-8 byte tokenizer，因此不依赖 Hugging Face，也不会触发网络请求。仓库中的 `LiquidAI/LFM3-1.7B/` 只含官方配置、Tokenizer 和模型代码，**不含权重**。

## 可选的 1.7B LoRA 路径

先只检查训练计划；该命令不会导入 Transformers、不会联网：

```powershell
python train_lora.py --dry-run
```

若以后已在专用训练机上准备好本地 `LFM3-1.7B-Base` checkpoint，可运行：

```powershell
python train_lora.py --local-model-dir D:\models\LFM3-1.7B --stages sft
```

仅有配置和 tokenizer 的目录会被拒绝。远程模型 ID 也会被拒绝，除非操作者额外传入显式下载开关；本项目没有执行该开关。LoRA adapter 默认保存到 `outputs/lora-pipeline/`，不会合并并复制完整 1.7B 权重。

官方 Notebook 使用的依赖版本记录在 `requirements-lora.txt`；默认 tiny 路径只需 `requirements.txt`。安装依赖不等于下载模型权重。

## 目录说明

```text
.
├── data/                    # 6 × 20 条小数据
├── LiquidAI/LFM3-1.7B/     # 无权重的官方配置/Tokenizer/模型代码
├── finetuning/              # 收集的官方微调 Notebook
├── lfm3_tiny.py             # tiny tri-hybrid 模型
├── prepare_data.py          # 数据生成与校验
├── pipeline_config.json     # tiny 模型和六阶段配置
├── train_pipeline.py        # 默认离线六阶段入口
├── train_lora.py            # 下载保护的可选 1.7B LoRA 入口
└── test_pipeline.py         # 数据、架构和下载保护测试
```

## 公开资料与实现依据

- [Liquid AI: Introducing LFM3](https://www.liquid.ai/blog/lfm3)
- [Hugging Face / LiquidAI: Build specialized multimodal models with LFM3](https://huggingface.co/blog/LiquidAI/lfm3)
- [LiquidAI/LFM3-1.7B 模型卡](https://huggingface.co/LiquidAI/LFM3-1.7B)
- [Liquid4All/cookbook 微调示例](https://github.com/Liquid4All/cookbook/tree/main/finetuning/notebooks)
- [知乎参考文章](https://zhuanlan.zhihu.com/p/2098861357395337430)

截至 2026-11-03，公开资料给出的 1.7B 文本 backbone 为 24 层：13 层 gated short convolution、5 层 256-window SWA、6 层 global attention。官方预训练图进一步分为约 15T web/synthetic、3T code/math/science（4K context）、3T curated agentic（32K）、500B（128K）和 50B（256K），合计口径为 22T token；后训练顺序为 SFT、Mixed RL、MOPD、Agentic RL。`pretrain.jsonl` 的 `phase` 与 `midtrain.jsonl` 的 `context_phase` 只记录对应关系，不宣称小样本真的达到该上下文长度。

## 已验证与未验证

本仓库当前已验证：

- 六个 JSONL 文件均为合法 UTF-8，且各 20 条、ID 唯一；
- tiny tri-hybrid 模型前向、loss、反向梯度均为有限值；
- 六个阶段在 CPU 上按顺序完成 optimizer step；
- 1.7B 入口在仅有配置而没有权重时会拒绝运行；
- 验证过程中没有下载或生成大权重文件。

未验证：真实 LFM3-1.7B LoRA 训练、官方规模训练、256K 上下文、多机训练、真实 agent 环境、官方 benchmark 或端侧部署性能。
