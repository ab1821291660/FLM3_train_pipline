# 多轮 Agent RL：SFT → 蒸馏 → GRPO → OpenEnv → Async GRPO

这个目录实现了 Hugging Face《A curriculum for post-training agents》对应的五阶段教学流程。它不是把单轮问答换个名字：SFT 数据保存完整工具轨迹，蒸馏和 GRPO 在生成期间执行工具循环，环境阶段维护跨轮状态，异步阶段把 rollout 生成与梯度更新解耦。

默认命令只做离线校验，不下载模型。真正训练前必须明确提供本地 checkpoint，或者显式添加 `--allow-model-download`。

## 流程与数据

| 阶段 | Trainer | 模型从什么学习 | 20 条数据 | 多轮工具调用 |
|---|---|---|---|---|
| 1. SFT | `SFTTrainer` | 正确的专家 Agent 轨迹 | `data/sft.jsonl` | 每条 6 次调用和 6 次工具返回 |
| 2. 蒸馏 | `DistillationTrainer` | 教师对学生 on-policy 工具轨迹的 token 分布 | `data/distillation.jsonl` | `tools=` 驱动学生反复调用工具 |
| 3. 工具 RL | `GRPOTrainer` | 最终金额正确性和必需工具集合奖励 | `data/grpo_tools.jsonl` | 无状态工具循环，最多 8 轮 |
| 4. 环境 RL | `GRPOTrainer` | 有状态环境的最终状态和二值奖励 | `data/grpo_openenv.jsonl` | 检查商品、加购物车、设地址、结账 |
| 5. 异步 Agent RL | `AsyncGRPOTrainer` | 后台并发 rollout 的环境奖励 | `data/async_grpo.jsonl` | 与阶段 4 相同，但生成和训练并行 |

五个文件各有且只有 20 条，ID 唯一。订单和价格完全确定，便于验证 reward；样本用于说明格式和打通代码，不代表足以训练出生产模型的数据规模。

模型交接关系如下：

```text
base student
  -> outputs/sft
  -> outputs/distillation
  -> outputs/grpo_tools
  -> outputs/grpo_openenv
  -> outputs/async_grpo
```

## 目录说明

```text
aa_trl_pipline/
├── data/                       # 5 × 20 条 JSONL
├── prepare_data.py             # 可重复生成并严格校验数据
├── toolkit.py                  # 无网络、确定性的四个订单工具
├── rewards.py                  # GRPO 可 pickling 的奖励函数
├── order_env.py                # 进程内、有状态 environment_factory
├── openenv_order/              # OpenEnv typed models/client/server
├── openenv_adapter.py          # OpenEnv client -> TRL tools 适配器
├── train_pipeline.py           # 五阶段训练和无下载 dry-run
├── train_harbor_async.py       # 可选：真实 Agent harness 黑盒训练
└── test_pipeline.py            # 数据、轨迹、reward、下载保护测试
```

## 先做离线验证

在仓库根目录执行：

```powershell
python -m aa_trl_pipline.prepare_data --check
python -m aa_trl_pipline.train_pipeline --dry-run
python -m unittest -v aa_trl_pipline.test_pipeline
```

重新生成数据：

```powershell
python -m aa_trl_pipline.prepare_data
```

这些命令不导入 TRL、不加载 checkpoint，也不访问网络。

## 安装与同步训练

先根据目标 CUDA 安装 PyTorch，再安装本流程依赖：

```powershell
python -m pip install -r aa_trl_pipline/requirements.txt
```

建议先使用本地、支持 tool calling chat template 的 Qwen checkpoint。以下示例连续执行前四个同步阶段；`--smoke` 会把每阶段缩为一个 optimizer step：

```powershell
python -m aa_trl_pipline.train_pipeline `
  --stages sft distillation grpo_tools grpo_openenv `
  --student-model D:/models/Qwen3-0.6B `
  --teacher-model D:/models/Qwen3-1.7B `
  --smoke
```

若明确允许从 Hub 下载，才添加：

```powershell
python -m aa_trl_pipline.train_pipeline `
  --stages sft distillation grpo_tools grpo_openenv `
  --allow-model-download --smoke
```

各阶段也可独立运行。此时用 `--student-model` 指向上一个阶段输出，例如：

```powershell
python -m aa_trl_pipline.train_pipeline `
  --stages grpo_tools `
  --student-model aa_trl_pipline/outputs/distillation `
  --smoke
```

## 真正的 OpenEnv 路径

不传 `--openenv-url` 时，环境阶段使用进程内 `OrderFulfillmentEnv`，适合最小验证。要走 OpenEnv 的 client/server 边界，安装可选依赖并启动 CPU 环境服务器：

```powershell
python -m pip install -r aa_trl_pipline/requirements-openenv.txt
python -m uvicorn aa_trl_pipline.openenv_order.server.app:app --host 127.0.0.1 --port 8001
```

另开终端训练：

```powershell
python -m aa_trl_pipline.train_pipeline `
  --stages grpo_openenv `
  --student-model aa_trl_pipline/outputs/grpo_tools `
  --openenv-url http://127.0.0.1:8001 `
  --smoke
```

OpenEnv 服务维护购物车和 episode 状态，TRL adapter 只把 `inspect_product`、`add_to_cart`、`set_destination`、`checkout` 暴露给模型；最终 reward 由服务端状态决定，而不是检查模型是否背出固定调用顺序。

## Async GRPO

`AsyncGRPOTrainer` 需要 Linux、vLLM、`transformers>=5.2.0`，并要求生成服务器和 trainer 使用不同 GPU。先完成阶段 4；然后让 vLLM 服务同一个 checkpoint。示意命令：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve aa_trl_pipline/outputs/grpo_openenv \
  --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --logprobs-mode processed_logprobs \
  --weight-transfer-config '{"backend":"nccl"}'
```

另一张 GPU 启动训练：

```bash
CUDA_VISIBLE_DEVICES=1 accelerate launch -m aa_trl_pipline.train_pipeline \
  --stages async_grpo \
  --student-model aa_trl_pipline/outputs/grpo_openenv \
  --vllm-server-url http://127.0.0.1:8000 \
  --smoke
```

工具、reward 函数、`environment_factory` 都定义在模块顶层或使用可 pickling 的 callable class，满足异步 rollout 子进程的约束。

## 真实 harness + Harbor（可选高级路径）

阶段 5 的本地样本用于 white-box environment loop。若要训练 OpenCode、Codex、mini-swe-agent 一类自己拥有工具循环的真实 Agent，应由 OpenEnv Harbor 管理任务、沙箱、verifier 和 token capture，再交给 `HarnessRolloutWorker`。本目录提供独立入口：

```powershell
python -m aa_trl_pipline.train_harbor_async --dry-run
```

真实运行还需要：OpenEnv Harbor server、独立 vLLM server、Linux 双 GPU、沙箱凭据以及 Harbor 格式任务集。它不会悄悄复用本地 20 条订单样本来冒充 coding sandbox 的端到端验收。

## Reward 设计

- 工具 GRPO：`TOTAL=xx.xx` 必须与 oracle 精确一致，并检查 rollout 是否包含必需工具名。
- 环境 GRPO：只有购物车商品与数量、城市、结账金额、观察步骤全部正确才返回 `1.0`，否则为 `0.0`。
- 文本格式 reward 只作为辅助信号；环境最终状态是主信号。
- 工具结果 token 由 TRL 从 loss 中屏蔽，训练目标只覆盖模型生成的 token。

## 验证边界

本仓库已能离线验证数据格式、工具调用轮数、oracle 成功路径、错误购物车零奖励、阶段交接计划和模型下载保护。没有在当前 Windows 工作区下载 Qwen 权重，也没有执行 GPU 梯度训练、vLLM、远程 OpenEnv 或 Harbor 沙箱。因此 dry-run/单元测试通过不等于五阶段模型训练已经完成。

TRL 的 Agent API 和 OpenEnv 仍在快速演进，尤其是 `environment_factory`、`AsyncGRPOTrainer` 和 Harbor capture；在真实集群运行时应锁定经过验证的 TRL、Transformers、OpenEnv、vLLM 版本。

## 参考资料

- [A curriculum for post-training agents](https://huggingface.co/blog/huggingface/train-agents)
- [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [TRL Distillation Trainer](https://huggingface.co/docs/trl/distillation_trainer)
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
- [TRL Async GRPO Trainer](https://huggingface.co/docs/trl/async_grpo_trainer)
- [TRL 与 OpenEnv 集成](https://huggingface.co/docs/trl/openenv)
- [OpenEnv 文档](https://huggingface.co/docs/openenv/index)
- [Agent 术语表](https://huggingface.co/blog/agent-glossary)
- [TRL 官方 Harbor AsyncGRPO 示例](https://github.com/huggingface/trl/tree/main/examples/async_grpo_harbor)
