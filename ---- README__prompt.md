


1）参考
---- 知乎：：LFM3：最强 2B 的多模态端侧全家桶：：https://zhuanlan.zhihu.com/p/2098861357395337430
Liquid AI《Introducing LFM3》
HF Blog
LFM3-1.7B 模型卡
，及全网搜索FLM3相关的资料，，你的任务是 构造适配的小批量数据(每个20条)放到data文件夹下，并复现整个训练流水线流程代码，训练代码放到项目根目录下
2）不要下载大的权重文件到本地
3）尽你所能，采用简单的方法尽可能的复现FLM3训练流水线
##===================================
已完成 LFM3 轻量训练流水线复现，全程未下载大权重。
----主要成果：
在 [data](D:/Github_romote/FLM3_train_pipline/data) 下生成 6 组数据：预训练、Mid-training、SFT、Mixed RL、MOPD、Agentic RL，每组严格 20 条。
[train_pipeline.py](D:/Github_romote/FLM3_train_pipline/train_pipeline.py) 可离线依次执行六个阶段。
[lfm3_tiny.py](D:/Github_romote/FLM3_train_pipline/lfm3_tiny.py) 实现 tiny tri-hybrid 架构，包含短卷积、SWA、全局注意力、GQA head gate、Partial RoPE 和 sandwich norm。
[train_lora.py](D:/Github_romote/FLM3_train_pipline/train_lora.py) 提供可选 1.7B LoRA 路径；缺少本地权重时默认拒绝运行，防止误下载约 3.46GB 的 Base 权重。
完整使用说明和官方训练阶段映射见 [README.md](D:/Github_romote/FLM3_train_pipline/README.md)。
----验证结果：
6 个 JSONL 文件均为 20 条，ID 唯一。
5 项单元测试全部通过。
smoke 模式六阶段全部完成真实反向传播。
默认 6 层、29.5 万参数版本完成每阶段 10 step；预训练 loss 从 5.556 降至 5.097，Mid-training 从 5.184 降至 4.513。
当前仓库中大权重文件数量为 0，也没有生成 outputs 目录。
----直接运行：
python prepare_data.py --check
python train_pipeline.py --smoke --device cpu --no-save
python -m unittest -v test_pipeline.py
----注意：：
实现依据是官方公开的 24 层 tri-hybrid 架构和
SFT → Mixed RL → MOPD → Agentic RL 路径；
官方 Base 模型卡还披露了 22T token 的分段训练规模。Liquid AI 官方博客、LFM3-1.7B-Base 模型卡、官方 Cookbook。
需要注意：Mixed RL、MOPD 和 Agentic RL 使用的是可在本地运行的离线教学代理，并不是 Liquid AI 未公开的教师模型、真实 agent 环境和大规模 on-policy 训练。

