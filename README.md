# NExT-GPT 实验代码包

本目录包含 NExT-GPT 多模态大语言模型部署与推理的完整实验代码。

## 目录结构

```
nextgpt-exp5/
├── README.md                          # 本文件
├── 实验五_多模态大语言模型_NExT-GPT_实验报告.md  # 实验报告
├── install.sh                         # 环境安装脚本
├── download_checkpoints.sh            # 模型权重下载脚本
├── fix_code.sh                        # 代码修复补丁脚本
├── run_demo.sh                        # Gradio Demo 启动脚本
├── assets/
│   └── images/
│       └── cat_on_grass.jpg           # 生成图像样例
└── code/
    ├── demo_app.py                    # Gradio Web 界面
    ├── inference.py                   # 推理脚本（带历史记录）
    ├── test_inference.py              # 轻量级推理测试
    ├── run_inference.py               # 图像生成推理脚本
    ├── header.py                      # 数据集头文件
    ├── process_embeddings.py           # Embedding 预计算脚本
    ├── config/
    │   ├── __init__.py
    │   ├── base.yaml                  # 基础配置
    │   ├── stage_1.yaml               # Stage 1 配置（编码侧对齐）
    │   ├── stage_2.yaml               # Stage 2 配置（解码侧对齐）
    │   └── stage_3.yaml               # Stage 3 配置（指令微调）
    └── model/
        ├── __init__.py
        ├── anyToImageVideoAudio.py    # 核心模型文件
        └── custom_sd.py               # Stable Diffusion 自定义模块
```

## 快速开始

### 1. 环境安装

```bash
bash install.sh
```

### 2. 下载模型权重

```bash
bash download_checkpoints.sh
```

### 3. 代码修复（如有需要）

```bash
bash fix_code.sh
```

### 4. 启动 Gradio Demo

```bash
bash run_demo.sh
```

### 5. 图像生成推理测试

```bash
conda activate nextgpt
cd code
python run_inference.py
```

## 依赖说明

- Python 3.8
- PyTorch 1.13.1 + CUDA 11.6
- transformers 4.29.2
- diffusers 0.17.0
- gradio 3.44.0
- accelerate 0.19.0
- peft 0.3.0
- deepspeed 0.9.3
- sentencepiece

## 模型权重说明

| 模型 | 大小 | 路径 | 下载方式 |
|------|------|------|----------|
| ImageBind | 4.5 GB | `ckpt/pretrained_ckpt/imagebind_ckpt/huge/` | wget |
| Vicuna-7B-v0 | 13.5 GB | `ckpt/pretrained_ckpt/vicuna_ckpt/7b_v0/` | HuggingFace (BetterHF) |
| NExT-GPT delta | 792 MB | `ckpt/delta_ckpt/nextgpt/7b_tiva_v0/` | HuggingFace |
| Stable Diffusion | ~5 GB | HuggingFace 缓存 | 自动下载 |

## 生成图像样例

Prompt: `draw a cat on the grass`

![cat_on_grass](assets/images/cat_on_grass.jpg)

## 已知问题与解决方案

### 问题 1：demo_app.py 缺少 `--mode` 参数

**现象**：`KeyError: 'mode'`

**解决**：运行 `fix_code.sh`，或手动在 `demo_app.py` 的 argparse 中添加：

```python
parser.add_argument('--mode', type=str, default='train')
```

### 问题 2：data 模块导入失败

**现象**：`ModuleNotFoundError: No module named 'data'`

**解决**：运行 `fix_code.sh`，或在 `code/` 目录下执行：

```bash
ln -s dataset data
```

### 问题 3：显存不足（OOM）

**现象**：`CUDA out of memory`

**原因**：Vicuna-7B（~18GB）+ Stable Diffusion（~5GB）超过 24GB 显存

**解决**：修改 `model/anyToImageVideoAudio.py`，将 SD 加载方式改为 CPU offload：

```python
# 原始代码（约第760行）：
generation_model = StableDiffusionPipeline.from_pretrained(...).to("cuda")

# 修改为：
generation_model = StableDiffusionPipeline.from_pretrained(..., torch_dtype=torch.float16)
generation_model.enable_sequential_cpu_offload(gpu_id=0)
generation_model.enable_vae_slicing()
```

### 问题 4：Vicuna 权重无法下载

**现象**：`401 Unauthorized`

**原因**：官方 `lmsys/vicuna-7b-v0` 需要授权

**解决**：使用社区版 `BetterHF/vicuna-7b`：

```bash
huggingface-cli download BetterHF/vicuna-7b \
    --local-dir ckpt/pretrained_ckpt/vicuna_ckpt/7b_v0 \
    --local-dir-use-symlinks False
```

## 实验结果

| 测试任务 | 状态 | 说明 |
|---------|------|------|
| 环境配置 | ✅ 完成 | conda + PyTorch + 所有依赖 |
| Gradio Demo | ✅ 成功 | 端口 24004 监听 |
| 文本推理 | ✅ 成功 | 正确生成文本响应 |
| 图像生成 | ✅ 成功 | 512x512 JPEG 图像 |

详见 `实验五_多模态大语言模型_NExT-GPT_实验报告.md`

## 参考资料

- NExT-GPT GitHub: https://github.com/NExT-GPT/NExT-GPT
- NExT-GPT 论文: Wu et al., "NExT-GPT: Any-to-Any Multimodal LLM", ICML 2024
- ImageBind: https://github.com/facebookresearch/ImageBind
- Vicuna: https://github.com/lm-sys/FastChat
- HuggingFace 镜像: https://hf-mirror.com
