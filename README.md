# ai_images

ControlNet 八足马结构对比：Lineart vs Scribble（Stable Diffusion 1.5 + **ControlNet 1.1**）。

目标硬件：**NVIDIA RTX 4090**（CUDA + float16）。也兼容 Apple MPS / CPU，但 4090 上会快很多。

## 环境（uv）

需要 [uv](https://docs.astral.sh/uv/) 与 Python 3.11+。

```bash
# 创建 .venv 并按 uv.lock 安装依赖（torch 固定为 2.5.1+cu121）
uv sync

# 本机若 LD_LIBRARY_PATH 含 EDA/旧 CUDA 路径，建议清空后再跑，避免覆盖 torch 自带库
export LD_LIBRARY_PATH=""

# 确认 CUDA 可用（4090 上应打印 True 与 GPU 名）
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```

`pyproject.toml` 已从 PyTorch 官方 `cu121` 索引拉取 `torch` / `torchvision`，以匹配本机 Driver 535 / CUDA 12.2。不要改用 PyPI 默认的 CUDA 13 轮子（会触发 `ncclCommResume` 等符号错误）。

首次运行会从 Hugging Face 拉取 SD1.5、ControlNet 与 Annotators 权重（约数 GB）。可设置 `HF_TOKEN` 提高限速。若直连 Hub 较慢或不稳，可改用镜像：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
```

基座权重使用 `variant="fp16"`（`*.fp16.safetensors`）。Prompt 已压到 CLIP 77 token 以内。

## 模型与参数

| 角色                            | ID / 值                                                           |
| ------------------------------- | ----------------------------------------------------------------- |
| 基座                            | `stable-diffusion-v1-5/stable-diffusion-v1-5`                     |
| Lineart ControlNet              | `lllyasviel/control_v11p_sd15_lineart`（1.1）                     |
| Scribble ControlNet             | `lllyasviel/control_v11p_sd15_scribble`（1.1）                    |
| Annotators                      | `lllyasviel/Annotators`（LineartDetector / HEDdetector scribble） |
| 尺寸                            | 768×512                                                           |
| seed                            | `20260710`                                                        |
| steps                           | 30（UniPCMultistepScheduler）                                     |
| CFG (`guidance_scale`)          | 7.5                                                               |
| `controlnet_conditioning_scale` | 两边均为 1.1                                                      |
| `control_guidance`              | 0.0 → 1.0（全程控制）                                             |

脚本关闭了 `safety_checker`（研究对比用）。两组测试共用同一 prompt、seed、尺寸与采样参数，仅控制图与对应 ControlNet 不同。

## 运行

参考图路径：`outputs/eight_legged_horse_reference.png`（已纳入仓库）。

```bash
export LD_LIBRARY_PATH=""
# 可选：国内镜像
# export HF_ENDPOINT="https://hf-mirror.com"
# export HF_HUB_DISABLE_XET=1

uv run python controlnet_eight_legged_horse_comparison.py
```

结果写入 `outputs/`：

| 文件                                           | 说明     |
| ---------------------------------------------- | -------- |
| `control_lineart.png` / `control_scribble.png` | 控制图   |
| `result_lineart.png` / `result_scribble.png`   | 生成结果 |
| `controlnet_comparison.png`                    | 三图对比 |

## 判断标准

1. 是否有 8 个清晰独立的腿根  
2. 是否有 8 条从腿根到马蹄的连续肢体  
3. 是否有 8 个可辨认马蹄  
4. 腹部中间是否误增腿  
5. 前后侧腿是否因遮挡合并  
6. 彩色骨架线是否被误读为装饰 / 缰绳 / 运动轨迹  

即使 ControlNet 跟随二维线条，SD1.5 的四足先验仍可能合并部分腿；本测试主要验证结构控制是否降低漏腿概率。
