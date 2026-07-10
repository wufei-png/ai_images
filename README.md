# ai_images

ControlNet 八足马结构对比：Lineart vs Scribble（Stable Diffusion 1.5）。

目标硬件：**NVIDIA RTX 4090**（CUDA + float16）。也兼容 Apple MPS / CPU，但 4090 上会快很多。

## 环境（uv）

需要 [uv](https://docs.astral.sh/uv/) 与 Python 3.11+。

```bash
# 创建 .venv 并按 uv.lock 安装依赖
uv sync

# 确认 CUDA 可用（4090 上应打印 True 与 GPU 名）
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```

若 Linux 上 `torch.cuda.is_available()` 为 `False`，安装带 CUDA 的 PyTorch 后再跑，例如：

```bash
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

首次运行会从 Hugging Face 拉取 SD1.5、ControlNet 与 Annotators 权重（约数 GB）。可设置 `HF_TOKEN` 提高限速。

## 运行

参考图路径：`outputs/eight_legged_horse_reference.png`（已纳入仓库）。

```bash
uv run python controlnet_eight_legged_horse_comparison.py
```

结果写入 `outputs/`：

| 文件 | 说明 |
|------|------|
| `control_lineart.png` / `control_scribble.png` | 控制图 |
| `result_lineart.png` / `result_scribble.png` | 生成结果 |
| `controlnet_comparison.png` | 三图对比 |

两组测试共用同一 prompt、seed（`20260710`）、尺寸（768×512）与主要采样参数，便于比较结构保持能力。

## 判断标准

1. 是否有 8 个清晰独立的腿根  
2. 是否有 8 条从腿根到马蹄的连续肢体  
3. 是否有 8 个可辨认马蹄  
4. 腹部中间是否误增腿  
5. 前后侧腿是否因遮挡合并  
6. 彩色骨架线是否被误读为装饰 / 缰绳 / 运动轨迹  

即使 ControlNet 跟随二维线条，SD1.5 的四足先验仍可能合并部分腿；本测试主要验证结构控制是否降低漏腿概率。
