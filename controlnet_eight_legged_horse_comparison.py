#!/usr/bin/env python3
"""八足马结构测试：ControlNet Lineart vs Scribble。

优先 CUDA（RTX 4090 等），其次 Apple MPS，最后 CPU。
"""

from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
from controlnet_aux import HEDdetector, LineartDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

INPUT_PATH = OUT / "eight_legged_horse_reference.png"
TARGET_W, TARGET_H = 768, 512
SEED = 20260710
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

PROMPT = (
    "full body three-quarter side view of a magnificent mythological horse with exactly eight distinct legs, "
    "four forelegs attached at the chest and four hind legs attached at the hips, "
    "all eight legs anatomically separate from root to hoof, all eight hooves clearly visible, "
    "no legs attached under the middle belly, elegant equine anatomy, dynamic galloping pose, "
    "realistic fantasy concept art, clean readable silhouette, detailed muscles, dramatic natural lighting"
)

NEGATIVE_PROMPT = (
    "four legs, five legs, six legs, seven legs, nine legs, ten legs, missing leg, fused legs, merged legs, "
    "duplicated limb fragments, extra legs under the belly, hidden hooves, cropped legs, motion blur, "
    "multiple horses, malformed anatomy, tangled limbs, text, labels, colored construction lines"
)


def pick_device() -> tuple[str, torch.dtype]:
    """Prefer NVIDIA CUDA (e.g. RTX 4090), then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def empty_cache(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def fit_canvas(
    img: Image.Image,
    size: tuple[int, int] = (TARGET_W, TARGET_H),
    background: str = "white",
) -> Image.Image:
    img = img.convert("RGB")
    contained = ImageOps.contain(img, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    x = (size[0] - contained.width) // 2
    y = (size[1] - contained.height) // 2
    canvas.paste(contained, (x, y))
    return canvas


def run_controlnet(
    controlnet_id: str,
    control_image: Image.Image,
    output_path: Path,
    device: str,
    dtype: torch.dtype,
    conditioning_scale: float = 1.15,
) -> Image.Image:
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # RTX 4090 (24GB) 跑 SD1.5 + ControlNet 显存充裕，不做 slicing 以提速。
    # MPS / 低显存设备再开启切片降低峰值占用。
    if device == "cuda":
        pipe.to(device)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
    else:
        pipe.enable_attention_slicing()
        if device == "mps":
            pipe.enable_vae_slicing()
        pipe.to(device)
        # MPS generator 不稳定，固定用 CPU seed。
        generator = torch.Generator(device="cpu").manual_seed(SEED)

    print(f"  infer {controlnet_id} …")
    with torch.inference_mode():
        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=control_image,
            width=TARGET_W,
            height=TARGET_H,
            num_inference_steps=30,
            guidance_scale=8.0,
            controlnet_conditioning_scale=conditioning_scale,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            generator=generator,
        ).images[0]

    result.save(output_path)
    print(f"  saved {output_path}")
    del pipe, controlnet
    empty_cache(device)
    return result


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"缺少参考图: {INPUT_PATH}\n"
            "请将 eight_legged_horse_reference.png 放到 outputs/"
        )

    device, dtype = pick_device()
    if device == "cuda":
        print(f"device=cuda ({torch.cuda.get_device_name(0)}) dtype={dtype}")
    else:
        print(f"device={device} dtype={dtype}")
    if device == "cpu":
        print("警告: 当前在 CPU 上运行，会非常慢。请在带 CUDA 的 GPU（如 RTX 4090）上运行。")

    reference = fit_canvas(Image.open(INPUT_PATH), background="white")
    reference.save(OUT / "reference_fitted.png")

    print("生成 Lineart / Scribble 控制图…")
    lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
    lineart_control = lineart_detector(
        reference, detect_resolution=768, image_resolution=768
    )
    lineart_control = fit_canvas(lineart_control, background="white")
    lineart_control.save(OUT / "control_lineart.png")

    hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
    scribble_control = hed_detector(
        reference, scribble=True, detect_resolution=768, image_resolution=768
    )
    scribble_control = fit_canvas(scribble_control, background="black")
    scribble_control.save(OUT / "control_scribble.png")

    fig = plt.figure(figsize=(18, 6))
    for i, (title, img) in enumerate(
        [
            ("Reference", reference),
            ("Lineart control", lineart_control),
            ("Scribble control", scribble_control),
        ],
        start=1,
    ):
        ax = fig.add_subplot(1, 3, i)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "controls_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("运行 ControlNet Lineart…")
    lineart_result = run_controlnet(
        "ControlNet-1-1-preview/control_v11p_sd15_lineart",
        lineart_control,
        OUT / "result_lineart.png",
        device=device,
        dtype=dtype,
        conditioning_scale=1.20,
    )

    print("运行 ControlNet Scribble…")
    scribble_result = run_controlnet(
        "lllyasviel/sd-controlnet-scribble",
        scribble_control,
        OUT / "result_scribble.png",
        device=device,
        dtype=dtype,
        conditioning_scale=1.10,
    )

    fig = plt.figure(figsize=(18, 6))
    for i, (title, img) in enumerate(
        [
            ("Reference", reference),
            ("ControlNet Lineart", lineart_result),
            ("ControlNet Scribble", scribble_result),
        ],
        start=1,
    ):
        ax = fig.add_subplot(1, 3, i)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "controlnet_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("输出文件：")
    for name in [
        "control_lineart.png",
        "control_scribble.png",
        "result_lineart.png",
        "result_scribble.png",
        "controlnet_comparison.png",
    ]:
        print(OUT / name)


if __name__ == "__main__":
    main()
