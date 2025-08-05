# GLM-4.5-104 B — SmoothQuant → GPTQ INT8 → 2 ∶ 4 Sparsity  
_Ampere-friendly recipe, vLLM 0.10.0 ready_

## ⭐ Overview
This guide explains how to turn **THUDM/glm-4.5-104 B** into an
INT8 (+ activation) & 2 ∶ 4 sparse checkpoint that

* quantises on **4 × RTX 3090** (24 GB) via *layer-sequential on-loading*:contentReference[oaicite:0]{index=0},  
* **keeps sensitive layers in FP16** (router, embeddings, layer norms, `gate_/up_/down_proj`, `lm_head`) based on MoE sensitivity studies:contentReference[oaicite:1]{index=1},  
* serves on **2 × RTX 3090** using vLLM 0.10.0’s V1 engine & CPU off-load flags:contentReference[oaicite:2]{index=2}.

---

## 🖥️ Requirements

| Phase | GPUs | Peak VRAM / GPU | Other |
|-------|------|-----------------|-------|
| **Quantisation** | 4 × 3090 | 7 – 9 GB | 128 GB system RAM, 200 GB NVMe at `/tmp/llmc_offload` |
| **Inference** (INT8 + 2 : 4) | 2 × 3090 | ~ 24 GB **+ 8 GB CPU offload** | `--cpu-offload-gb 8` and `--swap-space 8` stretch each GPU to a virtual 32 GB |

> ⚠️ 3090s are PCIe—not NVLink—so **NVMe swap** is essential for KV-cache growth.

---

## 🚀 Quick-start

```bash
# 0) Python env
python -m venv glm45-env && source glm45-env/bin/activate
pip install --upgrade pip wheel

# 1) Core packages (CUDA 12.8 / PyTorch 2.7)
pip install torch==2.7.0+cu128 -f https://download.pytorch.org/whl/torch_stable.html
pip install vllm==0.10.0 llmcompressor>=0.6.0 accelerate bitsandbytes

# 2) Four-GPU accelerate config
accelerate config default \
    --num_processes 4 --gpu_ids "0,1,2,3" \
    --mixed_precision fp16 --offload_dir /tmp/llmc_offload

# 3) Quantise (≈ 7–9 h)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/quantize_glm45.py
