# 📦 `run_compress.py`

End-to-end compression for **GLM-4.5-Air** that

1. **SmoothQuant 0.8**  
2. **GPTQ INT8 (W8A8)** post-training quantisation  
3. **2 : 4 structured sparsity** (50 %)

while **router, embeddings, LayerNorms, gate/up/down projections and `lm_head` remain FP16**.  
The resulting checkpoint loads straight into **vLLM** or **TensorRT-LLM**.

---

## ✨ Pipeline at a glance

| # | Modifier | Purpose |
|---|----------|---------|
| 1 | `SmoothQuantModifier` | Balance activation/weight ranges → INT8 accuracy |
| 2 | `GPTQModifier(scheme="W8A8")` | 8-bit weights **+** 8-bit activations |
| 3 | `SparseGPTModifier(pattern="2:4", sparsity=0.5)` | Enforce NVIDIA 2 : 4 sparsity for cuSPARSELt/TRT-LL speed-ups |

*Calibration* → **WikiText-103** (512 samples)  
*Output*     → `<output_dir>` (weights + tokenizer)

---

## 🚀 Quick start

### Minimum viable GPU rig — **4 × RTX 3090 (24 GB each)**

```bash
# 0.  Core deps (CUDA 12.8 wheels shown — tweak for your system)
pip install torch==2.3.0+cu128 transformers>=4.44 vllm>=0.4.2 \
            llmcompressor>=0.3.0 accelerate>=0.28

# 1.  Compress
export CUDA_VISIBLE_DEVICES=0,1,2,3            # optional pinning
python run_compress.py \
  zai-org/GLM-4.5-Air \
  --output_dir ./glm45-air-w8a8-s24-routerfp16 \
  --tensor_parallel_size 4 \
  --num_calibration_samples 512

# 2.  Serve with vLLM
python -m vllm.entrypoints.openai.api_server \
  --model ./glm45-air-w8a8-s24-routerfp16 \
  --tensor-parallel-size 4 \
  --port 8000
````

<details>
<summary>📊 VRAM budget per 24 GB card (TP = 4)</summary>

| Component            |  Size / GPU | Notes                       |
| -------------------- | ----------: | --------------------------- |
| Weights (INT8 + 2:4) | **13.3 GB** | 106 B × 1 byte × 0.5 ÷ 4    |
| GPTQ meta            |    \~0.3 GB | scales / zeros              |
| KV-cache\*           |      7-8 GB | 16 k context, batch 1       |
| **Free**             |  **≈ 2 GB** | activations + fragmentation |

\* hidden = 8 192, layers = 32.  Larger batch or context ⇒ add GPUs or move KV off-GPU.

</details>

---

## ⚙️ Valid tensor-parallel sizes (up to 8 GPUs)

All shardable dims (`hidden_size = 4096`, `heads = 96`, `experts = 128`) are divisible by **1, 2, 4, 8**.
On 24 GB cards, memory constraints rule out TP = 1 & 2:

| TP    | GPUs  | Weights / GPU | Fits 24 GB? | Comment                            |
| ----- | ----- | ------------- | ----------- | ---------------------------------- |
| 1     | 1     | 53 GB         | ❌           | Needs ≥ 48 GB (A100-80G, H100-80G) |
| 2     | 2     | 26.5 GB       | ❌           | Drop to W4A8 **or** add GPUs       |
| **4** | **4** | **13.3 GB**   | **✅**       | Recommended minimum                |
| **8** | **8** | **6.6 GB**    | **✅**       | Ample head-room                    |

> **Rule of thumb** Required VRAM ≈ **weights ÷ TP + KV-cache**.

---

## 📏 Context-length head-room (KV-cache footprint)

For FP16 keys + values the per-token cost:

```
bytes_per_token = hidden_size × 2 bytes × 2 (K+V) × num_layers
                = 4096 × 2 × 2 × 32  ≈ 524 288 B  (≈ 0.5 MB)
```

Dividing by TP:

| TP    | Bytes / token / GPU |   ≈ MB @ 8 k |  ≈ MB @ 16 k |  ≈ MB @ 32 k |
| ----- | ------------------: | -----------: | -----------: | -----------: |
| **4** |           131 072 B | **1 024 MB** | **2 048 MB** | **4 096 MB** |
| **8** |            65 536 B |   **512 MB** | **1 024 MB** | **2 048 MB** |

### Context limits on 24 GB cards

| GPUs (TP)    | Weights / GPU | Max ctx (batch = 1)\*  | Free VRAM |
| ------------ | ------------: | ---------------------- | --------- |
| **4 × 3090** |   **13.3 GB** | **≤ 32 k** (≈ 4 GB KV) | \~6 GB    |
| **8 × 3090** |    **6.6 GB** | **≤ 64 k** (≈ 4 GB KV) | \~13 GB   |

\* For batch > 1 use `total_KV = batch × context × bytes_per_token ÷ TP`.

---

## 🛠️ CLI reference

| Flag                        | Default | Description                               |
| --------------------------- | ------- | ----------------------------------------- |
| `model_id`                  | —       | HF repo or local FP16 path                |
| `--output_dir`              | —       | Where to write the compressed checkpoint  |
| `--tensor_parallel_size`    | 1       | TP world-size for compression & inference |
| `--num_calibration_samples` | 512     | WikiText samples for PTQ                  |
| `--max_seq_length`          | 2048    | Max tokens during calibration             |

---

## 📂 Output layout

```
glm45-air-w8a8-s24-routerfp16/
├── config.json
├── pytorch_model.bin      # INT8 + sparse
├── tokenizer.json
└── special_tokens_map.json
```

---

## 🖥️ Hardware cheat-sheet

| GPUs         | VRAM / card | TP | Fits W8A8 + 2:4? | Context head-room (bs = 1) |
| ------------ | ----------- | -- | ---------------- | -------------------------- |
| **4 × 3090** | 24 GB       | 4  | ✅                | ≤ 32 k                     |
| 4 × 4090     | 24 GB       | 4  | ✅                | ≤ 32 k                     |
| **8 × 3090** | 24 GB       | 8  | ✅                | ≤ 64 k                     |
| 4 × A800-80G | 80 GB       | 4  | ✅                | 100 k +                    |
| 1 × H100-80G | 80 GB       | 1  | ✅                | ≤ 16 k (single-GPU TP = 1) |

---

## 🔧 Tweak the recipe

* Adjust `SmoothQuantModifier(smoothing_strength=…)` (0 → off, 1 → aggressive).
* Switch `GPTQModifier` to `scheme="W4A8"` to halve weight size (fits 2 × 3090).
* Increase `sparsity` if your kernels exploit > 50 % 2 : 4.

---

## ❓ Troubleshooting

| Issue                | Likely cause           | Fix                                         |
| -------------------- | ---------------------- | ------------------------------------------- |
| CUDA OOM at load     | Not enough GPUs        | Use ≥ 4 cards or drop to W4A8               |
| Accuracy drop > 1 pp | Over-sparse or high SQ | Lower `sparsity` or `smoothing_strength`    |
| No TRT-LLM speed-up  | 2 : 4 kernels missing  | Re-build TRT-LLM with `SPARSITY_ENABLED=ON` |

---

## 📜 License

Apache-2.0 (same as [`llmcompressor`](https://github.com/Mikubill/LLM-COMPRESSOR) & [`vllm`](https://github.com/vllm-project/vllm))
© 2025 Thomas Whitworth & Contributors

```
::contentReference[oaicite:0]{index=0}
```
