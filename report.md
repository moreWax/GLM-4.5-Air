# End-to-End Script: GLM-4.5-Air 2:4 Sparsity + Mixed-Precision Quantization  
*(INT4-W4A16 for non-expert weights, INT8-W8A8 for expert weights, 2:4 50 % sparsity on both, router & lm_head kept in FP16)*

---

### 1. Overview  
The following script applies **2:4 50 % structured sparsity** to **all Linear layers** (model + experts) and then performs **mixed-precision quantization**:

| Layer Group      | Sparsity | Weight-Activation Scheme | Precision Kept |
|------------------|----------|--------------------------|----------------|
| Non-expert Linear| 2:4 50 % | INT4  weights, FP16 activations | – |
| Expert Linear    | 2:4 50 % | INT8 weights & activations | – |
| Router / lm_head | None     | FP16                     | ✅ |

---

### 2. Environment & Installation
```bash
# Create a fresh Python 3.11 env (conda or venv)
pip install --upgrade pip
pip install llm-compressor[all] transformers datasets accelerate safetensors
```

---

### 3. End-to-End Script (`quantize_glm45_air.py`)
```python
#!/usr/bin/env python3
"""
quantize_glm45_air.py

Applies 2:4 50 % sparsity + mixed-precision quantization to GLM-4.5-Air:
- All Linear layers → 2:4 50 % sparsity
- Non-expert Linear → INT4-W4A16
- Expert Linear     → INT8-W8A8
- Router & lm_head  → FP16 (ignored)
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import SparseGPTModifier
from llmcompressor.modifiers.quantization import QuantizationModifier   # handles both AWQ & GPTQ-like configs

MODEL_ID  = "zai-org/GLM-4.5-Air"      # or local path
SAVE_DIR  = "./glm45-air-2of4-int4-int8"
NUM_CALIB = 512
MAX_LEN   = 2048

# ------------------------------------------------------------------
# 1. Load tokenizer & model (FP16 on GPU via accelerate)
# ------------------------------------------------------------------
print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False        # needed for some MoEs during compression
            )

# ------------------------------------------------------------------
# 2. Prepare calibration data
# ------------------------------------------------------------------
print("Preparing calibration data...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_gen[:{NUM_CALIB}]")
ds = ds.shuffle(seed=42)

def tokenize(example):
    return tokenizer(
        tokenizer.apply_chat_template(example["messages"], tokenize=False),
        truncation=True,
        max_length=MAX_LEN,
        add_special_tokens=False
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

# ------------------------------------------------------------------
# 3. Build compression recipe
# ------------------------------------------------------------------
IGNORE_LAYERS = ["lm_head", "router", "gate_proj"]  # keep in FP16

recipe = [
    # Stage 1: 2:4 50 % sparsity on ALL Linear layers
    SparseGPTModifier(
        targets=["Linear"],
        sparsity=0.5,
        mask_structure="2:4",
        post_prune_method="reweight",
        ignore=IGNORE_LAYERS,
    ),

    # Stage 2A: INT4-W4A16 for Non-expert Linear layers
    QuantizationModifier(
        config_groups={
            "non_expert": {
                "targets": ["Linear"],
                "ignore": IGNORE_LAYERS + ["re:.*\.mlp\.experts\."],   # exclude experts
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "group_size": 128,
                },
                "activations": None,  # W4A16 (no activation quantization)
            },
        }
    ),

    # Stage 2B: INT8-W8A8 for Expert Linear layers
    QuantizationModifier(
        config_groups={
            "experts": {
                "targets": ["re:.*\.mlp\.experts\."],
                "ignore": IGNORE_LAYERS,
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "group_size": 128,
                },
                "activations": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "token",
                },
            },
        }
    ),
]

# ------------------------------------------------------------------
# 4. Apply compression
# ------------------------------------------------------------------
print("Starting compression...")
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    num_calibration_samples=NUM_CALIB,
    max_seq_length=MAX_LEN,
    output_dir=SAVE_DIR,
    save_compressed=True,
    trust_remote_code=True,
)

# ------------------------------------------------------------------
# 5. Save & cleanup
# ------------------------------------------------------------------
model.save_pretrained(SAVE_DIR, safe_serialization=True, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Compressed model saved to {os.path.abspath(SAVE_DIR)}")
```

---

### 4. Quick Inference Check with vLLM
```python
from vllm import LLM, SamplingParams

llm = LLM(model=SAVE_DIR, trust_remote_code=True)

prompt = "Explain how 2:4 sparsity accelerates inference."
sampling = SamplingParams(max_tokens=128, temperature=0.7)
out = llm.generate(prompt, sampling_params=sampling)
print(out[0].outputs[0].text)
```

---

### 5. Tips & Troubleshooting
| Issue | Fix |
|-------|-----|
| **CUDA OOM** | Use `device_map="auto"` and ensure ≥ 80 GB GPU RAM (A100/H100). Add `max_memory={0:"70GiB"}` if needed. |
| **Accuracy drop** | Increase `NUM_CALIB` (up to 1 k), reduce `group_size` (64), or lower sparsity to 25 %. |
| **Expert regex mismatch** | Print model keys (`for n, _ in model.named_parameters(): print(n)`) and adjust `"re:.*\.mlp\.experts\."` accordingly. |
| **Router still quantized** | Add exact layer names to `IGNORE_LAYERS` if `router` keyword is insufficient. |

---

### 6. Next Steps
- **Evaluation**: Run `lm_eval` or your own benchmark suite.
- **Recovery fine-tune**: Use Axolotl + LLM-Compressor integration to recover any lost accuracy.
- **Deployment**: Package with Docker + vLLM serving or push to Hugging Face Hub.

This script is ready to execute end-to-end on a single multi-GPU node and produces a compressed GLM-4.5-Air checkpoint optimized for both size and speed on NVIDIA GPUs.
