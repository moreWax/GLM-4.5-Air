#!/usr/bin/env python
"""
run_compress.py

Compresses a model with:
  • SmoothQuant 0.8
  • GPTQ INT8 (W8A8)
  • 2:4 structured sparsity
Router, embeddings, LayerNorms, gate/up/down_proj and lm_head stay in FP16.

Example
-------
python compress.py \
  zai-org/GLM-4.5-Air \
  --output_dir ./glm45-air-w8a8-s24-routerfp16 \
  --tensor_parallel_size 2
"""
import argparse
from pathlib import Path
from llmcompressor import oneshot
from llmcompressor.modifiers.smoothquant  import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.sparsity     import SparseGPTModifier
from transformers import AutoTokenizer

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
IGNORE_FP16 = [
    "lm_head",
    r"re:.*router.*",
    r"re:.*embedding.*",
    r"re:.*layernorm.*",
    r"re:.*gate_proj.*", r"re:.*up_proj.*", r"re:.*down_proj.*",
]

def build_recipe() -> list:
    """Return the list of LLM Compressor modifiers used in the pipeline."""
    return [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(scheme="W8A8", targets="Linear", ignore=IGNORE_FP16),
        SparseGPTModifier(pattern="2:4", sparsity=0.5, ignore=IGNORE_FP16),
    ]

def compress_model(model_id: str,
                   output_dir: str,
                   tensor_parallel_size: int,
                   num_calibration_samples: int = 512,
                   max_seq_length: int = 2048):
    """
    End-to-end compression helper that mirrors `run_inference` in the template.
    """
    recipe = build_recipe()
    oneshot(
        model=model_id,
        dataset="wikitext",
        num_calibration_samples=num_calibration_samples,
        max_seq_length=max_seq_length,
        recipe=recipe,
        output_dir=output_dir,
        tensor_parallel_size=tensor_parallel_size,
        verbose=True,
    )
    AutoTokenizer.from_pretrained(model_id).save_pretrained(output_dir)

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Compress a model with SmoothQuant ➜ GPTQ ➜ 2:4 sparsity."
    )
    parser.add_argument("model_id", type=str,
                        help="HF model hub identifier or local path.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Target directory for the compressed checkpoint.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel world-size (passed through to vLLM).")
    parser.add_argument("--num_calibration_samples", type=int, default=512,
                        help="Number of calibration samples (default: 512).")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for calibration/inference.")

    args = parser.parse_args()

    # Make sure the output path exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    compress_model(
        model_id=args.model_id,
        output_dir=args.output_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
    )

    print("✅  Compressed checkpoint saved to:", args.output_dir)


if __name__ == "__main__":
    main()
