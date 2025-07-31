Quantizing GLM-4.5-Air (MoE) with 2:4 Sparsity and Mixed Precision
1. Loading the MoE Model on Multiple GPUs
First, load the GLM-4.5-Air model using Hugging Face Transformers with trust_remote_code=True (since it’s a custom MoE architecture) and distribute it across 4 GPUs. The llm-compressor provides a helper to create a device_map that offloads layers to multiple GPUs (and reserves extra memory for quantization calculations)
GitHub
. We’ll use bfloat16 (torch_dtype=torch.bfloat16) as the precision for loading, which the GLM config indicates is the training dtype
huggingface.co
. For example:
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map

MODEL_ID = "zai-org/GLM-4.5-Air"

# Create a device map to spread model across 4 GPUs
device_map = calculate_offload_device_map(
    MODEL_ID, 
    reserve_for_hessians=True,  # reserve memory for quantization Hessians (used by GPTQ/AWQ)
    num_gpus=4, 
    torch_dtype="auto",        # "auto" uses bfloat16 as per model config
    trust_remote_code=True
)

# Load model and tokenizer with the specified device_map and dtype
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=device_map, torch_dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
Why multiple GPUs? GLM-4.5-Air is a 106B-parameter Mixture-of-Experts model (with 128 experts)
huggingface.co
, which in full FP16 would exceed 200 GB of memory. Spreading the model across 4 GPUs (and CPU if needed) is necessary to fit it in memory for compression. The calculate_offload_device_map utility above will partition the model layers across 4 GPUs (and even offload some to CPU if 4× GPU memory isn’t enough)
GitHub
.
2. Preparing Calibration Data for Quantization
Next, prepare a calibration dataset for the quantization process. Both AWQ and GPTQ (the algorithms used for weight quantization) require running a small sample of data through the model to collect activation statistics
docs.vllm.ai
. For MoE models, it’s important to use a reasonably large and diverse sample so that each expert layer gets activated at least a few times
GitHub
. In practice, a few hundred to a couple thousand tokens of representative text (e.g. dataset of conversations or Wikipedia text) works well. Here we use the Ultrachat dataset (an open chat dataset) as an example, limiting to 512 samples for calibration:
from datasets import load_dataset

# Use a publicly available dataset for calibration (e.g., ultrachat dialogues)
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load a subset of the dataset and shuffle
ds = load_dataset(DATASET_ID, split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

# Preprocess: format to plain text input for the model
def preprocess(example):
    # For chat datasets, apply the chat template if available (or just join messages)
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": text}

ds = ds.map(preprocess)

# Tokenize the inputs for the model
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQUENCE_LENGTH, padding=False)

ds = ds.map(tokenize, remove_columns=ds.column_names)
Calibration Data Tips: Using ~500–2000 tokens is generally sufficient. Because GLM-4.5-Air has many experts, using the higher end of that range (e.g. 1000+) can improve accuracy by ensuring all experts see some data
GitHub
. If you don’t have a ready dataset, you can create a small corpus of representative prompts/user queries and use that for calib_data. The calibration process does not require labels – it just feeds data through the model to gather activation statistics.
3. Designing the Compression Recipe
With the model and data ready, we define a compression recipe – a list of modifiers describing which compression techniques to apply. Our goal is to apply 2:4 structured pruning (50% sparsity) to all linear layers, and mixed-precision quantization (INT4 for base model weights, INT8 for expert weights) while keeping certain critical parts in FP16. We also incorporate AWQ to minimize the accuracy loss for 4-bit quantization. The recipe will consist of three stages:
3.1 2:4 Structured Sparsity with SparseGPT: We prune 50% of weights in all linear layers using the SparseGPT algorithm, which prunes in a structure of “2 out of every 4 contiguous weights” set to zero (NVIDIA Ampere’s 2:4 sparsity pattern)
docs.vllm.ai
. This yields a 50% reduction in weight count and can speed up inference on supported GPUs
docs.vllm.ai
. We ignore the router/gating layers and output layer during pruning, because those are small but critical for routing experts and final logits. Keeping them dense avoids destabilizing the MoE routing. In GLM-4.5, the gating layers are referred to as router (the MoE token routing linear) and the experts’ internal gate_proj (the gating projection in GLU activation within each expert). We’ll exclude any module names containing "router" or "gate_proj", as well as the "lm_head" final layer, from pruning. SparseGPT allows a sequential update mode which improves accuracy when pruning large models by updating weights iteratively; we enable that as well
GitHub
.
3.2 Activation-Weighted Quantization (AWQ) Pre-processing: Before actually quantizing weights, we apply the AWQ modifier to the model. AWQ (Activation-Weighted Quantization) analyzes calibration activations to identify the most important weight channels and scales them to reduce quantization error
docs.vllm.ai
docs.vllm.ai
. In essence, AWQ “protects” about 1% of channels from excessive rounding error by rescaling weights based on how sensitive they are
docs.vllm.ai
. We configure AWQ for 4-bit weights (since our hardest quantization is 4-bit for the base model). Notably, AWQ is a weight-only method (it does not quantize activations and expects weights in W*N/A16 format) – in fact, the API enforces that all groups use the same bit-width and no activation quantization during AWQ
docs.vllm.ai
. We will run AWQ with 4-bit configuration on the whole model (it will primarily benefit the INT4-targeted layers). This step adjusts the model’s weights but leaves them in floating-point, ready for final quantization.
3.3 Mixed-Precision Quantization (INT4 & INT8): Finally, we quantize the weights to low-bit integers. We want INT4 for the base transformer layers and INT8 for the expert layers, to balance memory/speed gains with the experts’ potentially higher sensitivity. We also quantize activations to 8-bit for those parts that use INT8 weights (for a full W8A8 effect on experts), while keeping activations at 16-bit for the INT4 parts (W4A16). In effect, all linear layers will be quantized except the ones we explicitly ignore. We will use QuantizationModifier to define two quantization configurations:
Base model layers: 4-bit weight, 16-bit activation (W4A16), symmetric integer quantization. These cover the transformer feed-forward layers and attention projections outside the MoE experts.
Expert layers: 8-bit weight, 8-bit activation (W8A8), symmetric integer quantization. These apply to the weights within the MoE experts’ feed-forward networks (except the parts we keep in FP16).
We will again ignore the router and gate_proj layers here so that those remain in FP16 precision
GitHub
. Keeping the MoE router (which decides expert routing) in full precision avoids degradation in expert selection. Likewise, each expert’s internal gate_proj (the gating half of its GLU activation) remains FP16 to preserve the gating dynamics in the expert’s feed-forward computation. The rest of the expert weights (up_proj and down_proj in the GLM expert MLP) will be quantized to 8-bit. The final lm_head is also left in FP16 to preserve generation quality.
Putting this together, we’ll construct the recipe using SparseGPTModifier, AWQModifier, and QuantizationModifier:
from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import SparseGPTModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
# We'll need classes for specifying quantization details:
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme, QuantizationType, QuantizationStrategy

# 1. 50% 2:4 structured pruning on all Linear layers (ignore router, gate_proj, lm_head)
prune_mod = SparseGPTModifier(
    sparsity=0.50,  # 50% weights pruned
    mask_structure="2:4",
    sequential_update=True,  # sequential layer-wise weight update for better recovery
    targets=[r"Linear"],     # target all Linear layers in the model
    ignore=["lm_head", "router", "gate_proj"]  # ignore MoE gating and output layers
)

# 2. AWQ weight preprocessing for 4-bit quantization
awq_mod = AWQModifier(
    bits=4,              # target 4-bit weight quantization
    symmetric=True,      # use symmetric quantization (no zero-point) for compatibility
    group_size=128,      # grouping granularity for weight quant (128 is common)
    ignore=["lm_head", "router", "gate_proj"]  # also ignore these in any AWQ scaling
)
# Note: AWQModifier here will analyze activations from calibration data and scale important weight channels.
# It does *not* actually quantize yet (weights remain float until next step).

# 3. Mixed Precision Quantization: INT4 for base, INT8 for experts (with 8-bit acts for experts)
quant_mod = QuantizationModifier(
    ignore=["lm_head", "router", "gate_proj"],  # keep these layers at FP16
    config_groups={
        # Group for expert layers – apply 8-bit weight & 8-bit activation quantization
        "expert_int8": QuantizationScheme(
            targets=[r"re:.*experts\..*"],  # regex to match layers in the 'experts' submodules
            weights=QuantizationArgs(
                num_bits=8, 
                type=QuantizationType.INT, 
                symmetric=True, 
                dynamic=False,             # static quant (non-dynamic per batch)
                strategy=QuantizationStrategy.GROUP, 
                group_size=128
            ),
            activations=QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT, 
                symmetric=True,
                dynamic=False,
                strategy=QuantizationStrategy.TENSOR  # per-tensor scaling for activations
            )
        ),
        # Group for non-expert (base model) layers – apply 4-bit weight quantization
        "base_int4": QuantizationScheme(
            targets=[r"re:(?!.*experts).*Linear"],  # match Linear layers not in 'experts'
            weights=QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                symmetric=True,
                dynamic=False,
                strategy=QuantizationStrategy.GROUP,
                group_size=128
            )
            # (No activations field here means activations stay at default 16-bit for these layers)
        )
    }
)

# Combine the recipe
recipe = [prune_mod, awq_mod, quant_mod]
A few clarifying points for the recipe:
We used regex in targets to distinguish expert vs base layers. The pattern re:.*experts\..* matches any module path that contains "experts." (which will include the expert feed-forward layers like mlp.experts.0.up_proj, etc.), ensuring those get the 8-bit scheme. The base layers use a negative lookahead regex (?!.*experts).*Linear to match Linear layers not under an "experts" submodule. This way, the two groups don’t overlap. All Linear layers will fall into one of these two groups. (Modules like the attention projections are not part of experts and will be quantized as 4-bit by the base group.)
We set symmetric=True for weight quantization in both groups. Symmetric quantization (zero-centered, no offset) is generally required for compatibility with structured sparsity in vLLM
GitHub
 – vLLM currently doesn’t support asymmetric 8-bit with 2:4 sparsity, so we enforce symmetric scales for weights. (Activations we also set symmetric=True for simplicity, though activation quantization could be asymmetric int8; symmetric is fine here and avoids negative zero-points.)
The ignore list is applied across all modifiers. By ignoring "router" and "gate_proj" in both the AWQ and Quantization steps, those layers remain in FP16. The "router" (global MoE gate) and each expert’s "gate_proj" are small (a few thousand parameters) compared to the whole model, so keeping them at high precision has minimal impact on model size but preserves model fidelity
GitHub
. We also ignore the output lm_head for the same reason – it’s typically left in higher precision to avoid hurting generation quality
GitHub
.
4. Running the One-Shot Compression
Now that the recipe is defined, we use the oneshot() function to execute all compression steps in one go. We provide the model, the calibration dataset, and the recipe. We also specify the max_seq_length (to match the model’s context length, e.g. 2048) and the number of calibration samples (for transparency, though the dataset slice already limits it). Setting save_compressed=True in oneshot (or calling model.save_pretrained(..., save_compressed=True) after) ensures the model is kept in the new compressed format in memory and will be saved as such
GitHub
.
# Apply the compression recipe in one shot
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    save_compressed=True,           # keep model weights in compressed (quantized) form
    trust_remote_code_model=True    # needed for custom model classes like GLM4 MoE
)  
During this step, the following happens under the hood:
Pruning: SparseGPTModifier prunes each linear layer to 50% sparsity (2:4 pattern). It will prune sequentially to minimize error, given we set sequential_update=True
GitHub
. Non-pruned layers (router, gate_proj, lm_head) are untouched.
AWQ Calibration: AWQModifier runs the 512 calibration samples through the model (capturing intermediate activations) and adjusts weight tensors by scaling important channels
docs.vllm.ai
. This improves the next quantization step’s accuracy, especially for 4-bit layers. (Since AWQ only scales weights, it’s fine that some layers are destined for 8-bit – AWQ’s channel scaling will mostly affect the most critical outlier channels, which tend to be in higher layers and projection matrices that are also quantized to 4-bit.)
Quantization: QuantizationModifier then applies the actual quantization. The base linear layers are quantized to 4-bit integers (with group-wise calibration if configured by group_size), and expert layers to 8-bit. Activations for expert layers will use 8-bit quantization at inference (the calibration ensures proper scaling). The model’s weights are now stored in a special compressed tensor format (int4/int8 values with calibration scales). The router, gate_proj, and lm_head remain in FP16 as we intended (since they were ignored).
5. Saving and Using the Compressed Model
After oneshot completes, the model is compressed in memory. We can now save it to disk in safetensors format with compression metadata so that it can be reloaded for inference. Use model.save_pretrained(..., save_compressed=True) to save the quantized weights and sparse masks
docs.vllm.ai
GitHub
:
OUTPUT_DIR = "GLM-4.5-Air-2of4-INT4INT8-mix"
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
This will produce a folder GLM-4.5-Air-2of4-INT4INT8-mix/ containing a pytorch_model.bin or model.safetensors (with weight compression info embedded) and the tokenizer files. The 2:4 sparsity is encoded in a custom sparse weight format; if you prefer to save weights densely (with zeros for pruned weights), you could pass disable_sparse_compression=True as well, but that would increase the disk size
docs.vllm.ai
. By default, save_compressed=True will store a truly compressed sparse format. Expected Model Size: The original FP16 model is about 218 GB (100% of weights). After 2:4 pruning, that’s effectively 50% density. We quantized ~70% of weights to 4-bit and ~30% to 8-bit (since roughly 30% of the model’s params are in the experts in this MoE design). The resulting compressed model is around 64 GB – roughly a 3.4× size reduction – matching the “FP16 (router) + INT4 base + INT8 experts + 2:4 sparsity” scenario in the table (about 63.8 GB) from the question. Finally, you can load and use this compressed model with vLLM or Hugging Face Transformers. For example, using vLLM’s inference engine:
from vllm import LLM, SamplingParams
# Load with tensor parallelism if you have multiple GPUs for inference too
llm_engine = LLM(model=OUTPUT_DIR, tensor_parallel_size=4, dtype="auto")
output = llm_engine.generate("Hello, my name is ", SamplingParams(max_tokens=100))[0].outputs[0].text
print(output)
This will automatically use the compressed weights and sparse patterns for faster inference. In summary, we’ve successfully applied 2:4 semi-structured sparsity and mixed INT4/INT8 quantization to GLM-4.5-Air using LLM Compressor. By keeping the gating and output in FP16, and leveraging AWQ for the 4-bit portions, we maintain model accuracy as much as possible while achieving significant memory and speed gains
GitHub
docs.vllm.ai
.
