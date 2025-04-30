from exo.inference.shard import Shard
from typing import Optional, List

model_cards = {
  ### llama
  "llama-3.3-70b": {
    "layers": 80,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.3-70B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.3-70B-Instruct",
    },
  },
  "llama-3.2-1b": {
    "layers": 16,
    "repo": {
      "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-1B-Instruct-4bit",
      "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-1B-Instruct",
    },
  },
  "llama-3.2-1b-8bit": {
    "layers": 16,
    "repo": {
      "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-1B-Instruct-8bit",
      "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-1B-Instruct",
    },
  },
  "llama-3.2-3b": {
    "layers": 28,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct",
    },
  },
  "llama-3.2-3b-8bit": {
    "layers": 28,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct-8bit",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct",
    },
  },
  "llama-3.2-3b-bf16": {
    "layers": 28,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct",
    },
  },
  "llama-3.1-8b": {
    "layers": 32,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    },
  },
  "llama-3.1-70b": {
    "layers": 80,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "NousResearch/Meta-Llama-3.1-70B-Instruct",
    },
  },
  "llama-3.1-70b-bf16": {
    "layers": 80,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED",
       "TinygradDynamicShardInferenceEngine": "NousResearch/Meta-Llama-3.1-70B-Instruct",
    },
  },
  "llama-3-8b": {
    "layers": 32,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R",
    },
  },
  "llama-3-70b": {
    "layers": 80,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3-70B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-70B-R",
    },
  },
  "llama-3.1-405b": { "layers": 126, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-405B-4bit", }, },
  "llama-3.1-405b-8bit": { "layers": 126, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-405B-Instruct-8bit", }, },
  ### mistral
  "mistral-nemo": { "layers": 40, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Mistral-Nemo-Instruct-2407-4bit", }, },
  "mistral-large": { "layers": 88, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Mistral-Large-Instruct-2407-4bit", }, },
  ### deepseek
  "deepseek-coder-v2-lite": { "layers": 27, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx", }, },
  "deepseek-coder-v2.5": { "layers": 60, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-V2.5-MLX-AQ4_1_64", }, },
  "deepseek-v3": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-V3-4bit", }, },
  "deepseek-v3-3bit": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-V3-3bit", }, },
  "deepseek-r1": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-4bit", }, },
  "deepseek-r1-3bit": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-3bit", }, },
  ### deepseek distills
  "deepseek-r1-distill-qwen-1.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/deepseek-r1-distill-qwen-1.5b", }, },
  "deepseek-r1-distill-qwen-1.5b-3bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-3bit", }, },
  "deepseek-r1-distill-qwen-1.5b-6bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-6bit", }, },
  "deepseek-r1-distill-qwen-1.5b-8bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-8bit", }, },
  "deepseek-r1-distill-qwen-1.5b-bf16": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-bf16", }, },
  "deepseek-r1-distill-qwen-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit", }, },
  "deepseek-r1-distill-qwen-7b-3bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-3bit", }, },
  "deepseek-r1-distill-qwen-7b-6bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-6bit", }, },
  "deepseek-r1-distill-qwen-7b-8bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit", }, },
  "deepseek-r1-distill-qwen-7b-bf16": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-bf16", }, },
  "deepseek-r1-distill-qwen-14b": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit", }, },
  "deepseek-r1-distill-qwen-14b-3bit": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-3bit", }, },
  "deepseek-r1-distill-qwen-14b-6bit": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-6bit", }, },
  "deepseek-r1-distill-qwen-14b-8bit": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-8bit", }, },
  "deepseek-r1-distill-qwen-14b-bf16": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-bf16", }, },
  "deepseek-r1-distill-qwen-32b": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit", }, },
  "deepseek-r1-distill-qwen-32b-3bit": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-3bit", }, },
  "deepseek-r1-distill-qwen-32b-6bit": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-6bit", }, },
  "deepseek-r1-distill-qwen-32b-8bit": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit", }, },
  "deepseek-r1-distill-qwen-32b-bf16": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-bf16", }, },
  "deepseek-r1-distill-llama-8b": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit", }, },
  "deepseek-r1-distill-llama-8b-3bit": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-3bit", }, },
  "deepseek-r1-distill-llama-8b-6bit": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-6bit", }, },
  "deepseek-r1-distill-llama-8b-8bit": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-8bit", }, },
  "deepseek-r1-distill-llama-8b-bf16": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-bf16", }, },
  "deepseek-r1-distill-llama-70b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit", }, },
  "deepseek-r1-distill-llama-70b-3bit": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-3bit", }, },
  "deepseek-r1-distill-llama-70b-6bit": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-6bit", }, },
  "deepseek-r1-distill-llama-70b-8bit": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-8bit", }, },
  ### llava
  "llava-1.5-7b-hf": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "llava-hf/llava-1.5-7b-hf", }, },
  ### qwen
  "qwen-2.5-0.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", }, },
  "qwen-2.5-1.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-1.5B-Instruct-4bit", }, },
  "qwen-2.5-coder-1.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit", }, },
  "qwen-2.5-3b": { "layers": 36, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-3B-Instruct-4bit", }, },
  "qwen-2.5-coder-3b": { "layers": 36, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit", }, },
  "qwen-2.5-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-7B-Instruct-4bit", }, },
  "qwen-2.5-coder-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", }, },
  "qwen-2.5-math-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-7B-Instruct-4bit", }, },
  "qwen-2.5-14b": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-14B-Instruct-4bit", }, },
  "qwen-2.5-coder-14b": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit", }, },
  "qwen-2.5-32b": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-32B-Instruct-4bit", }, },
  "qwen-2.5-coder-32b": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", }, },
  "qwen-2.5-72b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-72B-Instruct-4bit", }, },
  "qwen-2.5-math-72b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-72B-Instruct-4bit", }, },
  ### nemotron
  "nemotron-70b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/nvidia_Llama-3.1-Nemotron-70B-Instruct-HF_4bit", }, },
  "nemotron-70b-bf16": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-bf16", }, },
  # gemma
  "gemma2-9b": { "layers": 42, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-9b-it-4bit", }, },
  "gemma2-27b": { "layers": 46, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-27b-it-4bit", }, },
  # stable diffusion
  "stable-diffusion-2-1-base": { "layers": 31, "repo": { "MLXDynamicShardInferenceEngine": "stabilityai/stable-diffusion-2-1-base" } },
  # phi
  "phi-3.5-mini": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Phi-3.5-mini-instruct-4bit", }, },
  "phi-4": { "layers": 40, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/phi-4-4bit", }, },
  # dummy
  "dummy": { "layers": 8, "repo": { "DummyInferenceEngine": "dummy", }, },
}

# Dictionary to store dynamically registered local models
local_models = {}

def register_local_model(model_id: str, config: dict):
    """Register a local model that isn't in the standard model_cards"""
    if model_id in local_models:
        return  # Already registered
    
    # Store model info in the local_models dictionary
    local_models[model_id] = config
    
    # Add pretty name for the model
    if model_id not in pretty_name:
        # Extract a sensible display name from the model_id
        display_name = model_id.split('/')[-1].replace('-', ' ').title()
        pretty_name[model_id] = f"{display_name} (Local)"

pretty_name = {
  "llama-3.3-70b": "Llama 3.3 70B",
  "llama-3.2-1b": "Llama 3.2 1B",
  "llama-3.2-1b-8bit": "Llama 3.2 1B (8-bit)",
  "llama-3.2-3b": "Llama 3.2 3B",
  "llama-3.2-3b-8bit": "Llama 3.2 3B (8-bit)",
  "llama-3.2-3b-bf16": "Llama 3.2 3B (BF16)",
  "llama-3.1-8b": "Llama 3.1 8B",
  "llama-3.1-70b": "Llama 3.1 70B",
  "llama-3.1-70b-bf16": "Llama 3.1 70B (BF16)",
  "llama-3.1-405b": "Llama 3.1 405B",
  "llama-3.1-405b-8bit": "Llama 3.1 405B (8-bit)",
  "gemma2-9b": "Gemma2 9B",
  "gemma2-27b": "Gemma2 27B",
  "nemotron-70b": "Nemotron 70B",
  "nemotron-70b-bf16": "Nemotron 70B (BF16)",
  "mistral-nemo": "Mistral Nemo",
  "mistral-large": "Mistral Large",
  "deepseek-coder-v2-lite": "Deepseek Coder V2 Lite",
  "deepseek-coder-v2.5": "Deepseek Coder V2.5",
  "deepseek-v3": "Deepseek V3 (4-bit)",
  "deepseek-v3-3bit": "Deepseek V3 (3-bit)",
  "deepseek-r1": "Deepseek R1 (4-bit)",
  "deepseek-r1-3bit": "Deepseek R1 (3-bit)",
  "llava-1.5-7b-hf": "LLaVa 1.5 7B (Vision Model)",
  "qwen-2.5-0.5b": "Qwen 2.5 0.5B",
  "qwen-2.5-1.5b": "Qwen 2.5 1.5B",
  "qwen-2.5-coder-1.5b": "Qwen 2.5 Coder 1.5B",
  "qwen-2.5-3b": "Qwen 2.5 3B",
  "qwen-2.5-coder-3b": "Qwen 2.5 Coder 3B",
  "qwen-2.5-7b": "Qwen 2.5 7B",
  "qwen-2.5-coder-7b": "Qwen 2.5 Coder 7B",
  "qwen-2.5-math-7b": "Qwen 2.5 7B (Math)",
  "qwen-2.5-14b": "Qwen 2.5 14B",
  "qwen-2.5-coder-14b": "Qwen 2.5 Coder 14B",
  "qwen-2.5-32b": "Qwen 2.5 32B",
  "qwen-2.5-coder-32b": "Qwen 2.5 Coder 32B",
  "qwen-2.5-72b": "Qwen 2.5 72B",
  "qwen-2.5-math-72b": "Qwen 2.5 72B (Math)",
  "phi-3.5-mini": "Phi-3.5 Mini",
  "phi-4": "Phi-4",
  "llama-3-8b": "Llama 3 8B",
  "llama-3-70b": "Llama 3 70B",
  "stable-diffusion-2-1-base": "Stable Diffusion 2.1",
  "deepseek-r1-distill-qwen-1.5b": "DeepSeek R1 Distill Qwen 1.5B",
  "deepseek-r1-distill-qwen-1.5b-3bit": "DeepSeek R1 Distill Qwen 1.5B (3-bit)",
  "deepseek-r1-distill-qwen-1.5b-6bit": "DeepSeek R1 Distill Qwen 1.5B (6-bit)",
  "deepseek-r1-distill-qwen-1.5b-8bit": "DeepSeek R1 Distill Qwen 1.5B (8-bit)",
  "deepseek-r1-distill-qwen-1.5b-bf16": "DeepSeek R1 Distill Qwen 1.5B (BF16)",
  "deepseek-r1-distill-qwen-7b": "DeepSeek R1 Distill Qwen 7B",
  "deepseek-r1-distill-qwen-7b-3bit": "DeepSeek R1 Distill Qwen 7B (3-bit)",
  "deepseek-r1-distill-qwen-7b-6bit": "DeepSeek R1 Distill Qwen 7B (6-bit)",
  "deepseek-r1-distill-qwen-7b-8bit": "DeepSeek R1 Distill Qwen 7B (8-bit)",
  "deepseek-r1-distill-qwen-7b-bf16": "DeepSeek R1 Distill Qwen 7B (BF16)",
  "deepseek-r1-distill-qwen-14b": "DeepSeek R1 Distill Qwen 14B",
  "deepseek-r1-distill-qwen-14b-3bit": "DeepSeek R1 Distill Qwen 14B (3-bit)",
  "deepseek-r1-distill-qwen-14b-6bit": "DeepSeek R1 Distill Qwen 14B (6-bit)",
  "deepseek-r1-distill-qwen-14b-8bit": "DeepSeek R1 Distill Qwen 14B (8-bit)",
  "deepseek-r1-distill-qwen-14b-bf16": "DeepSeek R1 Distill Qwen 14B (BF16)",
  "deepseek-r1-distill-qwen-32b": "DeepSeek R1 Distill Qwen 32B",
  "deepseek-r1-distill-qwen-32b-3bit": "DeepSeek R1 Distill Qwen 32B (3-bit)",
  "deepseek-r1-distill-qwen-32b-8bit": "DeepSeek R1 Distill Qwen 32B (8-bit)",
  "deepseek-r1-distill-qwen-32b-bf16": "DeepSeek R1 Distill Qwen 32B (BF16)",
  "deepseek-r1-distill-llama-8b-8bit": "DeepSeek R1 Distill Llama 8B (8-bit)",
  "deepseek-r1-distill-llama-70b-6bit": "DeepSeek R1 Distill Llama 70B (6-bit)",
  "deepseek-r1-distill-llama-70b-8bit": "DeepSeek R1 Distill Llama 70B (8-bit)",
  "deepseek-r1-distill-llama-8b": "DeepSeek R1 Distill Llama 8B",
  "deepseek-r1-distill-llama-8b-3bit": "DeepSeek R1 Distill Llama 8B (3-bit)",
  "deepseek-r1-distill-llama-8b-6bit": "DeepSeek R1 Distill Llama 8B (6-bit)",
  "deepseek-r1-distill-llama-8b-8bit": "DeepSeek R1 Distill Llama 8B (8-bit)",
  "deepseek-r1-distill-llama-8b-bf16": "DeepSeek R1 Distill Llama 8B (BF16)",
  "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill Llama 70B",
  "deepseek-r1-distill-llama-70b-3bit": "DeepSeek R1 Distill Llama 70B (3-bit)",
  "deepseek-r1-distill-llama-70b-6bit": "DeepSeek R1 Distill Llama 70B (6-bit)",
  "deepseek-r1-distill-llama-70b-8bit": "DeepSeek R1 Distill Llama 70B (8-bit)",
  "deepseek-r1-distill-qwen-32b-6bit": "DeepSeek R1 Distill Qwen 32B (6-bit)",
}

def normalize_model_id(model_id: str) -> str:
    """Normalize model ID for consistent lookups"""
    # First check if this is mlx-community--model format
    if model_id.startswith('mlx-community--'):
        return model_id.replace('--', '/', 1)
    return model_id

def is_valid_model(model_id: str, inference_engine_classname: str) -> bool:
    """Check if a model is registered and valid for the given inference engine"""
    normalized_id = normalize_model_id(model_id)
    
    # Check if model exists in standard model cards
    model_config = model_cards.get(model_id) or model_cards.get(normalized_id)
    if model_config and inference_engine_classname in model_config.get("repo", {}):
        return True
    
    # Check if model exists in local models
    local_config = local_models.get(model_id) or local_models.get(normalized_id)
    if local_config and inference_engine_classname in local_config.get("repo", {}):
        return True
    
    return False

def get_repo(model_id: str, inference_engine_classname: str) -> Optional[str]:
    """Get repository for the given model ID and inference engine"""
    normalized_id = normalize_model_id(model_id)
    
    # First try with the original model_id
    repo = model_cards.get(model_id, {}).get("repo", {}).get(inference_engine_classname, None)
    if repo:
        return repo
    
    # Then try with the normalized ID for standard models
    if normalized_id != model_id:
        repo = model_cards.get(normalized_id, {}).get("repo", {}).get(inference_engine_classname, None)
        if repo:
            return repo
    
    # Then check local models with original ID
    local_config = local_models.get(model_id, {})
    local_repo = local_config.get("repo", {}).get(inference_engine_classname, None)
    if local_repo:
        return local_repo
    
    # Finally check local models with normalized ID
    if normalized_id != model_id:
        local_config = local_models.get(normalized_id, {})
        local_repo = local_config.get("repo", {}).get(inference_engine_classname, None)
        if local_repo:
            return local_repo
    
    # If we're looking for a model that contains "mistral" in its name,
    # we might be able to use mistral-compatible architectures
    if "mistral" in model_id.lower() and inference_engine_classname == "MLXDynamicShardInferenceEngine":
        # Only return a default repo for well-known Mistral variants
        if any(x in model_id.lower() for x in ["7b", "instruct", "chat"]):
            return "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
    
    # No valid repo found - return None instead of model_id to prevent download attempts
    return None

def get_pretty_name(model_id: str) -> Optional[str]:
    return pretty_name.get(model_id, model_id.split('/')[-1].replace('-', ' ').title())

def build_base_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
    """Build a base shard for the given model ID and inference engine"""
    normalized_id = normalize_model_id(model_id)
    
    # First check standard models with original ID
    n_layers = model_cards.get(model_id, {}).get("layers", 0)
    if n_layers > 0:
        repo = get_repo(model_id, inference_engine_classname)
        if repo is not None:
            return Shard(model_id, 0, 0, n_layers)
    
    # Then try with normalized ID
    if normalized_id != model_id:
        n_layers = model_cards.get(normalized_id, {}).get("layers", 0)
        if n_layers > 0:
            repo = get_repo(normalized_id, inference_engine_classname)
            if repo is not None:
                return Shard(normalized_id, 0, 0, n_layers)
    
    # Then check local models with original ID
    n_layers = local_models.get(model_id, {}).get("layers", 0)
    if n_layers > 0:
        return Shard(model_id, 0, 0, n_layers)
    
    # Then check local models with normalized ID
    if normalized_id != model_id:
        n_layers = local_models.get(normalized_id, {}).get("layers", 0)
        if n_layers > 0:
            return Shard(normalized_id, 0, 0, n_layers)
    
    # For completely unregistered models, try to infer from name
    if "mistral" in model_id.lower():
        if "7b" in model_id.lower():
            return Shard(model_id, 0, 0, 32)  # Mistral 7B has 32 layers
        if "22b" in model_id.lower() or "codestral" in model_id.lower():
            return Shard(model_id, 0, 0, 48)  # Mistral Codestral 22B has 48 layers
    
    return None

def build_full_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
  base_shard = build_base_shard(model_id, inference_engine_classname)
  if base_shard is None: return None
  return Shard(base_shard.model_id, 0, base_shard.n_layers - 1, base_shard.n_layers)

def get_supported_models(supported_inference_engine_lists: Optional[List[List[str]]] = None) -> List[str]:
    result = []
    
    # Get standard models
    if not supported_inference_engine_lists:
        result = list(model_cards.keys())
    else:
        from exo.inference.inference_engine import inference_engine_classes
        supported_inference_engine_lists = [
            [inference_engine_classes[engine] if engine in inference_engine_classes else engine for engine in engine_list]
            for engine_list in supported_inference_engine_lists
        ]

        def has_any_engine(model_info: dict, engine_list: List[str]) -> bool:
            return any(engine in model_info.get("repo", {}) for engine in engine_list)

        def supports_all_engine_lists(model_info: dict) -> bool:
            return all(has_any_engine(model_info, engine_list)
                    for engine_list in supported_inference_engine_lists)

        result = [
            model_id for model_id, model_info in model_cards.items()
            if supports_all_engine_lists(model_info)
        ]
    
    # Add local models
    result.extend(local_models.keys())
    
    return result
