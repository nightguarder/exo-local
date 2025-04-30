import os
import json
from pathlib import Path
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional
import traceback

# Import DEBUG with a fallback value if import fails
try:
    from exo import DEBUG
except ImportError:
    DEBUG = int(os.environ.get("DEBUG", "0"))

from exo.models import register_local_model

# URL for the local server API
LOCAL_API_URL = "http://localhost:8000"

# List of directory names that should never be considered as models
INVALID_MODEL_DIRS = [
    'venv', 'env', '.venv', '__pycache__', '.git', '.github', '.idea', '.vscode',
    'bin', 'lib', 'include', 'share', 'man', 'tmp', 'temp', 'cache', '.cache',
    'node_modules', 'dist', 'build', 'target', 'out', 'output', 'site-packages'
]

# Flag to enable offline mode - set this to True to prevent downloads
OFFLINE_MODE = os.environ.get("EXO_OFFLINE", "0").lower() in ("1", "true", "yes")

def normalize_model_id(model_id: str) -> str:
    """Normalize model ID to handle different formats"""
    # Replace double hyphens with slashes for consistency
    if '--' in model_id:
        return model_id.replace('--', '/')
    return model_id

async def fetch_local_models() -> Dict:
    """Fetch models from the local API server"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{LOCAL_API_URL}/initial_models", timeout=5) as response:
                if response.status == 200:
                    models = await response.json()
                    if DEBUG >= 1:
                        print(f"Fetched {len(models)} models from local API server")
                    return models
                else:
                    if DEBUG >= 1:
                        print(f"Local API server returned status {response.status}")
                    return {}
    except Exception as e:
        if DEBUG >= 1:
            print(f"Error fetching local models: {e}")
        return {}

def determine_layers_from_model_type(model_type: str, model_id: str) -> int:
    """Determine the number of layers based on model type and ID"""
    # Default layer counts for known architectures
    if model_type == "mistral" or "mistral" in model_id.lower():
        if "7b" in model_id.lower():
            return 32
        if "8x7b" in model_id.lower():
            return 32  # 8 expert model still has 32 layers
        if "22b" in model_id.lower() or "codestral" in model_id.lower():
            return 48  # Mistral Codestral 22B has 48 layers
    elif model_type == "llama" or "llama" in model_id.lower():
        if "7b" in model_id.lower():
            return 32
        if "13b" in model_id.lower():
            return 40
        if "70b" in model_id.lower():
            return 80
    
    # Default to a reasonable value if we can't determine
    return 32  # Common layer count for 7B models

# Add a function to check if a model ID is valid
def is_valid_model_id(model_id: str) -> bool:
    """Check if a model ID appears to be a valid model and not a system directory."""
    # Skip common non-model directories
    for invalid_dir in INVALID_MODEL_DIRS:
        if invalid_dir == model_id.lower() or f"/{invalid_dir}" in model_id.lower() or f"\\{invalid_dir}" in model_id.lower():
            if DEBUG >= 1:
                print(f"Skipping invalid model directory: {model_id}")
            return False
    
    # Skip directories that start with a dot (hidden directories)
    parts = model_id.split('/')
    for part in parts:
        if part.startswith('.'):
            if DEBUG >= 1:
                print(f"Skipping hidden directory: {model_id}")
            return False
    
    # For Mistral models, require the name to contain common model indicators
    if "mistral" in model_id.lower():
        model_indicators = ['7b', '8b', '13b', '70b', '22b', 'instruct', 'chat', 'v0.1', 'v0.2', 'finetune', 'codestral']
        if not any(indicator in model_id.lower() for indicator in model_indicators):
            if DEBUG >= 1:
                print(f"Skipping invalid Mistral model (missing size indicator): {model_id}")
            return False
    
    return True

async def scan_cache_directory() -> Dict:
    """Scan the cache directory for local models"""
    try:
        # Define path to cache directory
        cache_dir = Path(os.path.expanduser("~/.cache/exo/downloads"))
        if not cache_dir.exists():
            if DEBUG >= 1:
                print(f"Cache directory not found: {cache_dir}")
            return {}
        
        models = {}
        # Scan all subdirectories
        for model_dir in cache_dir.glob("*"):
            if not model_dir.is_dir():
                continue
                
            model_id = model_dir.name.replace("--", "/")
            
            # Validate model ID
            if not is_valid_model_id(model_id):
                continue
                
            # Check for config.json
            config_path = model_dir / "config.json"
            if not config_path.exists():
                continue
                
            # Read config.json
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
            except Exception as e:
                if DEBUG >= 1:
                    print(f"Error reading config.json for {model_id}: {e}")
                continue
                
            # Determine model type and layers
            model_type = config.get("model_type", "")
            if not model_type:
                if "mistral" in model_id.lower():
                    model_type = "mistral"
                elif "llama" in model_id.lower():
                    model_type = "llama"
            
            # Determine layer count
            num_layers = config.get("num_hidden_layers", 0)
            if num_layers == 0:
                num_layers = determine_layers_from_model_type(model_type, model_id)
                
            # Store model info
            models[model_id] = {
                "name": model_id.split("/")[-1].replace("-", " ").title(),
                "model_type": model_type,
                "layers": num_layers,
                "downloaded": True,
                "local": True,
                "download_percentage": 100
            }
            
            if DEBUG >= 1:
                print(f"Found local model: {model_id} ({model_type})")
                
        return models
    except Exception as e:
        if DEBUG >= 1:
            print(f"Error scanning cache directory: {e}")
            traceback.print_exc()
        return {}

async def register_all_local_models():
    """Discover and register all local models"""
    # Get models from the local server
    local_models_api = await fetch_local_models()
    
    # Get models from the cache directory
    cached_models = await scan_cache_directory()
    
    # Merge models
    all_models = {**local_models_api, **cached_models}
    
    # Print a clear message when in offline mode
    if OFFLINE_MODE:
        print(f"🔌 OFFLINE MODE: Found {len(all_models)} locally available models")
        if len(all_models) == 0:
            print("⚠️ Warning: No local models found. You may need to download models first.")
    
    for model_id, model_info in all_models.items():
        # Validate the model ID before registering
        if not is_valid_model_id(model_id):
            continue
            
        # Get model architecture if available
        model_type = model_info.get("model_type", "")
        
        # If no model_type is provided, try to infer from the name
        if not model_type:
            if "mistral" in model_id.lower():
                model_type = "mistral"
                print(f"Setting model_type to 'mistral' for {model_id}")
            elif "llama" in model_id.lower():
                model_type = "llama"
        
        # Determine layer count
        layers = model_info.get("layers", 0)
        if layers == 0:
            layers = determine_layers_from_model_type(model_type, model_id)
        
        # Register the model with exo's model registry
        model_config = {
            "layers": layers,
            "repo": {
                "MLXDynamicShardInferenceEngine": model_id,
                "TinygradDynamicShardInferenceEngine": model_id
            },
            "model_type": model_type
        }
        
        if DEBUG >= 1:
            print(f"Registering local model: {model_id} ({model_type}) with {layers} layers")
        
        register_local_model(model_id, model_config)
        
        # If the original ID is different from normalized, register that too
        if model_id != normalize_model_id(model_id):
            alt_config = model_config.copy()
            alt_config["repo"] = {
                "MLXDynamicShardInferenceEngine": normalize_model_id(model_id),
                "TinygradDynamicShardInferenceEngine": normalize_model_id(model_id)
            }
            register_local_model(normalize_model_id(model_id), alt_config)
            if DEBUG >= 1:
                print(f"Also registered with original ID: {model_id}")
