import os
from pathlib import Path
import json
from flask import Flask, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


def clean_model_name(raw_name: str) -> str:
    """Convert directory name to human-readable format"""
    return (
        raw_name
        .replace("mlx-community--", "")  # Remove common prefix
        .replace("mistralai--", "")      # Handle Mistral models
        .replace("--", " - ")  # Handle double hyphens
        .replace("-", " ")
        .replace("_", " ")
        .title()
    )

def folder_size(folder_path: Path) -> int:
    """Calculate total size of directory in bytes"""
    total_size = 0
    try:
        for entry in folder_path.glob('**/*'):
            if entry.is_file():
                total_size += entry.stat().st_size
    except Exception as e:
        print(f"Error calculating size for {folder_path}: {e}")
    return total_size

@app.route("/initial_models")
def get_initial_models():
    """Endpoint to list locally available models"""
    models = {}
    downloads_dir = Path.home() / ".cache" / "exo" / "downloads"
    
    if not downloads_dir.exists():
        return jsonify(models)
        
    for model_dir in downloads_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        config_path = model_dir / "config.json"
        
        # Handle models with or without config files
        model_id = model_dir.name
        
        # Extract repo/model name pattern from directory structure
        if "/" in model_id:
            # The directory likely follows a repo/model pattern
            repo_name, model_name = model_id.split("/", 1)
            model_key = model_name.lower().replace(" ", "-")
        else:
            # Use the directory name directly
            model_key = model_id.lower().replace(" ", "-")
        
        try:
            config = {}
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Error loading {config_path}: {e}")
                    # Continue with empty config but don't skip the model
            
            total_bytes = folder_size(model_dir)
            
            # Determine model architecture type from config or fallback to heuristics
            architecture = config.get("architectures", ["unknown"])[0]
            model_type = config.get("model_type", "")
            
            # Use heuristics to identify model type if not in config
            if not model_type:
                if "mistral" in model_id.lower():
                    model_type = "mistral"
                elif "llama" in model_id.lower():
                    model_type = "llama"
                elif "qwen" in model_id.lower():
                    model_type = "qwen"
                else:
                    model_type = "unknown"
            
            # Provide a friendly display name
            display_name = config.get("model_name", clean_model_name(model_id.split("/")[-1]))
            
            model_data = {
                "name": display_name,
                "downloaded": True,
                "download_percentage": 100,
                "total_size": total_bytes,
                "total_downloaded": total_bytes,
                "loading": False,
                "local": True,
                "model_type": model_type,
                "architecture": architecture,
                "unregistered": True,  # Mark as unregistered for special handling
                "original_path": str(model_dir)
            }
            
            # Add optional fields
            if "quantization" in config:
                model_data["quantization"] = config["quantization"]
            
            # For special model-specific parameters
            if "mistral" in model_id.lower():
                # Mistral-specific parameters for the frontend
                if "7b" in model_id.lower() or "7B" in model_id:
                    model_data["layers"] = 32  # Mistral 7B has 32 layers
            
            models[model_key] = model_data
            
        except Exception as e:
            print(f"Error processing {model_dir}: {e}")
            continue
            
    return jsonify(models)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0", 
        port=8000,
        threaded=True,
        debug=True
    )