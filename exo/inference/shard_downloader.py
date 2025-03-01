from pathlib import Path
from huggingface_hub import snapshot_download
from exo.inference.shard import Shard

class ShardDownloader:
    async def download(self, shard: Shard) -> Path:
        """Modified download to support local cache"""
        cache_path = Path.home() / ".cache/exo/downloads" / shard.model_id
        
        if not cache_path.exists():
            snapshot_download(
                repo_id=shard.model_id,
                local_dir=cache_path,
                allow_patterns=["*.json", "*.safetensors", "*.model"]
            )
            
        return cache_path