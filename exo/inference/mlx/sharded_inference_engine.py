import os
from pathlib import Path
from functools import partial
import shutil
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_sampler
import mlx.optimizers as optim
from ..inference_engine import InferenceEngine
from .sharded_utils import load_model_shard, resolve_tokenizer
from .losses import loss_fns
from ..shard import Shard
from typing import Dict, Optional, Tuple
from exo.download.shard_download import ShardDownloader
from exo.models import register_local_model
import asyncio
from collections import OrderedDict
from mlx_lm.models.cache import make_prompt_cache
from concurrent.futures import ThreadPoolExecutor

class MLXDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.caches = OrderedDict()
    self.sampler_params: tuple[float, float] = (0.0, 0.0, 0.0, 1)
    self.sampler = make_sampler(*self.sampler_params)
    self._mlx_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
    self._tokenizer_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tokenizer")
    self.session = {}
    self._shard_lock = asyncio.Lock()

    # Initialize with existing local models
    self.local_model_dir = Path.home() / ".cache/exo/downloads"
    self._loaded_shards = {}  # Initialize here before scanning
    self.local_registry = {}  # Initialize here before scanning

    self._scan_local_models()

  def _scan_local_models(self):
    """Populate initial registry of local models"""
    if not self.local_model_dir.exists():
      print(f"Local model directory {self.local_model_dir} does not exist.")
      return
    
    for model_dir in self.local_model_dir.glob("*/*"):
      if model_dir.is_dir() and (model_dir / "config.json").exists():
        model_id = str(model_dir.relative_to(self.local_model_dir))
        self._loaded_shards[model_id] = model_dir
        print(f"Loaded local model: {model_id}")
        
    # Also scan direct subdirectories
    for model_dir in self.local_model_dir.glob("*"):
      if model_dir.is_dir() and (model_dir / "config.json").exists():
        model_id = model_dir.name
        self._loaded_shards[model_id] = model_dir
        print(f"Loaded local model: {model_id}")

  async def _eval_mlx(self, *args):
    await asyncio.get_running_loop().run_in_executor(self._mlx_thread, mx.eval, *args)

  async def poll_state(self, request_id: str, max_caches=2):
    if request_id in self.caches:
      self.caches.move_to_end(request_id)
    else:
      newcache = make_prompt_cache(self.model)
      if len(self.caches) > max_caches:
        self.caches.popitem(last=False)
      self.caches[request_id] = newcache
    return {"cache": self.caches[request_id]}

  async def sample(self, x: np.ndarray, temp: float = 0.0, top_p: float = 1.0) -> np.ndarray:
    if (temp, top_p, 0.0, 1) != self.sampler_params:
      self.sampler_params = (temp, top_p, 0.0, 1)
      self.sampler = make_sampler(*self.sampler_params)
    logits = mx.array(x)
    logits = logits[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    result = self.sampler(logprobs)
    await self._eval_mlx(result)
    return np.asarray(result, dtype=int)

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    await self.ensure_shard(shard)
    return np.asarray(
      await asyncio.get_running_loop().run_in_executor(
        self._tokenizer_thread,
        self.tokenizer.encode,
        prompt
      )
    )

  async def decode(self, shard: Shard, tokens) -> str:
    await self.ensure_shard(shard)
    return await asyncio.get_running_loop().run_in_executor(
      self._tokenizer_thread,
      self.tokenizer.decode,
      tokens
    )

  async def save_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    await asyncio.get_running_loop().run_in_executor(self._mlx_thread, lambda: self.model.save_weights(path))

  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    await asyncio.get_running_loop().run_in_executor(self._mlx_thread, lambda: self.model.load_weights(path))

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    await self.ensure_shard(shard)
    state = await self.poll_state(request_id) if self.model.model_type != 'StableDiffusionPipeline' else {}
    x = mx.array(input_data)

    if self.model.model_type != 'StableDiffusionPipeline':
      output_data = await asyncio.get_running_loop().run_in_executor(
        self._mlx_thread,
        lambda: self.model(x, **state, **(inference_state or {}))
      )
      inference_state = None
    else:
      result = await asyncio.get_running_loop().run_in_executor(
        self._mlx_thread,
        lambda: self.model(x, **state, **(inference_state or {}))
      )
      output_data, inference_state = result

    await self._eval_mlx(output_data)
    output_data = await asyncio.get_running_loop().run_in_executor(
      self._mlx_thread,
      lambda: np.array(output_data, copy=False)
    )
    return output_data, inference_state

  async def evaluate(self, request_id: str, shard: Shard, inputs, targets, lengths, loss: str = "length_masked_ce"):
    await self.ensure_shard(shard)
    await self.save_session('loss', loss_fns[loss])
    x = mx.array(inputs)
    y = mx.array(targets)
    l = mx.array(lengths)

    score = await asyncio.get_running_loop().run_in_executor(
      self._mlx_thread,
      lambda: self.session['loss'](self.model, x, y, l)
    )
    return score

  async def ensure_train(self, shard: Shard, loss: str, opt=optim.SGD, lr=1e-5, trainable_layers=['input_layernorm', 'gate_proj']):
    await self.ensure_shard(shard)

    if 'train_layers' not in self.session or self.session['train_layers'] != trainable_layers:
      await self.save_session('train_layers', trainable_layers)
      def freeze_unfreeze():
        self.model.freeze()
        self.model.apply_to_modules(
          lambda k, v: v.unfreeze() if any(k.endswith(layer_name) for layer_name in trainable_layers) else None
        )
      await asyncio.get_running_loop().run_in_executor(self._mlx_thread, freeze_unfreeze)

    if 'lossname' not in self.session or 'LVaG' not in self.session or self.session['lossname'] != loss:
      await self.save_session('lossname', loss)
      await self.save_session('LVaG', nn.value_and_grad(self.model, loss_fns[loss]))

    if 'opt' not in self.session:
      await self.save_session('opt', opt(lr))
    return True

  async def train(self, request_id: str, shard: Shard, inputs, targets, lengths, loss: str = "length_masked_ce", opt=optim.SGD, lr=1e-5):
    await self.ensure_train(shard, loss, opt, lr)

    def train_step(inp, tar, lng):
      lval, grad = self.session['LVaG'](self.model, inp, tar, lng)
      gradlayers = grad['model']['layers']
      self.session['opt'].update(self.model, grad)
      return lval, gradlayers, (self.model.parameters(), self.session['opt'].state, lval)

    x = mx.array(inputs)
    y = mx.array(targets)
    l = mx.array(lengths)
    score, gradients, eval_args = await asyncio.get_running_loop().run_in_executor(
      self._mlx_thread,
      lambda: train_step(x, y, l)
    )
    await self._eval_mlx(*eval_args)

    layers = [{k: v["weight"] for k, v in layer.items() if 'weight' in v} for layer in gradients if layer]
    first_layer = np.array(layers[0]['input_layernorm'], copy=False)
    await self._eval_mlx(first_layer)
    return score, first_layer

  async def ensure_shard(self, shard: Shard):
    async with self._shard_lock:
      if self.shard == shard: return
      
      print(f"Ensuring shard for model: {shard.model_id}")
      
      # Try different path formats for local models
      local_paths = []
      
      # Original format
      local_paths.append(self.local_model_dir / shard.model_id)
      
      # HF-style format with --
      if '/' in shard.model_id:
        local_paths.append(self.local_model_dir / shard.model_id.replace("/", "--"))
      
      # Directory structure format
      if '/' in shard.model_id:
        parts = shard.model_id.split('/')
        if len(parts) == 2:
          local_paths.append(self.local_model_dir / parts[0] / parts[1])
      
      # Handle case where model_id already contains --
      if '--' in shard.model_id:
        local_paths.append(self.local_model_dir / shard.model_id.replace("--", "/"))
      
      # Try each possible local path
      for local_path in local_paths:
        if local_path.exists():
          try:
            print(f"Loading local model from: {local_path}")
            model_shard = await asyncio.get_running_loop().run_in_executor(
              self._mlx_thread,
              lambda: load_model_shard(local_path, shard, lazy=False)
            )
            if hasattr(model_shard, "tokenizer"):
              self.tokenizer = model_shard.tokenizer
            else:
              self.tokenizer = await resolve_tokenizer(local_path)
            self.shard = shard
            self.model = model_shard
            self.caches = OrderedDict()
            self.session = {}
            return
          except Exception as e:
            print(f"Local load failed for {local_path}: {e}")
      
      # If we get here, we couldn't find a local version, so try download
      print(f"Could not find local model for {shard.model_id}, attempting download")
      
      # Check if the model ID has alternative formats we should try for download
      alt_model_ids = [shard.model_id]
      if '/' in shard.model_id:
        alt_model_ids.append(shard.model_id.replace("/", "--"))
      elif '--' in shard.model_id:
        alt_model_ids.append(shard.model_id.replace("--", "/"))
      
      for model_id in alt_model_ids:
        try:
          alt_shard = Shard(model_id, shard.start_layer, shard.end_layer, shard.n_layers)
          model_path = await self.shard_downloader.ensure_shard(alt_shard, self.__class__.__name__)
          if model_path and model_path.exists():
            model_shard = await asyncio.get_running_loop().run_in_executor(
              self._mlx_thread,
              lambda: load_model_shard(model_path, alt_shard, lazy=False)
            )
            if hasattr(model_shard, "tokenizer"):
              self.tokenizer = model_shard.tokenizer
            else:
              self.tokenizer = await resolve_tokenizer(model_path)
            self.shard = alt_shard
            self.model = model_shard
            self.caches = OrderedDict()
            self.session = {}
            return
        except Exception as e:
          print(f"Failed to download model {model_id}: {e}")
      
      # If we get here, all attempts failed
      raise ValueError(f"Could not find or download model {shard.model_id}")

  async def cleanup(self):
    self._mlx_thread.shutdown(wait=True)
    # Add this to prevent resource leaks
    self._tokenizer_thread.shutdown(wait=True)
    self.shard = None
    self.model = None
    self.caches = None
    self.session = None

  async def _load_local_shard(self, model_path: Path, shard: Shard):
        """Direct local model loading without registry checks"""
        if not (model_path / "config.json").exists():
            raise FileNotFoundError("Missing config.json in local model")
        
        return (
            await asyncio.get_running_loop().run_in_executor(
                self._mlx_thread,
                partial(load_model_shard, model_path, shard, lazy=False)
            ),
            await resolve_tokenizer(model_path)
        )

  def _update_model_state(self, shard: Shard, model_shard):
        """Update state with loaded shard"""
        self.shard = shard
        self.model = model_shard
        self.stateful_sharded_model = StatefulShardedModel(shard, model_shard)
        self.caches = OrderedDict()
        self.session = {}

  async def _process_downloaded_shard(self, shard: Shard, model_path: Path):
        """Handle downloaded models and local registration"""
        if self.shard != shard:
            model_shard = await asyncio.get_running_loop().run_in_executor(
                self._mlx_thread,
                partial(load_model_shard, model_path, shard, lazy=False)
            )
            self._update_model_state(shard, model_shard)
            
            # Register in local cache
            local_dir = self.local_model_dir / shard.model_id
            if not local_dir.exists():
                local_dir.mkdir(parents=True)
                for f in model_path.glob("*"):
                    if f.is_file():
                        os.link(f, local_dir / f.name)
                self.local_registry[shard.model_id] = local_dir

class StatefulShardedModel:
    """Manages runtime state for a loaded model shard"""
    
    def __init__(self, shard: Shard, model: nn.Module):
        self.shard = shard
        self.model = model
        self.cache = None  # For attention layer caching
        
        # Initialize cache for attention layers
        self.cache = [None] * (self.shard.end_layer - self.shard.start_layer + 1)
        
    def __call__(self, inputs: mx.array):
        """Execute the model shard with current cache"""
        outputs = inputs
        new_cache = []
        
        # Process only the layers in this shard's range
        for layer_idx in range(self.shard.start_layer, self.shard.end_layer + 1):
            layer = self.model.layers[layer_idx]
            outputs, layer_cache = layer(outputs, self.cache[layer_idx] if self.cache else None)
            new_cache.append(layer_cache)
            
        self.cache = new_cache
        return outputs
    
    def reset(self):
        """Reset state between generations"""
        self.cache = None