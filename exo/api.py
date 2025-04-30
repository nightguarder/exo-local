from exo.models import get_supported_models
from exo.local_models import register_all_local_models

class ChatGPTAPI:
  def __init__(self, node, inference_engine_name, response_timeout=60, on_chat_completion_request=None, default_model=None, system_prompt=None):
    self.node = node
    self.inference_engine_name = inference_engine_name
    self.response_timeout = response_timeout
    self.on_chat_completion_request = on_chat_completion_request
    self.default_model = default_model
    self.system_prompt = system_prompt
    self.on_model_discovery_trigger = None

  async def handle_models(self, request: Request) -> Response:
    await register_all_local_models()
    models = get_supported_models([[self.inference_engine_name]])
    return JSONResponse({"models": models})

  async def chat_completions(self, request):
    body = await request.json()
    
    # Log the incoming model request
    model_id = body.get("model", self.default_model)
    print(f"API received request for model: {model_id}")
    
    # Check if we're in offline mode
    OFFLINE_MODE = os.environ.get("EXO_OFFLINE", "0").lower() in ("1", "true", "yes")
    if OFFLINE_MODE:
        # In offline mode, ensure we have the model available locally
        from exo.local_models import get_local_models_list
        local_models = await get_local_models_list()
        
        if model_id not in local_models:
            available_models = list(local_models.keys())
            if available_models:
                fallback_model = available_models[0]
                print(f"OFFLINE MODE: Model '{model_id}' not available locally, falling back to '{fallback_model}'")
                model_id = fallback_model
            else:
                error_msg = "No models available in offline mode"
                print(f"OFFLINE MODE ERROR: {error_msg}")
                return web.json_response({"error": error_msg}, status=404)
    
    # ...other methods...