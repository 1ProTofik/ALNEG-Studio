import os
import gc
import json
import threading
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Wymuszenie wyłączenia zbugowanego torchcodec na Ubuntu 24.04
os.environ["TTS_USE_TORCHCODEC"] = "0"

router = APIRouter()

MODEL_PATH = os.environ.get("ALNEG_MODEL_PATH", "/home/maciej/ALNEG_STUDIO/models/nemotron")
CONFIG_PATH = os.environ.get("ALNEG_SHARD_CONFIG_PATH", "/home/maciej/ALNEG_STUDIO/shard_config.json")
MAX_MEMORY_PER_GPU = os.environ.get("ALNEG_MAX_MEMORY_PER_GPU", "21GiB")


class ShardState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None


state = ShardState()


class ShardRequest(BaseModel):
    action: str  # "load", "unload", "status"


@router.post("/api/v1/gpu/shard")
def handle_sharding(request: ShardRequest):
    if request.action == "status":
        model_status = "LOADED" if state.loaded else "UNLOADED"
        try:
            import torch
        except ImportError:
            return {"status": "ONLINE", "model": model_status, "message": "torch niedostępne"}
        if torch.cuda.is_available():
            devices = {i: torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
            return {
                "status": "ONLINE",
                "model": model_status,
                "gpus": devices,
                "allocated_vram": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            }
        return {"status": "ERROR", "model": model_status, "message": "CUDA niedostępne"}

    if request.action == "load":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with state.lock:
            if state.loaded:
                return {"status": "ALREADY_LOADED"}
            try:
                # Twarda konfiguracja podziału wag 62GB na dwie karty po 24GB VRAM
                # Optymalizacja pod float16 i TF32 (czysty metal)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                print("[ALNEG] Inicjalizacja podziału wag modelu Nemotron...")

                max_memory = {0: MAX_MEMORY_PER_GPU, 1: MAX_MEMORY_PER_GPU}

                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    device_map="auto",
                    max_memory=max_memory,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )

                state.model = model
                state.tokenizer = tokenizer

                # Zapis statusu sukcesu do matrycy prawdy
                config_dir = os.path.dirname(CONFIG_PATH)
                if config_dir:
                    os.makedirs(config_dir, exist_ok=True)
                with open(CONFIG_PATH, "w") as f:
                    json.dump(
                        {"status": "LOADED", "model": "Nemotron-62GB", "precision": "float16"},
                        f,
                    )

                return {"status": "SUCCESS", "message": "Model załadowany i rozbity na 2x RTX 3090"}

            except Exception as e:
                state.model = None
                state.tokenizer = None
                error_msg = str(e)
                print(f"[ALNEG ERROR] Sharding klęknął: {error_msg}")
                raise HTTPException(status_code=500, detail=f"Błąd ładowania: {error_msg}")

    if request.action == "unload":
        with state.lock:
            state.model = None
            state.tokenizer = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            if os.path.exists(CONFIG_PATH):
                os.remove(CONFIG_PATH)
            return {"status": "UNLOADED", "message": "VRAM wyczyszczony"}

    raise HTTPException(status_code=400, detail=f"Nieznana akcja: {request.action}")
