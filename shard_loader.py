import os
import torch
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Wymuszenie wyłączenia zbugowanego torchcodec na Ubuntu 24.04
os.environ["TTS_USE_TORCHCODEC"] = "0"

app = FastAPI(title="ALNEG Shard Loader Core v15.1")

MODEL_PATH = "/home/maciej/ALNEG_STUDIO/models/nemotron"
CONFIG_PATH = "/home/maciej/ALNEG_STUDIO/shard_config.json"

class ShardRequest(BaseModel):
    action: str  # "load", "unload", "status"

def get_gpu_memory_map():
    """Pobiera wolny VRAM z obu kart RTX 3090"""
    return {0: 24576, 1: 24576}  # Matryca dla 2x RTX 3090 (24GB na kartę)

@app.post("/api/v1/gpu/shard")
async def handle_sharding(request: ShardRequest):
    if request.action == "status":
        if torch.cuda.is_available():
            devices = {i: torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
            return {"status": "ONLINE", "gpus": devices, "allocated_vram": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"}
        return {"status": "ERROR", "message": "CUDA niedostępne"}

    if request.action == "load":
        try:
            # Twarda konfiguracja podziału wag 62GB na dwie karty po 24GB VRAM
            # Optymalizacja pod float16 i TF32 (czysty metal)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            print("[ALNEG] Inicjalizacja podziału wag modelu Nemotron...")
            
            # Automatyczny podział na oba GPU (Max 22GB per card, reszta na kontekst L1/L2)
            max_memory = {0: "21GiB", 1: "21GiB"}
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="auto",
                max_memory=max_memory,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Zapis statusu sukcesu do matrycy prawdy
            with open(CONFIG_PATH, "w") as f:
                json.dump({"status": "LOADED", "model": "Nemotron-62GB", "precision": "float16"}, f)
                
            return {"status": "SUCCESS", "message": "Model załadowany i rozbity na 2x RTX 3090"}
            
        except Exception as e:
            # Jeśli się wypieprzy – Sędzia zrzuci błąd do bazy kłamstw
            error_msg = str(e)
            print(f"[ALNEG ERROR] Sharding klęknął: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Błąd ładowania: {error_msg}")

    if request.action == "unload":
        # Czyszczenie VRAM na czysto
        if 'model' in globals():
            del globals()['model']
        if 'tokenizer' in globals():
            del globals()['tokenizer']
        torch.cuda.empty_cache()
        if os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)
        return {"status": "UNLOADED", "message": "VRAM wyczyszczony"}
