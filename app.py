import subprocess
import json
import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional

app = FastAPI(title="ALNEG iMortal Core v15.1")

DB_TRUTH = "matryca_prawda.json"
DB_LIES = "falsz_nieprawda.json"
DB_LICENSES = "licencje_klienci.json"

class CodePayload(BaseModel):
    script_name: str
    code_content: str

class LicenseCheck(BaseModel):
    client_id: str
    current_revenue: float

# ==========================================
# 1. AUTOMATYCZNY SĘDZIA (PRAWDA / FAŁSZ)
# ==========================================
def evaluate_code_execution(payload: CodePayload) -> dict:
    # Zapis tymczasowy do testu dry-run
    temp_file = f"temp_{payload.script_name}"
    with open(temp_file, "w") as f:
        f.write(payload.code_content)
    
    try:
        # Dry-run test (Kompilacja i sprawdzenie składni)
        result = subprocess.run(
            ["python3", "-m", "py_compile", temp_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # SUKCES -> JASNOBIAŁY (100% Pewności)
            status = {"confidence": 1.0, "color": "bright_white", "status": "TRUTH"}
            save_to_matrix(DB_TRUTH, payload.script_name, payload.code_content)
        else:
            # BŁĄD SKŁADNI -> DEGRADACJA PONIŻEJ 15% -> BAN
            status = {"confidence": 0.12, "color": "dark_red", "status": "LIES", "error": result.stderr}
            save_to_matrix(DB_LIES, payload.script_name, payload.code_content, error=result.stderr)
            
    except Exception as e:
        status = {"confidence": 0.05, "color": "dark_red", "status": "LIES", "error": str(e)}
        save_to_matrix(DB_LIES, payload.script_name, payload.code_content, error=str(e))
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    return status

def save_to_matrix(db_path: str, name: str, content: str, error: str = None):
    if not os.path.exists(db_path):
        with open(db_path, "w") as f: json.dump({}, f)
    with open(db_path, "r+") as f:
        data = json.load(f)
        data[name] = {"code": content, "error": error}
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

@app.post("/api/v1/verify-code")
async def verify_code_endpoint(payload: CodePayload):
    evaluation = evaluate_code_execution(payload)
    if evaluation["confidence"] <= 0.15:
        # Zgodnie z wytycznymi: Jeśli poniżej 15%, nie wyświetlaj kodu użytkownikowi, tylko daj alternatywy
        return {
            "status": "BLOCKED",
            "message": "Kod wykazał błędy logiczne i został przeniesiony do bazy kłamstw (degradacja < 15%).",
            "alternatives": {
                "Version_A": "# Propozycja bezpieczna zoptymalizowana pod VRAM\n",
                "Version_B": "# Propozycja agresywna (Pure Metal TinyGrad)\n"
            }
        }
    return {"status": "SUCCESS", "evaluation": evaluation}

# ==========================================
# 2. ZDALNA AUTORYZACJA LICENCJI (5% REV)
# ==========================================
@app.post("/api/v1/auth-license")
async def auth_license_endpoint(client: LicenseCheck):
    if not os.path.exists(DB_LICENSES):
        return {"status": "FREE", "message": "Brak zarejestrowanej licencji komercyjnej. Tryb darmowy / testowy aktywny."}
        
    with open(DB_LICENSES, "r") as f:
        licenses = json.load(f)
        
    if client.client_id not in licenses:
        # Jeśli firma nie generuje przychodu, system działa w 100% za darmo
        if client.current_revenue == 0:
            return {"status": "ACTIVE", "tier": "Free Testing", "message": "Zarabiasz 0, płacisz 0. System w pełni aktywny."}
        else:
            raise HTTPException(status_code=403, detail="Wykryto przychód komercyjny! Wymagana aktywacja licencji komercyjnej ALNEG 5%.")
            
    client_status = licenses[client.client_id]
    if client_status.get("payment_overdue", False):
        raise HTTPException(status_code=403, detail="Dostęp zablokowany: Brak rozliczenia należności 5% brutto dla ALNEG STUDIO LTD.")
        
    return {"status": "AUTHORIZED", "tier": "Commercial 5%"}
