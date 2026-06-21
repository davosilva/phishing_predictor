# token_verify.py - Script para verificar el token de Hugging Face
from huggingface_hub import HfApi

# Reemplaza con tu token real
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

try:
    api = HfApi()
    user = api.whoami(token=HF_TOKEN)  # <--- PARÉNTESIS CORREGIDO
    print(f"✅ Token válido - Usuario: {user['name']}")
    print(f"✅ Token: {HF_TOKEN[:10]}...{HF_TOKEN[-10:]}")
except Exception as e:
    print(f"❌ Error con el token: {e}")
    print("\n📌 Soluciones:")
    print("1. Verifica que el token sea correcto")
    print("2. Obtén un nuevo token en: https://huggingface.co/settings/tokens")
