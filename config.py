import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER", "")
PUSHOVER_API_TOKEN = os.getenv("PUSHOVER_TOKEN", "")

HF_FINE_TUNED_MODEL = os.getenv(
    "HF_FINE_TUNED_MODEL",
    "faxoi/llama-3.2-3b-price-engine-Rev2"
)
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "main")

MODAL_WORKSPACE_NAME = os.getenv("MODAL_WORKSPACE_NAME", "faxoii27")

FRONTIER_OPENAI_MODEL = "gpt-5-mini"
GROQ_MODEL = "openai/gpt-oss-120b"

# Production behavior
FRONTIER_ENABLED = os.getenv("FRONTIER_ENABLED", "true").lower() == "true"
GROQ_ENABLED = os.getenv("GROQ_ENABLED", "true").lower() == "true"
QLORA_ENABLED = os.getenv("QLORA_ENABLED", "true").lower() == "true"

DEAL_THRESHOLD = float(os.getenv("DEAL_THRESHOLD", "0.80"))
STRONG_DEAL_THRESHOLD = float(os.getenv("STRONG_DEAL_THRESHOLD", "0.70"))
