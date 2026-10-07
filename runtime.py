"""Load local configuration without overriding explicit environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

def configure():
    load_dotenv(ROOT / '.env', override=False)
    load_dotenv(ROOT / '.local.env', override=False)
    for key, value in {
        'AGG_USE_DEEPSEEK_FINAL': 'false',
        'ORCH_ENABLE_ROUTE_DEEPSEEK_CORRECTION': 'false',
        'ORCH_FORCE_ROUTE_DEEPSEEK_CORRECTION': 'false',
        'TIKTOKEN_CACHE_DIR': str(ROOT / 'data/tokenizer_cache'),
    }.items():
        os.environ.setdefault(key, value)
