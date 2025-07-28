import os
from dotenv import load_dotenv


def get_config():
    """Load environment variables and return simple config"""
    load_dotenv()
    
    return {
        "local": {
            "base_url": os.getenv("LOCAL_BASE_URL", "http://localhost:8001")
        },
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        },
        "llama": {
            "api_key": os.getenv("LLAMA_API_KEY"),
            "base_url": os.getenv("LLAMA_BASE_URL", "https://api.llama.com/compat/v1/")
        }
    }