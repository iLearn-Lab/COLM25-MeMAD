from pathlib import Path

# repalce your dirs
CONFIG = {
    # "PROJECT_PATH": Path("MeMAD dir"),
    # "PERSISTENT_DIR": Path("ChromaDB persistent dir"),
    # "DATA_DIR": Path("dataset dir"),
    # "MEMORY_DATA_DIR": Path("Memory Data Dir"),
    # "CONFIG_DIR": Path("this file dir"),
    # "API_RATE_LIMIT": 60  # calls per minute
}

LLM_CONFIG = {
    "gpt35": {
        "model_name": "gpt-3.5-turbo-0125",
        "temperature": 0.7,
        "serve_provider": "openai",
        "max_retries": 3,
    },
    "gpt4omini": {
        "model_name": "gpt-4o-mini",
        "temperature": 0.7,
        "serve_provider": "openai",
        "max_retries": 3,
    },
    "gpt4o": {
        "model_name": "gpt-4o-2024-08-06",
        "temperature": 0.7,
        "serve_provider": "openai",
        "max_retries": 3,
    },
    "oqwen14b": {
        "model_name": "ollama/qwen2.5:14b-instruct-fp16",
        "temperature": 0.7,
        "serve_provider": "ollama",
        "max_retries": 3,
    },
    "oqwen32b": {
        "model_name": "ollama/qwen2.5:32b-instruct-q8_0",
        "temperature": 0.7,
        "serve_provider": "ollama",
        "max_retries": 3,
    },
    "dsv3": {
        "model_name": "deepseek-chat",
        "temperature": 0.7,
        "serve_provider": "ds",
        "max_retries": 3,
    },
}
