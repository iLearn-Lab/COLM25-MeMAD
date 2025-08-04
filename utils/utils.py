import sys
import os
from typing import Optional, List, Dict
import re
import time
import asyncio

import logging
import json
from pathlib import Path

# sys.path.append("MeMAD folder Path")

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import LLMMessage
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.config import CONFIG


def write_json(data, file_path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def format_mmlupro_choices(options: Dict[str, str]) -> str:
    return '\n'.join([idx + '. ' + item for idx, item in list(options.items())])


class DataLoader:

    @staticmethod
    def load_dataset(question_type: str) -> Dict:
        dataset_paths = {
            "MATH500": "data path",
            "MATH500_TRAIN": "data path",
            "GPQA": "data path",
            "GPQA_TRAIN": "data path",
            "MMLUPro_Law": "data path",
            "MMLUPro_Law_TRAIN": "data path",
            "MMLUPro_Economics": "data path",
            "MMLUPro_Economics_TRAIN": "data path",
            "MMLUPro_Math_Valid": "data path"
        }

        if question_type not in dataset_paths:
            raise ValueError(f"DataLoader: Unsupported question type in reading data: {question_type}")

        dataset = read_json(CONFIG["DATA_DIR"] / dataset_paths[question_type])
        return dataset

    @staticmethod
    def format_question(item: Dict, question_type: str) -> tuple:
        formatters = {
            "MATH500": lambda x: (x['problem'], x['answer'], x["solution"], x["subject"]),
            "MATH500_TRAIN": lambda x: (x['problem'], x['answer'], x["solution"], x["subject"]),
            "GPQA": lambda x: (
                f"Question: {x['question']}\n\nA. {x['A']}\nB. {x['B']}\nC. {x['C']}\nD. {x['D']}\n",
                x['answer'],
                x['solution'],
                x["domain"]
            ),
            "GPQA_TRAIN": lambda x: (
                f"Question: {x['question']}\n\nA. {x['A']}\nB. {x['B']}\nC. {x['C']}\nD. {x['D']}\n",
                x['answer'],
                x['solution'],
                x["domain"]
            ),
            "MMLUPro_Law": lambda x: (
                f"Question: {x['question']}\n\n{format_mmlupro_choices(x['options'])}",
                x['answer'],
                x['answer'],
                "law"
            ),
            "MMLUPro_Law_TRAIN": lambda x: (
                f"Question: {x['question']}\n\n{format_mmlupro_choices(x['options'])}",
                x['answer'],
                x['answer'],
                "law"
            ),
            "MMLUPro_Economics": lambda x: (
                f"Question: {x['question']}\n\n{format_mmlupro_choices(x['options'])}",
                x['answer'],
                x['answer'],
                "economics"
            ),
            "MMLUPro_Economics_TRAIN": lambda x: (
                f"Question: {x['question']}\n\n{format_mmlupro_choices(x['options'])}",
                x['answer'],
                x['answer'],
                "economics"
            ),
            "MMLUPro_Math_Valid": lambda x: (
                f"Question: {x['question']}\n\n{format_mmlupro_choices(x['options'])}",
                x['answer'],
                x['answer'],
                "math"
            )
        }

        if question_type not in formatters:
            raise ValueError(f"DataLoader: Unsupported question type in format question: {question_type}")

        return formatters[question_type](item)


class RateLimiter:

    def __init__(self, calls_per_minute: int = CONFIG["API_RATE_LIMIT"]):
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0.0

    async def wait(self):
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        wait_time = max(0, self.min_interval - elapsed)

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        self.last_call_time = time.time()


class ModelClient:

    def __init__(self, model_name: str = "ollama/qwen2.5:14b-instruct-q8_0",
                 serve_provider: str = "ollama", temperature: float = 0.7, max_retries: int = 3):
        self.model_name = model_name
        self.serve_provider = serve_provider

        print(f"Temperature: {temperature}")

        with open(CONFIG["CONFIG_DIR"] / "autogen_ollama_models.json", "r") as f:
            OLLAMA_MODEL_CAPABILITIES = json.load(f)

        with open(CONFIG["CONFIG_DIR"] / "autogen_openai_models.json", "r") as f:
            OPENAI_MODEL_CAPABILITIES = json.load(f)

        with open(CONFIG["DATA_DIR"] / "autogen_ds_models.json", "r") as f:
            DS_MODEL_CAPABILITIES = json.load(f)

        print("=" * 100, f"\nServe Provider is: {serve_provider} and Model is: {model_name}\n", "=" * 100)

        if serve_provider == "ollama":
            self.client = OpenAIChatCompletionClient(
                model=model_name,
                model_capabilities=OLLAMA_MODEL_CAPABILITIES[model_name],
                api_key="RequiredButUnused",
                base_url="http://0.0.0.0:4000",
                max_retries=max_retries,
                temperature=temperature,
                timeout=1200,
            )
        elif serve_provider == "ds":
            self.client = OpenAIChatCompletionClient(
                model=model_name,
                model_capabilities=DS_MODEL_CAPABILITIES[model_name],
                api_key="your Key",
                base_url="your URL",
                max_retries=max_retries,
                temperature=temperature,
            )
        elif serve_provider == "openai":
            self.client = OpenAIChatCompletionClient(
                model=model_name,
                model_capabilities=OPENAI_MODEL_CAPABILITIES[model_name],
                api_key="your Key",
                base_url="your URL",
                max_retries=max_retries,
                temperature=temperature,
                timeout=600,
            )
        else:
            raise ValueError(f"Unsupported serve provider: {serve_provider}")

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def create(self, messages: List[Dict]) -> LLMMessage:
        try:
            return await self.client.create(messages)
        except Exception as e:
            print(f"Model call failed: {str(e)}")
            raise


class Logger:
    _instance = None   
    _initialized = False  
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 name: str = "app",
                 log_file: Optional[str] = None,
                 level: int = logging.INFO,
                 format: str = "%(asctime)s - %(name)s - %(levelname)s\n%(message)s") -> None:
        if self._initialized:
            return
            
        try:
            self.logger = logging.getLogger(name)
            self.logger.setLevel(level)
            
            if self.logger.handlers:
                self.logger.handlers.clear()
            
            formatter = logging.Formatter(format)
            
            if log_file:
                try:
                    log_dir = os.path.dirname(log_file)
                    if log_dir:
                        os.makedirs(log_dir, exist_ok=True)
                    
                    file_handler = logging.FileHandler(
                        filename=log_file,
                        encoding='utf-8'
                    )
                    file_handler.setFormatter(formatter)
                    self.logger.addHandler(file_handler)
                except Exception as e:
                    sys.stderr.write(f"Failed to create file handler.: {str(e)}\n")
                    self.logger.error(f"Failed to create file handler.: {str(e)}")
            
            self._initialized = True
            
        except Exception as e:
            sys.stderr.write(f"Failed to initialize logger: {str(e)}\n")
            raise
    
    def debug(self, message: str) -> None:
        try:
            self.logger.debug(self._ensure_string(message))
        except Exception as e:
            sys.stderr.write(f"Failed to log debug message: {str(e)}\n")
    
    def info(self, message: str) -> None:
        try:
            self.logger.info(self._ensure_string(message))
        except Exception as e:
            sys.stderr.write(f"Failed to log info message: {str(e)}\n")
    
    def warning(self, message: str) -> None:
        try:
            self.logger.warning(self._ensure_string(message))
        except Exception as e:
            sys.stderr.write(f"Failed to log warning message.: {str(e)}\n")
    
    def error(self, message: str) -> None:
        try:
            self.logger.error(self._ensure_string(message))
        except Exception as e:
            sys.stderr.write(f"Failed to log error message: {str(e)}\n")
    
    def critical(self, message: str) -> None:
        try:
            self.logger.critical(self._ensure_string(message))
        except Exception as e:
            sys.stderr.write(f"Failed to log critical message: {str(e)}\n")
    
    @staticmethod
    def _ensure_string(message: any) -> str:
        if not isinstance(message, str):
            return str(message)
        return message
    
    def close(self) -> None:
        try:
            for handler in self.logger.handlers:
                handler.close()
                self.logger.removeHandler(handler)
        except Exception as e:
            sys.stderr.write(f"Failed to close the log handler: {str(e)}\n")
    
    def __del__(self):
        self.close()


def extract_log_messages(log_file_path):
    messages = []
    current_message = []
    log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (.+) - INFO - (.+)'
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.match(log_pattern, line)
                
                if match:
                    if current_message:
                        messages.append('\n'.join(current_message))
                        current_message = []
                    
                    message = match.group(3).strip()
                    current_message.append(message)
                else:
                    if current_message:
                        current_message.append(line.strip())
            
            if current_message:
                messages.append('\n'.join(current_message))
    
    except FileNotFoundError:
        print(f"Error: File not found at {log_file_path}")
        return []
    except Exception as e:
        print(f"An error occurred while processing the file:{str(e)}")
        return []
    
    return messages


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
