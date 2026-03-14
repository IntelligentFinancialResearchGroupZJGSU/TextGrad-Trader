from pprint import pprint
import os

config = {
    "data_dir": "./data",
    "data_cache": "./data_cache",
    "result": "./result",
    "deep_think_llm": "deepseek-reasoner",
    "quick_think_llm": "deepseek-chat",
    "base_url": "https://api.deepseek.com",
    "api_key": ""
}


def get_config():
    _config = config.copy()
    if not _config.get("api_key"):
        _config["api_key"] = os.getenv("DEEPSEEK_API_KEY")
    return _config


os.makedirs(config["data_dir"], exist_ok=True)
os.makedirs(config["data_cache"], exist_ok=True)
os.makedirs(config["result"], exist_ok=True)




