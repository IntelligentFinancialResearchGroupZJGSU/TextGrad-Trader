import os
from getpass import getpass


def _set_env(var: str):
    # 如果环境变量尚未设置，提示用户输入
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Enter your {var}: ")

    # 为我们将使用的服务设置API密钥


