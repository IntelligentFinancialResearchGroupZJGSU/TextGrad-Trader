import os
import shutil
import time


if __name__ == '__main__':
    symbol = "000063"
    train_dir = f"./data_cache/{symbol}/train"
    test_dir = f"./data_cache/{symbol}/test"
    if os.path.exists(test_dir):
        try:
            time.sleep(0.5)
            shutil.rmtree(test_dir)
        except OSError as e:
            print(f"警告：清理失败,错误: {e}")
    shutil.copytree(train_dir, test_dir)


