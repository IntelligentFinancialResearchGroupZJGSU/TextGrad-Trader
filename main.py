import os
import shutil
from datetime import datetime
import pandas as pd
from graph.train import TextGradTraderRunner
from config import get_config
from environment import _set_env
from utils import setup_test_sandbox


def get_time_series(file):
    df = pd.read_csv(file)
    dates_list = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').tolist()
    return dates_list


if __name__ == '__main__':
    _set_env("DEEPSEEK_API_KEY")
    config = get_config()

    symbol = "000063"
    model = "test"  # train/test

    models = TextGradTraderRunner(config, symbol, model)
    file = f"data/technical_indicators/{symbol}.csv"
    date_list = get_time_series(file)

    # start_date = "2023-01-01"
    start_date = "2024-01-01"
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    # end_date = "2024-01-01"
    end_date = "2024-06-01"
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    cash = 10000.00
    day_count = 0

    # 读写锁卡住，需要单独执行
    # if model == "test":
    #     setup_test_sandbox(symbol)
    print(f"{symbol} began {model} form {start_date} to {end_date}")
    for date in date_list:
        date = datetime.strptime(date, "%Y-%m-%d").date()
        print(date)
        if date < start_date:
            continue
        if date >= end_date:
            break
        result = models.run(date, cash, day_count)
        cash = result["cash"]
        day_count += 1
    print("ALL Done")

