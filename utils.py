import json
from typing import Annotated
import pandas as pd
import numpy as np
import joblib
import os
import torch
import akshare as ak
from datetime import datetime
from langchain.tools import tool
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import shutil
import time
import sys

DATA_DIR = "./data"


class Toolkit:
    def __init__(self, config):
        self.config = config
        self.get_technical_indicators = get_technical_indicators
        self.get_news = get_news
        self.get_balance_sheet = get_balance_sheet
        self.get_Cash_Flow_Statement = get_Cash_Flow_Statement
        self.get_Income_Statement = get_Income_Statement
        self.get_industry_news = get_industry_news
        self.get_industry_indicator = get_industry_indicator
        self.get_momentum_data = get_akshare_data


@tool
def get_technical_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "end date in yyyy-mm-dd format"],
) -> str:
    """
    This tool extracts technical_indicators within a specified date range from a CSV file containing technical_indicators data.

    return str
    """
    # read in data file
    file_path = os.path.join(
            DATA_DIR,
            f"technical_indicators/{symbol}.csv",
        )
    print(file_path)
    data = pd.read_csv(file_path, parse_dates=["date"])
    # Convert string type dates to date objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["date"] >= pd.to_datetime(start_date)) & (data["date"] <= pd.to_datetime(end_date))
        ]
    #
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None):
        df_string = filtered_data.to_string(index=False)

    return f"## Stock Data for {symbol} from {start_date} to {end_date}:\n\n{df_string}"


@tool
def get_news(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "end date in yyyy-mm-dd format"],
) -> str:
    """
    This tool extracts stock news within a specified date range from a CSV file containing stock news.

    return str
    """
    # read in data file
    file_path = os.path.join(
            DATA_DIR,
            f"news/{symbol}.csv",
        )

    data = pd.read_csv(file_path, parse_dates=["Publish Time"])
    # Convert string type dates to date objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["Publish Time"] >= pd.to_datetime(start_date)) & (data["Publish Time"] <= pd.to_datetime(end_date))
        ]
    content_data = filtered_data[['Publish Time', 'Title']]
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None):
        df_string = content_data.to_string(index=False)

    return f"## Stock news for {symbol} from {start_date} to {end_date}:\n\n{df_string}"


@tool
def get_balance_sheet(
    symbol: Annotated[str, "ticker symbol of the company"],
    frequency: Annotated[str, "reporting frequency of the company's financial history: semi-annual / quarterly"],
    start_date: Annotated[str, "The time currently being analyzed in yyyy-mm-dd format"],
) -> str:
    """
    This tool extracts quarterly range company financial balance_sheet report data from CSV files containing company financial balance_sheet report data.
    return str
    """
    file_path = os.path.join(
        DATA_DIR,
        f"fundamentals/Balance Sheet/{symbol}.csv",
    )

    data = pd.read_csv(file_path, parse_dates=["统计截止日期"])
    # Convert string type dates to date objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["统计截止日期"] <= pd.to_datetime(start_date))
        ]
    if filtered_data.empty:
        print("No balance sheet available before the given current date.")
        return ""
    latest_balance_sheet = filtered_data.loc[filtered_data["统计截止日期"].idxmax()]
    return (
        f"## {frequency} balance sheet for {symbol} released on {str(latest_balance_sheet['统计截止日期'])[0:10]}: \n"
        + str(latest_balance_sheet)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of assets, liabilities, and equity. Assets are grouped as current (liquid items like cash and receivables) and noncurrent (long-term investments and property). Liabilities are split between short-term obligations and long-term debts, while equity reflects shareholder funds such as paid-in capital and retained earnings. Together, these components ensure that total assets equal the sum of liabilities and equity."
    )


@tool
def get_Cash_Flow_Statement(
    symbol: Annotated[str, "ticker symbol of the company"],
    frequency: Annotated[str, "reporting frequency of the company's financial history: semi-annual / quarterly"],
    start_date: Annotated[str, "The time currently being analyzed in yyyy-mm-dd format"],
) -> str:
    """
    This tool extracts semi-annual range company financial Cash_Flow_Statement report data from CSV files containing company financial Cash_Flow_Statement report data.
    return str
    """
    file_path = os.path.join(
        DATA_DIR,
        f"fundamentals/Cash Flow Statement/{symbol}.csv",
    )

    data = pd.read_csv(file_path, parse_dates=["统计截止日期"])
    # Convert string type dates to date objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["统计截止日期"] <= pd.to_datetime(start_date))
        ]
    if filtered_data.empty:
        print("No cash flow statement available before the given current date.")
        return ""
    latest_cash_flow = filtered_data.loc[filtered_data["统计截止日期"].idxmax()]
    return (
            f"## {frequency} cash flow statement for {symbol} released on {str(latest_cash_flow['统计截止日期'])[0:10]}: \n"
            + str(latest_cash_flow)
            + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of cash movements. Operating activities show cash generated from core business operations, including net income adjustments for non-cash items and working capital changes. Investing activities cover asset acquisitions/disposals and investments. Financing activities include debt transactions, equity issuances/repurchases, and dividend payments. The net change in cash represents the overall increase or decrease in the company's cash position during the reporting period."
    )


@tool
def get_Income_Statement(
    symbol: Annotated[str, "ticker symbol of the company"],
    frequency: Annotated[str, "reporting frequency of the company's financial history: semi-annual / quarterly"],
    start_date: Annotated[str, "The time currently being analyzed in yyyy-mm-dd format"],
) -> str:
    """
    This tool extracts quarterly range company financial Income_Statement report data from CSV files containing company financial Income_Statement report data.
    return str
    """
    file_path = os.path.join(
        DATA_DIR,
        f"fundamentals/Income Statement/{symbol}.csv",
    )

    data = pd.read_csv(file_path, parse_dates=["统计截止日期"])
    # Convert string type dates to date objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["统计截止日期"] <= pd.to_datetime(start_date))
        ]
    if filtered_data.empty:
        print("No income statement available before the given current date.")
        return ""
    latest_income = filtered_data.loc[filtered_data["统计截止日期"].idxmax()]
    return (
            f"## {frequency} income statement for {symbol} released on {str(latest_income['统计截止日期'])[0:10]}: \n"
            + str(latest_income)
            + "\n\nThis includes metadata like reporting dates and currency, share details, and a comprehensive breakdown of the company's financial performance. Starting with Revenue, it shows Cost of Revenue and resulting Gross Profit. Operating Expenses are detailed, including SG&A, R&D, and Depreciation. The statement then shows Operating Income, followed by non-operating items and Interest Expense, leading to Pretax Income. After accounting for Income Tax and any Extraordinary items, it concludes with Net Income, representing the company's bottom-line profit or loss for the period."
    )


@tool
def get_industry_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "end date in yyyy-mm-dd format"],
) -> str:
    """
    This tool extracts data from industry status reports from CSV files containing Fama5 factors.

    return str
    """
    file_path = os.path.join(
        DATA_DIR,
        f"industry_indicator/{symbol}.csv",
    )

    data = pd.read_csv(file_path, parse_dates=["交易日期"])
    # Convert string type dates to date objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["交易日期"] >= pd.to_datetime(start_date)) & (data["交易日期"] <= pd.to_datetime(end_date))
        ]
    #

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None):
        df_string = filtered_data.to_string(index=False)

    return f"## Stock Data for {symbol} from {start_date} to {end_date}:\n\n{df_string}"


@tool
def get_industry_news(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "end date in yyyy-mm-dd format"],
) -> str:
    """
    This tool extracts industry news data from CSV files.

    return str
    """
    file_path = os.path.join(
        DATA_DIR,
        f"industry_news/{symbol}.csv",
    )

    data = pd.read_csv(file_path, parse_dates=["date"])
    # Convert string type dates to date objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["date"] >= pd.to_datetime(start_date)) & (data["date"] <= pd.to_datetime(end_date))
        ]
    #
    content_data = filtered_data[['summary']]

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None):
        df_string = content_data.to_string(index=False)
    return f"## Stock Data for {symbol} from {start_date} to {end_date}:\n\n{df_string}"


@tool
def get_akshare_data(
        symbols: Annotated[str, "Stock code collection, e.g. '000001,600036'"],
        start_date: Annotated[str, "start date in yyyyMMdd format"],
        end_date: Annotated[str, "end date in yyyyMMdd format"],
) -> str:
    """
    获取多只股票的历史行情数据，用于动量分析。
    返回格式包含：date, open, close, high, low, amount。
    """
    # 1. 处理输入字符串为列表
    if isinstance(symbols, str):
        symbol_list = [s.strip() for s in symbols.split(',')]
    else:
        symbol_list = symbols
    report_list = []
    for symbol in symbol_list:
        try:
            # 识别市场前缀
            clean_symbol = ''.join(filter(str.isdigit, symbol))
            if clean_symbol.startswith('6'):
                display_symbol = f"sh{clean_symbol}"
            else:
                display_symbol = f"sz{clean_symbol}"
            # 获取数据 (前复权 qfq)
            df = ak.stock_zh_a_hist_tx(display_symbol, start_date=start_date, end_date=end_date, adjust="qfq", timeout=5)
            if df.empty:
                continue
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            table_md = df.to_markdown(index=False)
            report_content = f"### Stock: {display_symbol}\n{table_md}"
            report_list.append(report_content)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue
    if not report_list:
        return "No data found."
    return "\n\n".join(report_list)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=dropout)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )

    def forward(self, x):
        # x: (B, seq_len, F)
        out, h = self.gru(x)  # out: (B, seq_len, hidden)
        last = out[:, -1, :]  # (B, hidden)
        return self.out(last)  # (B, 1)


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # we'll transpose to (B, d_model, seq_len) then pool
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, x):
        # x: (B, seq_len, F)
        x = self.input_proj(x)  # (B, seq_len, d_model)
        x = self.transformer(x)  # (B, seq_len, d_model)
        # pool over seq_len
        x_t = x.permute(0, 2, 1)  # (B, d_model, seq_len)
        pooled = self.pool(x_t).squeeze(-1)  # (B, d_model)
        return self.head(pooled)  # (B,1)


@tool
def get_predication(
    symbol: Annotated[str, "ticker symbol of the company"],
    end_date: Annotated[str, "The date for which the prediction is based on in yyyy-mm-dd format"],
    look_window: Annotated[int, "The number of look-back trading days required for prediction"],
) -> str:
    """
    This tool uses a trained model to predict the closing price for the 8th day in the future (i.e. the second day after 7 days) from a dataset containing 7-day historical data of stocks.
    return str
    """
    file_path = os.path.join(DATA_DIR, f"technical_indicators/{symbol}.csv")
    try:
        data = pd.read_csv(file_path, parse_dates=["date"])
    except FileNotFoundError:
        return f"Error: Data file not found for symbol {symbol} at {file_path}"

    # 排序
    data = data.sort_values("date").reset_index(drop=True)

    # 检查日期
    end_date_dt = pd.to_datetime(end_date)
    if end_date_dt not in data["date"].values:
        return f"Error: End date {end_date} not found in dataset."

    end_index = data[data["date"] == end_date_dt].index[0]
    start_index = end_index - look_window + 1
    if start_index < 0:
        return f"Error: Not enough history for {look_window} lookback days."


    df = data.iloc[start_index: end_index + 1].copy()
    if len(df) != look_window:
        return f"Internal error: extracted {len(df)} days but expected {look_window}"


    feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 'amount', 'mkt_cap',
        'macd', 'macds', 'macdh', 'kdjk', 'kdjd', 'kdjj',
        'rsi_6', 'rsi_12', 'close_5_sma', 'close_10_sma',
        'boll', 'boll_ub', 'boll_lb'
    ]
    features = df[feature_cols].copy()
    # 去除 NaN / inf
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(method="ffill", inplace=True)
    features.fillna(method="bfill", inplace=True)
    if features.isna().sum().sum() > 0:
        return "Error: Data still contains NaN after cleaning."


    scaler_path = r".\data\model\scaler.joblib    "
    feature_scaler = joblib.load(scaler_path)
    scaled_features = feature_scaler.transform(features)

    X_input = scaled_features.reshape(1, look_window, len(feature_cols))
    X_tensor = torch.tensor(X_input, dtype=torch.float32)


    gru_path = r".\data\model\gru.pth"
    transformer_path = r".\data\model\transformer.pth"

    gru_model = GRUModel(input_size=len(feature_cols))
    transformer_model = TransformerModel(input_size=len(feature_cols))

    gru_model.load_state_dict(torch.load(gru_path, map_location="cpu"))
    transformer_model.load_state_dict(torch.load(transformer_path, map_location="cpu"))

    gru_model.eval()
    transformer_model.eval()


    with torch.no_grad():
        gru_pred = gru_model(X_tensor).numpy()[0, 0]
        trans_pred = transformer_model(X_tensor).numpy()[0, 0]

    next_day = data.iloc[end_index + 1]["date"].strftime("%Y-%m-%d")

    return (
        f"Prediction for {symbol} using {look_window} days up to {end_date}:\n"
        f"Next day ({next_day}) predicted daily return:\n"
        f"  - GRU: {gru_pred:.4f}\n"
        f"  - Transformer: {trans_pred:.4f}"
    )


# 同一个main调用会被进程加锁，调用后test.
def setup_test_sandbox(symbol: str, base_cache_dir: str = "./data_cache") -> str:
    project_root = os.path.join(base_cache_dir, symbol)
    train_dir = os.path.join(project_root, "train")
    test_dir = os.path.join(project_root, "test")
    if not os.path.exists(train_dir):
        error_msg = (
            f"路径不存在: {train_dir}\n, 测试模式需要继承训练经验，请先运行训练模式")
        raise FileNotFoundError(error_msg)

    if os.path.exists(test_dir):
        try:
            time.sleep(0.5)
            shutil.rmtree(test_dir)
        except OSError as e:
            print(f"警告：清理失败,错误: {e}")

    try:
        shutil.copytree(train_dir, test_dir)
    except Exception as e:
        raise RuntimeError(f"记忆克隆失败: {e}")
    return test_dir