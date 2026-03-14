

def init_state(symbol, date, cash, day_count, model):
    return {
        "symbol": symbol,
        "trade_date": date,
        "cash": cash,
        "model": model,
        "day_count": day_count,
        "market_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "industry_report": "",
        "momentum_report": "",
    }