import pandas as pd
import re


def execution_simulation_node(state) -> dict:
    decision_text = state.get("final_decision")
    current_date_str = state.get("trade_date")
    symbol = state.get("symbol")
    cash = state.get("cash")
    # 正则提取操作和仓位
    action_match = re.search(r"操作[：:]\s*\[?(买入|观望)\]?", decision_text)
    action = action_match.group(1) if action_match else "观望"
    pos_match = re.search(r"仓位[：:]\s*\[?(\d+(?:\.\d+)?)%?\]?", decision_text)
    pos_pct = float(pos_match.group(1))if pos_match else 0.0
    pos_pct = max(0.0, min(100.0, pos_pct))
    position_ratio = pos_pct / 100.0
    # 获取真实涨跌幅
    pct_change = get_price_change_from_csv(current_date_str, symbol)
    if action == "买入":
        cash += cash * position_ratio * (pct_change / 100.0)

    outcome_description = ""
    if action == "买入":
        if pct_change > 0:
            if pct_change > 3.0:  # 大涨
                if pos_pct > 50:
                    outcome_description = f"【大获全胜】重仓({pos_pct}%)出击抓住了 {pct_change:.2f}% 的大涨。决策果断，收益显著。"
                elif pos_pct < 20:
                    outcome_description = f"【遗憾盈利】虽然看涨正确（涨 {pct_change:.2f}%），但仓位过低({pos_pct}%)，导致资金效率低下，浪费了绝佳机会。"
                else:
                    outcome_description = f"【稳健获利】仓位适中({pos_pct}%)，吃到 {pct_change:.2f}% 的涨幅。策略执行符合预期。"
            else:
                outcome_description = f"【小幅获利】买入后上涨 {pct_change:.2f}%，仓位 {pos_pct}%。属正常套利。"

        elif pct_change < 0:
            if pct_change < -3.0:
                if pos_pct > 60:
                    outcome_description = f"【重大决策失误】在高风险下重仓({pos_pct}%)买入，遭遇 {pct_change:.2f}% 暴跌。这是致命的激进错误。"
                else:
                    outcome_description = f"【风控幸存】虽然方向判断错误（跌 {pct_change:.2f}%），但好在仓位较低({pos_pct}%)，避免了伤筋动骨。"
            else:  # 微跌
                outcome_description = f"【试错成本】买入后小幅回调 {pct_change:.2f}%，属于正常的博弈成本。"
        else:
            outcome_description = f"【资金空转】买入后平盘，扣除成本后可能微亏。仓位 {pos_pct}%。"

    elif action == "观望":
        # 观望时，position_ratio 视为 0，我们看“如果买入会发生什么”来评价决策
        if pct_change < 0:
            if pct_change < -2.0:
                outcome_description = f"【神级防御】空仓观望，成功躲避了 {pct_change:.2f}% 的大跌。不亏就是大赚。"
            else:
                outcome_description = f"【成功规避】选择观望，规避了 {pct_change:.2f}% 的下跌风险。"

        elif pct_change > 0:
            if pct_change > 3.0:
                outcome_description = f"【严重踏空】由于过度保守选择观望，完美错过了 {pct_change:.2f}% 的暴涨。应反思为何对高确定性机会视而不见。"
            elif pct_change > 1.0:
                outcome_description = f"【错失机会】市场上涨 {pct_change:.2f}%，但选择了观望。过于谨慎导致机会成本增加。"
            else:
                outcome_description = f"【合理观望】市场微涨 {pct_change:.2f}%，鱼尾行情不吃也罢。决策合理。"
        else:
            outcome_description = "【无效波动】市场平盘，观望是正确的节省精力的方式。"

    return {
        "actual_outcome": outcome_description,
        "cash": cash,
    }


def get_price_change_from_csv(current_date_str, symbol):
    df_stock = pd.read_csv(f"./result/{symbol}.csv")
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    current_date = pd.to_datetime(current_date_str).date()
    pct_change = df_stock[df_stock['date'].dt.date == current_date]['pct_change']
    return pct_change.item()
