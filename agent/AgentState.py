from langgraph.graph import MessagesState
from typing import TypedDict


class AgentState(MessagesState):
    symbol: str  # 股票代码
    trade_date: str  # 日期
    sender: str  # 追踪哪个智能体最后修改了状态
    model: str   # train/test

    # 分析师将报告字段
    market_report: str
    news_report: str
    fundamentals_report: str
    industry_report: str
    momentum_report: str

    # 投资计划
    investment_plan: str  # 经理的计划
    risk_plan: str  # 风险交易员的可执行计划
    final_decision: str  # 最终决定

    # 更新router和prompt
    router: str
    learned_beliefs: str
    weekly_summary: str
    # 结果
    actual_outcome: str
    cash: float
    day_count: int


