import json
from pathlib import Path
import pandas as pd
from langchain_openai import ChatOpenAI
from graph.graph_workflow import graph_workflow
from agent import *
from graph.init_state import init_state


class train_model():
    def __init__(self,
                 config,
                 symbol,
                 model,
                 ):
        self.config = config
        self.symbol = symbol
        self.model = model
        self.deep_think_llm = ChatOpenAI(model=config["deep_think_llm"], base_url=config["base_url"], api_key=config["api_key"], temperature=1,
    max_tokens=4096)
        self.quick_think_llm = ChatOpenAI(model=config["quick_think_llm"], base_url=config["base_url"], api_key=config["api_key"], temperature=1,
    max_tokens=4096)
        self.router_memory = FinancialSituationMemory("router_memory", f"./data_cache/{self.symbol}/{self.model}/router_memory")
        self.manager_memory = FinancialSituationMemory("manager_memory", f"./data_cache/{self.symbol}/{self.model}/manager_memory")
        self.risk_memory = FinancialSituationMemory("risk_memory", f"./data_cache/{self.symbol}/{self.model}/risk_memory")
        self.final_memory = FinancialSituationMemory("final_memory", f"./data_cache/{self.symbol}/{self.model}/final_memory")
        self.graph_setup = graph_workflow(self.quick_think_llm, self.deep_think_llm, self.router_memory, self.manager_memory, self.risk_memory, self.final_memory)
        self.graph = self.graph_setup.setup_graph()
        self.log_states_dict = {}

    def run(self, date, cash, day_count):
        init_agent_state = init_state(self.symbol, date, cash, day_count, self.model)
        final_state = self.graph.invoke(init_agent_state)
        self._log_state(date, final_state)
        return final_state

    def _log_state(self, date, final_state):
        self.log_states_dict[str(date)] = {
            "company_of_interest": str(final_state.get("symbol", "")),
            "trade_date": str(final_state.get("trade_date", "")),
            "market_report": str(final_state.get("market_report", "")),
            "news_report": str(final_state.get("news_report", "")),
            "fundamentals_report": str(final_state.get("fundamentals_report", "")),
            "industry_report": str(final_state.get("industry_report", "")),
            "momentum_report": str(final_state.get("momentum_report", "")),
            "router_report": str(final_state.get("router", "")),
            "investment_plan": str(final_state.get("investment_plan", "")),
            "risk_plan": str(final_state.get("risk_plan", "")),
            "final_decision": str(final_state.get("final_decision", "")),

            "learned_beliefs": str(final_state.get("learned_beliefs", "")),
            "weekly_summary": str(final_state.get("weekly_summary", "")),
            "actual_outcome": str(final_state.get("actual_outcome", "")),
            "cash": float(final_state.get("cash", 0.0)),
        }
        directory = Path(f"eval_results/{self.symbol}/logs/{self.model}")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
                f"eval_results/{self.symbol}/logs/{self.model}/full_states_log_{date}.json",
                "w",
                encoding="utf-8",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4, ensure_ascii=False)






