from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from agent import *


class graph_workflow:
    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        router_memory,
        manager_memory,
        risk_memory,
        final_memory,
    ):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.router_memory = router_memory
        self.manager_memory = manager_memory
        self.risk_memory = risk_memory
        self.final_memory = final_memory

    def setup_graph(self):
        analyst_node = AnalystNode(self.quick_thinking_llm)
        market_analyst = analyst_node.create_market_analyst_node()
        news_analyst = analyst_node.create_news_analyst_node()
        fundamentals_analyst = analyst_node.create_fundamentals_analyst_node()
        industry_analyst = analyst_node.create_industry_analyst_node()
        momentum_analyst = analyst_node.create_momentum_analyst_node()

        router = create_router(self.quick_thinking_llm, self.router_memory)
        manager = create_research_manager(self.deep_thinking_llm, self.manager_memory)
        risk_manager = create_risk_manager(self.deep_thinking_llm, self.risk_memory)
        final_manager = create_final_manager(self.deep_thinking_llm, self.final_memory)

        memories = [self.router_memory, self.manager_memory, self.risk_memory, self.final_memory]
        reflection = create_memory_reflection(self.quick_thinking_llm, memories)
        weekly_reflection = create_weekly_reflection_node(self.quick_thinking_llm, memories)

        workflow = StateGraph(AgentState)
        workflow.add_node("market_analyst", market_analyst)
        workflow.add_node("news_analyst", news_analyst)
        workflow.add_node("fundamentals_analyst", fundamentals_analyst)
        workflow.add_node("industry_analyst", industry_analyst)
        workflow.add_node("momentum_analyst", momentum_analyst)
        workflow.add_node("router", router)

        workflow.add_node("manager", manager)
        workflow.add_node("risk_manager", risk_manager)
        workflow.add_node("final_decision", final_manager)
        workflow.add_node("environment_simulation", execution_simulation_node)
        workflow.add_node("reflector", reflection)
        workflow.add_node("weekly_reflector", weekly_reflection)

        analysts = ["market_analyst", "news_analyst", "fundamentals_analyst", "industry_analyst", "momentum_analyst"]
        for analyst in analysts:
            workflow.add_edge(START, analyst)

        for analyst in analysts:
            workflow.add_edge(analyst, "router")

        workflow.add_edge("router", "manager")
        workflow.add_edge("manager", "risk_manager")
        workflow.add_edge("risk_manager", "final_decision")
        workflow.add_edge("final_decision", "environment_simulation")
        workflow.add_edge("environment_simulation", "reflector")
        workflow.add_conditional_edges(
            "reflector",
            check_weekly_trigger,
            {
                "weekly_reflector": "weekly_reflector",
                END: END
            }
        )
        workflow.add_edge("weekly_reflector", END)
        return workflow.compile()


def check_weekly_trigger(state) -> str:
    day = state.get("day_count", 0)
    if day > 0 and day % 7 == 0:
        return "weekly_reflector"
    else:
        return END
