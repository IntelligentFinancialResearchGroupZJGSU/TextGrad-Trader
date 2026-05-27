import time
import json


def create_bayesian_contextual_gating_router(llm, memory):
    def router_node(state) -> dict:
        market_report = state.get("market_report")
        news_report = state.get("news_report")
        fundamentals_report = state.get("fundamentals_report")
        industry_report = state.get("industry_report")
        momentum_report = state.get("momentum_report", "")

        curr_situation = (
            f"【市场分析】{market_report}\n"
            f"【新闻分析】{news_report}\n"
            f"【基本面】{fundamentals_report}\n"
            f"【行业分析】{industry_report}\n"
            f"【动量分析】{momentum_report}"
        )

        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = []
        for mem in past_memories:
            rec = mem["recommendation"]
            score = mem["similarity_score"]
            situation_short = mem["matched_situation"][:50] + "..."
            entry = (
                f"【历史场景】: {situation_short}\n"
                f"【参考权重】: {rec}\n"
                f"【匹配度】: {score:.2%}"
            )
            past_memory_str.append(entry)


        prompt = f"""
        # 你是一个权重计算器。根据 5 位分析师的报告和当前市场状态，计算每个分析师的**可信度权重**。
        # 输入数据
        ## 1. 当前分析师报告
        {curr_situation}

        ## 2. 历史参考资料 (仅供启发，非强制规则)
        *注意：相似历史情境与权重仅作为先验参考，而非直接结论。即使历史经验表明某类分析师在类似环境中权重较高，若其在当前情境下的逻辑不自洽、信号分散或证据薄弱，你必须主动下调其影响力。
        请立足于当下市场的独特结构与信号强度，动态调整各分析师权重，而非照搬历史配置。

        以下为相似历史情境、当时的分析师权重分布，以及它们与当前环境的匹配程度：
        {past_memory_str}

        # 计算逻辑
        1. **评估信号强度**：谁的逻辑最严密？谁的信号最强烈（Strong Signal）？
        2. **处理冲突**：若技术面与消息面冲突，结合历史倾向和当前主导因素（如突发新闻主导 vs 震荡市技术主导）进行裁决。
        3. **动态归一化**：确保输出的权重能反映你对该分析师的信任程度（总和为 1.0）。

        # 输出格式 (STRICT JSON ONLY)
        * 严禁输出 Markdown 代码块 (no ```json)。
        * 严禁输出任何解释性文字或理由。
        * 直接返回一个 JSON 对象。

        格式模版：
        {{
            "market_analyst": 0.x,
            "news_analyst": 0.x,
            "fundamentals_analyst": 0.x,
            "industry_analyst": 0.x,
            "momentum_analyst": 0.x
        }}
        """
        response = llm.invoke(prompt)
        return {
            "router": response.content,
        }

    return router_node


create_router = create_bayesian_contextual_gating_router
