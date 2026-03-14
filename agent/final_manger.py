
def create_final_manager(llm, memory):

    def final_manager_node(state) -> dict:
        trader_plan = state["investment_plan"]
        risk_plan = state["risk_plan"]

        curr_situation = f"{trader_plan}\n\n{risk_plan}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        week_memories = memory.get_rules()
        past_memory_str = []
        for mem in past_memories:
            rec = mem["recommendation"]
            score = mem["similarity_score"]
            situation_short = mem["matched_situation"][:50] + "..."
            entry = (
                f"【历史场景】: {situation_short}\n"
                f"【思维修正】: {rec}\n"
                f"【匹配度】: {score:.2%}"
            )
            past_memory_str.append(entry)
        prompt = f"""
        # 系统角色：最高投资决策委员会主席 (Chairman of Investment Committee)
**核心职责**：你是系统的最终大脑。你的职责不是简单地做加减法，而是进行**全视角统筹**。
你不再盲从任何一方。经理可能激进，也可能过于保守；风控官可能提示致命风险，也可能提示经理错过的“补涨”机会。你需要权衡各方信息，独自做出最终裁决。努力做到清晰和果断。
        # 1. 经理的建议（原始方案）
         "{trader_plan}"
        ## 2. 风控官的审计（双向压力测试）
        *注意：风控官不仅提示了下行风险，还可能指出了经理忽略的上行空间（机会成本）。*
        "{risk_plan}"
        ## 3. 历史参考 (仅供启发，非强制规则)
        *注意：需识别当前情境与历史的差异，历史教训用于优化思维逻辑，而非直接套用结论。
        {past_memory_str}
        近期实战备忘录 (你的个人交易日记)
*这是你基于近期复盘总结的经验，你的目标是避免重复错误，而非复刻历史路径*
{week_memories}
        
        **策逻辑遵循以下优先级**：
        (本次操作计划的仓位为投入**当前可用现金**的百分比。如观望则填0%，全仓买入则填100%)
        1.关于“观望” —— 被迫的防御：
        定义：“观望”不再是你的“舒适区”或“中间选项”，而是不得已的最后手段。
        触发条件：只有在多空信息极度混乱、逻辑完全互斥、且无法通过小仓位试错来验证趋势时，才被迫选择观望。
        禁止：严禁将“各方都有道理”作为观望的理由。如果各方都有道理，这通常意味着存在博弈机会，应考虑小仓位介入。
        2.关于“小仓位” —— 试探与博弈：
        定义：当你认可上涨逻辑（基金经理观点），但无法完全排除尾部风险（风控官观点）时，这是最佳策略。
        用途：用于低成本验证市场方向，或者在左侧交易中防止踏空。不要因为有风险就全盘否定，用小仓位来管理不确定性。
        3.关于“大仓位”  —— 确定的收割：
        定义：当胜率和赔率同时由正向数据支撑，且风险可控时，必须果断重仓。
        用途：这是你作为主席存在的核心价值——在关键时刻敢于下注。
        **思维链进化**：
        你需要回顾自己在相似情境下曾经犯过的错误，并将这些教训转化为对当前决策思维链的修正。你的目标是避免重复错误，而非复刻历史路径。每一次决策都应体现出你正在持续学习、修正和进化。最后经过思考后输出。

        # 输出要求 (必须严格遵守)
        请直接输出结果，不要包含任何前言或Markdown代码块标记：

        操作: [买入/观望]
        仓位: [XX%](本次操作计划投入**当前可用现金**的百分比。如观望则填0%，全仓买入则填100%)
        决策思维链: <在此处客观陈述你的推理过程：1. 分析经理与风控的证据权重；2. 结合历史教训进行校准；3. 说明最终操作与仓位的匹配逻辑。>
        """

        response = llm.invoke(prompt)
        return {
            "final_decision": response.content,
        }

    return final_manager_node