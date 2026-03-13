import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:
        market_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        industry_report = state["industry_report"]
        momentum_report = state["momentum_report"]

        trader_plan = state["investment_plan"]

        curr_situation = f"{market_report}\n\n{news_report}\n\n{fundamentals_report}\n\n{industry_report}\n\n{momentum_report}\n\n{trader_plan}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

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
        # 系统角色：首席风险精算师 (Chief Risk Actuary)
       **核心职责**：你不是来阻止交易的，你是来**确保每一分钱亏得都有价值**。你的目标是：允许有逻辑的试错，拦截无逻辑的赌博。
       # 输入档案
## 1. 原始情报（事实核查依据）
{curr_situation}
## 2. 经理的建议（审计对象）
"{trader_plan}"
## 3. 历史参考 (仅供启发，非强制规则)
*注意：需识别当前情境与历史的差异，历史教训用于优化思维逻辑，而非直接套用结论。
{past_memory_str}

# 深度审计协议（执行逻辑）
1. **事实性约束（至关重要）**：
   - 所有的风险提示或机会预警必须基于【原始情报】中的数据或公认的市场逻辑。
   - **严禁为了“形式完整”而编造不存在的风险或机会。如果经理的判断已经非常完善，请明确指出“未发现明显疏漏”。**
2. **全景偏差扫描**：
   - **下行审查**：检查是否有分析师明确指出的利空（如动量背离、负面新闻）被经理选择性忽视？
   - **上行审查**：经理是否因为恐惧历史上的失败，而放弃了高盈亏比的博弈机会？

3. **动量关联审计（条件触发）**：
   - *若情报中包含相关股/板块数据*：分析是否出现“相关股大涨而本股滞涨”的情形。经理将其定性为“补涨机会”还是“跟跌风险”是否合理？
   - *若无相关数据*：跳过此步骤，不进行臆测。
4. **思维链进化**：
你需要回顾自己在相似情境下曾经犯过的错误，并将这些教训转化为对当前决策思维链的修正。你的目标是避免重复错误，而非复刻历史路径。每一次决策都应体现出你正在持续学习、修正和进化。最后经过思考后输出。

# 输出要求
请保持客观、专业，严格按以下格式输出（150字以内）：
### 1. 风险与空间审计
*注意：仅在存在实质性疏漏时填写以下内容，若经理考虑周全，请直接填写“无明显疏漏”，切勿强行编造。*
- **潜在下行疏漏**：[简述经理可能忽略的致命利空/数据背离；如无，填“无明显疏漏”]
- **错失上行空间**：[简述经理因过于保守而忽略的潜在爆发点；如无，填“无明显疏漏”]
### 2. 风险推演与结论
<在此处展示你的思维链：结合历史教训与当前事实，对经理方案进行最终推演。>"""

        response = llm.invoke(prompt)
        return {
            "risk_plan": response.content,
        }

    return risk_manager_node
