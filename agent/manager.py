import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        market_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        industry_report = state["industry_report"]
        momentum_report = state["momentum_report"]

        router_report = state.get("router", "{}")

        curr_situation = f"{market_report}\n\n{news_report}\n\n{fundamentals_report}\n\n{industry_report}\n\n{momentum_report}"
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
你是由 MCTS (蒙特卡洛树搜索) 驱动的策略博弈核心。
**你的核心任务不再是预测未来，而是针对“不确定性”进行“情景-应对”推演。**
你必须评估不同仓位策略在面对未来多种可能市场走势时的**鲁棒性 (Robustness)** 和 **盈亏不对称性 (Asymmetry)**。请忽略账户层面的资金硬性约束（如总回撤、保证金限制）。你需要做的是根据机会的完美程度，给出“意愿仓位”建议。**

1.以下为当前市场环境与各分析师的判断结果。你应将 router 给出的分析师权重视为先验关注度，用于指导信息读取的侧重点，当个别分析师权重为0或者很低时，应该慎重考虑该分析师意见。但不得将其视为不可调整的结论性分配。
你不需要寻求所有分析师的“共识”。你的任务是寻找**“最强单一信号”**。如果单个分析师给出了极高的置信度（>=0.7）的买入建议，即使 多数分析师觉得一般，你也可以发起进攻。
{router_report}和{curr_situation} 

近期实战备忘录 (你的个人交易日记)
*这是你基于近期复盘总结的经验，你的目标是避免重复错误，而非复刻历史路径*
{week_memories}

# Thinking Process (MCTS 模拟与推演)
请在内心进行以下三个步骤的推演（不要直接输出过程，融入到最终的“决策及思维链”中）：

## Thinking Process (MCTS 情景推演)
请在内心执行以下三步推演（融入最终输出，无需分段显示）：

## Step 1: 构建策略分支 (Strategy Branches)
建立三个互斥的持仓假设：
1.  **分支A (重仓进攻 >70%)**：假设当前是胜率极高的主升浪起点。
2.  **分支B (试错博弈 20%-50%)**：假设当前存在值得博弈的预期差，但需验证。
3.  **分支C (空仓/极轻仓防御)**：假设当前风险大于机会，或机会成本过高。

## Step 2: 压力测试与情景模拟 (Scenario Stress Test)
**这是最关键的一步。不要预测单一结果，而是针对每个分支，模拟以下三种情景的后果：**
*   **情景 X (下行/风险触发)**：如果技术破位或利空发酵，该分支的撤退成本是多少？是否会造成不可挽回的本金磨损？(止损明确性)
*   **情景 Y (震荡/磨损)**：如果市场在该位置横盘震荡，该分支的时间成本和心态损耗如何？
*   **情景 Z (上行/爆发)**：如果行情突然启动，该分支能否提供足够的收益弹性？(踏空痛感 vs 持仓收益)

**计算“不对称性 (Asymmetry)”**：
*   评估每个分支的 **[潜在最大亏损] vs [潜在预期收益]**。

## Step 3: 决策选择 (Selection & Calibration)
*   结合 {past_memory_str} 中的历史相似案例进行校准。
*   **否决权机制**：如果某个分支在“情景 X”下的后果不可控（如止损模糊、回撤过大），直接否决该分支。
*   选择那个**“即便判断错误，后果也可控；一旦判断正确，收益最大化”**的分支。

# Output Format (严格遵守)
请直接输出以下格式，不要包含Markdown标题或其他废话：
操作: [买入/观望]
仓位: [XX%](本次操作计划投入**当前可用现金**的百分比。如观望则填0%，全仓买入则填100%)
决策及思维链: <这里详细描述你的MCTS推演过程。首先简述三个分支的模拟情况，重点说明为何放弃了另外两个分支。最后阐述选中方案的思维链。>
"""
        response = llm.invoke(prompt)
        return {
            "investment_plan": response.content,
        }

    return research_manager_node
