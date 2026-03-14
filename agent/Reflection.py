from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_memory_reflection(llm, memories):
    """
    创建反思节点：根据 Execution Node 的计算结果，对三个角色的决策进行复盘。
    """

    def memory_reflection_node(state) -> dict:
        curr_situation = "\n\n".join([
            f"【市场报告】{state.get('market_report', '')}",
            f"【新闻报告】{state.get('news_report', '')}",
            f"【基本面报告】{state.get('fundamentals_report', '')}",
            f"【行业报告】{state.get('industry_report', '')}",
            f"【动量报告】{state.get('momentum_report', '')}",
        ])

        router = state.get("router")
        trader_plan = state.get("investment_plan")
        risk_plan = state.get("risk_plan")
        final_decision = state.get("final_decision")

        actual_outcome = state.get("actual_outcome")

        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """# 系统角色：动态权重优化引擎
        你的任务是根据【实际交易结果】对五个分析师的预测准确度进行回测，并计算出下一轮的信任权重 (Weights)**。

        # 输入数据
        ## 1. 实际发生的市场结果
        {actual_outcome}

        ## 2. 各分析师在交易前的报告以及之前生成的路由权重。
        {curr_situation}
        {router}

        # 评分逻辑
        1. **准确度奖励**：谁的报告准确预判了结果（包括涨跌方向和幅度），谁的权重就应该显著增加。
        2. **误导惩罚**：谁给出了相反的建议或忽略了致命风险，谁的权重应大幅降低。
        3. **噪音过滤**：如果没有相关信息（报告为空或废话），保持低权重或中性。
        4. **归一化约束**：所有分析师的权重之和必须严格等于 1.0。

        # 输出要求 (Strict JSON)
        请仅输出一个 JSON 对象，键名必须与下方模板完全一致，不要包含 Markdown 代码块或其他文字。
        格式模版：
        {{
            "market_analyst": 0.x,
            "news_analyst": 0.x,
            "fundamentals_analyst": 0.x,
            "industry_analyst": 0.x,
            "momentum_analyst": 0.x
        }}
        """),
        ])

        router_chain = router_prompt | llm | StrOutputParser()
        router_reflection = router_chain.invoke({
            "actual_outcome": actual_outcome,
            "curr_situation": curr_situation,
            "router": router,
        })
        router_memory = memories[0]
        memory_router = f"Situation Summary:\n{curr_situation}"
        router_memory.add_situations([(memory_router, router_reflection)])

        role_prompt = ChatPromptTemplate.from_messages([
            ("system", """# 系统角色：客观复盘审计官
        你的任务是对角色的思维链进行**实事求是**的审计。
        **原则**：严禁为了批评而批评。如果思维链逻辑是正确的，请给予肯定；只有在确实存在逻辑漏洞时才进行修正。

        # 输入数据
        1. **当时的输入情报**：
        {curr_situation}

        2. **角色的原始思维链与决策**：
        角色身份：{role_name}
        方案内容：{decision_plan}

        3. **实际结果 (Ground Truth)**：
        {actual_outcome}

        # 任务指令 (三选一逻辑)
        请对比【思维链】与【实际结果】，判断属于以下哪种情况，并按对应格式输出：
        ## 情况 1：思维链存在实质性错误 (逻辑漏洞导致亏损/踏空)
        *适用场景：决策错误，或者理由完全编造。*
        - **[LOOPHOLE]**：摘录原方案中错误的具体那句话。
        - **[CORRECTION]**：修正后的正确思维逻辑（重写这句话）。
        - **[LESSON]**：提取一条“禁止干什么”的教训。

        ## 情况 2：思维链大体正确，但有优化空间 (方向对但细节不足)
        *适用场景：虽然赚钱了但没赚够，或者虽然避险了但过于恐慌。*
        - **[POINT]**：指出哪里做得不够完美。
        - **[IMPROVEMENT]**：提出具体的改进建议（如：仓位应该更重一点，或止损应该更宽一点）。

        ## 情况 3：思维链完全正确 (精准预判/完美执行)
        *适用场景：决策逻辑严密，且结果符合预期。*
        - **[WINNING_POINT]**：指出思维链中决定胜负的那个关键判断。
        - **[SUGGESTION]**：将这个成功案例提炼为一条通用的操作建议。

        # 输出格式 (Strict Format)
        请仅输出选中的那一种情况对应的标签内容，不要输出多余的解释。
        """),
        ])

        roles_memories = memories[1:]
        chain = role_prompt | llm | StrOutputParser()
        plans = [trader_plan, risk_plan, final_decision]
        roles = ["经理 (Manager)", "风控 (Risk Manager)", "最终决策者 (Final Decision Maker)"]
        all_new_beliefs = []
        for memory, plan, role in zip(roles_memories, plans, roles):
            if not plan: continue
            reflection_result = chain.invoke({
                "curr_situation": curr_situation,
                "role_name": role,
                "decision_plan": plan,
                "actual_outcome": actual_outcome
            })
            memory_key = f"Situation Summary:\n{curr_situation}"
            memory.add_situations([(memory_key, reflection_result)])
            all_new_beliefs.append(reflection_result)
        return {
            "learned_beliefs": "\n".join(all_new_beliefs) if all_new_beliefs else "本次无显著信念更新。"
        }

    return memory_reflection_node