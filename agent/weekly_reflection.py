from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_weekly_reflection_node(llm, memories):

    target_memories = memories[1:]
    roles = ["经理 (Manager)", "风控 (Risk Manager)", "最终决策者 (Chairman)"]

    def weekly_node(state):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """# 系统角色：策略复盘助理
    你正在整理【{role_name}】过去一周的交易日志。
     # 输入 (碎片化反思)
    {daily_logs}
    # 任务
    请从碎片化的反思中，提炼出 1-3 条**策略备忘**。
    # 风格要求
    不要使用“绝对禁止”、“神圣法则”等极端词汇。
    使用**专业、客观**的建议口吻。
    # 示例
    不好: "绝对禁止在RSI高位买入，否则是死罪！"
    好: "注意：在动量指标高位时，应适当收紧止损，避免激进追高。"
    # 输出格式
    [MEMO]: <备忘内容>
            """),
        ])
        chain = prompt | llm | StrOutputParser()
        summary_report = []
        for mem, role in zip(target_memories, roles):
            recent_logs = mem.fetch_recent_logs(k=7)
            if not recent_logs: continue
            logs_text = "\n".join(recent_logs)
            res = chain.invoke({"role_name": role, "daily_logs": logs_text})
            mem.add_weekly_rule(res)
            summary_report.append(f"【{role}】: {res}")
        return {
            "weekly_summary": "\n".join(summary_report)
        }
    return weekly_node