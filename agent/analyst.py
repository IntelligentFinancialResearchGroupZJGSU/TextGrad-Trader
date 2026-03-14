from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils import Toolkit
from langchain_openai import ChatOpenAI
from config import get_config


def create_analyst_node(llm, toolkit, system_message, tools, output_field):
    """
    参数：
        llm: 智能体使用的语言模型实例
        toolkit: 智能体可用的工具集合
        system_message: 定义智能体角色和目标的具体指令
        tools: 此智能体被允许使用的工具包中特定工具的列表
        output_field: AgentState中存储此智能体最终报告的键
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """# SYSTEM CONTEXT
    You are a specialized AI agent collaborating within a hierarchy of financial analysts.
    Current Date: {current_day}
    Target Company Ticker: {ticker}

    # YOUR ROLE & OBJECTIVES
    {system_message}

    # TOOL USAGE & COLLABORATION
    You have access to the following tools: [{tool_names}].
    - Use tools vigorously to gather data and back up your analysis.
    - You are part of a team. If you cannot answer the question fully because of tool limitations:
      1. Execute what you can to make significant progress.
      2. Explicitly state what you have found.
      3. Explicitly state what is missing so the next agent knows exactly what to do.
    - Do NOT make up information. If the tool returns no data, report that.

    # RESPONSE GUIDELINES
    Follow the output format specified in your role description above strictly.
    """),
        MessagesPlaceholder(variable_name="messages"),
    ])
    tool_name = ", ".join([tool.name for tool in tools])
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=tool_name)

    # 这是将作为图中节点执行的实际函数
    def analyst_node(state):
        prompt_with_data = prompt.partial(current_day=state["trade_date"], ticker=state["symbol"])
        chain = prompt_with_data | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])
        report = ""
        messages = [result]  # 初始消息

        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']

                # 查找对应的工具函数
                tool_function = None
                for tool in tools:
                    if tool.name == tool_name:
                        tool_function = tool
                        break

                if tool_function:
                    # 执行工具调用
                    try:
                        tool_result = tool_function.invoke(tool_args)
                        tool_message = ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call['id']
                        )
                        messages.append(tool_message)
                    except Exception as e:
                        error_message = ToolMessage(
                            content=f"工具执行错误: {str(e)}",
                            tool_call_id=tool_call['id']
                        )
                        messages.append(error_message)
                else:
                    error_message = ToolMessage(
                        content=f"错误: 找不到工具 {tool_name}",
                        tool_call_id=tool_call['id']
                    )
                    messages.append(error_message)

        final_result = chain.invoke(messages)
        report = final_result.content
        return {
            "messages": messages,
            output_field: report,
        }

    return analyst_node


