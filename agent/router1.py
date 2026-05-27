import json
import math
import re


EXPERT_KEYS = [
    "market_analyst",
    "news_analyst",
    "fundamentals_analyst",
    "industry_analyst",
    "momentum_analyst",
]


def _normalize_weights(weights):
    cleaned = {}
    total = 0.0
    for key in EXPERT_KEYS:
        value = max(float(weights.get(key, 0.0)), 0.0)
        cleaned[key] = value
        total += value

    if total <= 0:
        uniform = 1.0 / len(EXPERT_KEYS)
        return {key: uniform for key in EXPERT_KEYS}

    return {key: value / total for key, value in cleaned.items()}


def _extract_json_object(text):
    if not text:
        return {}

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _parse_weight_distribution(text):
    data = _extract_json_object(text)
    if not isinstance(data, dict):
        return _normalize_weights({})
    return _normalize_weights(data)


def _parse_score_distribution(text):
    data = _extract_json_object(text)
    if not isinstance(data, dict):
        return {key: 0.0 for key in EXPERT_KEYS}

    scores = {}
    for key in EXPERT_KEYS:
        try:
            scores[key] = float(data.get(key, 0.0))
        except (TypeError, ValueError):
            scores[key] = 0.0
    return scores


def _compute_prior_weights(past_memories, temperature=0.2):
    if not past_memories:
        return _normalize_weights({})

    logits = []
    historical_weights = []
    for mem in past_memories:
        similarity = max(float(mem.get("similarity_score", 0.0)), 0.0)
        logits.append(similarity / max(temperature, 1e-6))
        historical_weights.append(_parse_weight_distribution(mem.get("recommendation", "")))

    max_logit = max(logits)
    exp_scores = [math.exp(logit - max_logit) for logit in logits]
    score_sum = sum(exp_scores)
    if score_sum <= 0:
        return _normalize_weights({})

    prior = {key: 0.0 for key in EXPERT_KEYS}
    for memory_weight, historical_weight in zip(exp_scores, historical_weights):
        coeff = memory_weight / score_sum
        for key in EXPERT_KEYS:
            prior[key] += coeff * historical_weight[key]

    return _normalize_weights(prior)


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
                f"【历史场景】{situation_short}\n"
                f"【历史权重】{rec}\n"
                f"【匹配度】{score:.2%}"
            )
            past_memory_str.append(entry)

        prior_weights = _compute_prior_weights(past_memories)

        prompt = f"""
# 角色
你是一个 Bayesian Contextual Gating 模块。你的任务不是直接输出最终权重，而是评估 5 位分析师在当前市场状态下的逻辑一致性、信号强度与冲突解决能力，并输出每位分析师的 Lcoh score。

# 当前分析师报告
{curr_situation}

# 历史相似场景
以下内容仅作为历史参考，用于帮助你理解相似市场结构。系统已经在本地根据这些历史样本计算好了 Bayesian prior，你不需要重复计算 prior。
{chr(10).join(past_memory_str) if past_memory_str else "无历史相似场景。"}

# 已计算的 Bayesian prior
{json.dumps(prior_weights, ensure_ascii=False)}

# 你的评分原则
1. 评估每位分析师当前报告的因果逻辑是否自洽。
2. 评估每位分析师的信号是否强、是否与当前市场状态匹配。
3. 当不同分析师之间出现冲突时，优先给逻辑更完整、证据更强、解释力更高的一方更高分数。
4. 你输出的是 Lcoh score，不是最终权重。系统会在本地按如下公式做最终计算：
   w_i ∝ w_prior_i * exp(Lcoh_i)

# 输出要求
1. 只输出一个 JSON 对象。
2. 不要输出 markdown。
3. 不要输出解释文字。
4. 每个值都是 float score，建议范围 [-5, 5]。

输出模板：
{{
    "market_analyst": 0.0,
    "news_analyst": 0.0,
    "fundamentals_analyst": 0.0,
    "industry_analyst": 0.0,
    "momentum_analyst": 0.0
}}
"""
        response = llm.invoke(prompt)
        coherence_scores = _parse_score_distribution(response.content)
        posterior_weights = {
            key: prior_weights[key] * math.exp(coherence_scores[key])
            for key in EXPERT_KEYS
        }
        final_weights = _normalize_weights(posterior_weights)

        return {
            "router": json.dumps(final_weights, ensure_ascii=False),
        }

    return router_node


create_router = create_bayesian_contextual_gating_router
