"""Model pricing table and cost estimation utilities.

Prices are in CNY per 1M tokens (硅基流动 SiliconFlow published prices).
Update these when prices change.
"""

from __future__ import annotations

from dataclasses import dataclass

# (input_price_per_1M_tokens_CNY, output_price_per_1M_tokens_CNY)
SILICONFLOW_PRICING: dict[str, tuple[float, float]] = {
    # DeepSeek
    "deepseek-ai/DeepSeek-V3":           (2.0,  8.0),
    "deepseek-ai/DeepSeek-R1":           (4.0,  16.0),
    "Pro/deepseek-ai/DeepSeek-R1":       (4.0,  16.0),
    "deepseek-ai/DeepSeek-V2.5":         (1.33, 1.33),
    # Qwen
    "Qwen/Qwen2.5-72B-Instruct":        (4.13, 4.13),
    "Qwen/Qwen2.5-32B-Instruct":        (1.26, 1.26),
    "Qwen/Qwen2.5-14B-Instruct":        (0.70, 0.70),
    "Qwen/Qwen2.5-7B-Instruct":         (0.35, 0.35),
    "Qwen/Qwen2.5-Coder-32B-Instruct":  (1.26, 1.26),
    "Qwen/QwQ-32B":                      (1.26, 1.26),
    # Llama
    "meta-llama/Meta-Llama-3.1-70B-Instruct": (4.13, 4.13),
    "meta-llama/Meta-Llama-3.1-8B-Instruct":  (0.42, 0.42),
    # GLM
    "THUDM/GLM-4-9B-Chat":              (0.50, 0.50),
}

# OpenAI / Anthropic etc. (USD per 1M tokens)
OTHER_PRICING_USD: dict[str, tuple[float, float]] = {
    "gpt-4o":         (2.50, 10.00),
    "gpt-4o-mini":    (0.15, 0.60),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
}

CNY_PER_USD = 7.25


@dataclass
class CostEstimate:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost: float   # CNY
    output_cost: float  # CNY
    total_cost: float   # CNY
    currency: str = "CNY"
    model: str = ""


def _strip_provider_prefix(model: str) -> str:
    """Remove 'openai/' prefix to get the bare model name."""
    if model.startswith("openai/"):
        return model[len("openai/"):]
    if model.startswith("anthropic/"):
        return model[len("anthropic/"):]
    return model


def get_price_per_1m(model: str) -> tuple[float, float, str]:
    """Return (input_price, output_price, currency) per 1M tokens.

    Tries SiliconFlow table first, then falls back to OTHER table.
    Returns (0, 0, "CNY") if model not found.
    """
    bare = _strip_provider_prefix(model)

    if bare in SILICONFLOW_PRICING:
        inp, out = SILICONFLOW_PRICING[bare]
        return inp, out, "CNY"

    if bare in OTHER_PRICING_USD:
        inp, out = OTHER_PRICING_USD[bare]
        return inp * CNY_PER_USD, out * CNY_PER_USD, "CNY"

    return 0.0, 0.0, "CNY"


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> CostEstimate:
    """Estimate cost for a given token usage."""
    inp_price, out_price, currency = get_price_per_1m(model)
    input_cost = prompt_tokens / 1_000_000 * inp_price
    output_cost = completion_tokens / 1_000_000 * out_price
    return CostEstimate(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        currency=currency,
        model=model,
    )


def estimate_batch_cost(
    model: str,
    num_tasks: int,
    avg_prompt_tokens: int,
    avg_completion_tokens: int,
) -> CostEstimate:
    """Estimate total cost for a batch of tasks."""
    total_prompt = avg_prompt_tokens * num_tasks
    total_completion = avg_completion_tokens * num_tasks
    return estimate_cost(model, total_prompt, total_completion)


def format_cost_report(est: CostEstimate) -> str:
    """Human-readable single-line cost report."""
    return (
        f"Tokens: prompt={est.prompt_tokens:,} completion={est.completion_tokens:,} "
        f"total={est.total_tokens:,}  |  "
        f"Cost: ¥{est.total_cost:.4f} "
        f"(input ¥{est.input_cost:.4f} + output ¥{est.output_cost:.4f})"
    )


def format_batch_estimate(
    model: str,
    num_tasks: int,
    avg_prompt_tokens: int,
    avg_completion_tokens: int,
) -> str:
    """Pretty-print a batch cost estimate for user confirmation."""
    est = estimate_batch_cost(model, num_tasks, avg_prompt_tokens, avg_completion_tokens)
    bare = _strip_provider_prefix(model)
    inp_price, out_price, _ = get_price_per_1m(model)

    lines = [
        "=" * 60,
        "  批量任务费用预估  /  Batch Cost Estimate",
        "=" * 60,
        f"  模型 (Model):           {bare}",
        f"  定价 (Price/1M tokens): input ¥{inp_price:.2f}  output ¥{out_price:.2f}",
        f"  任务数 (Tasks):         {num_tasks}",
        f"  预估 prompt tokens:     ~{avg_prompt_tokens:,} × {num_tasks} = {avg_prompt_tokens * num_tasks:,}",
        f"  预估 completion tokens: ~{avg_completion_tokens:,} × {num_tasks} = {avg_completion_tokens * num_tasks:,}",
        "-" * 60,
        f"  预估总费用 (Est. cost):  ¥{est.total_cost:.4f}",
        "=" * 60,
    ]
    if inp_price == 0 and out_price == 0:
        lines.insert(-1, "  ⚠  未找到该模型的定价信息，费用显示为 0")
    return "\n".join(lines)
