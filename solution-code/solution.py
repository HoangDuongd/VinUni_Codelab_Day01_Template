"""Completed solution for Day 1 — LLM API Foundation."""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv


# Load API keys from a .env file when present. Existing environment variables
# are preserved by python-dotenv's default override=False behaviour.
load_dotenv()


PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call OpenAI and return response text, latency, and token usage."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start

    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return response.choices[0].message.content or "", latency, usage


def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call Gemini and return response text, latency, and token usage."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.perf_counter() - start

    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }
    return response.text or "", latency, usage


def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call Anthropic and return response text, latency, and token usage."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return response.content[0].text or "", latency, usage


def _calculate_cost(model: str, usage: dict) -> float:
    """Calculate token cost using the fixed price table supplied by the lab."""
    rates = PRICING_1M_TOKENS[model]
    return (
        usage["input_tokens"] * rates["input"]
        + usage["output_tokens"] * rates["output"]
    ) / 1_000_000


def _comparison_entry(model: str, result: tuple[str, float, dict]) -> dict:
    """Convert one provider result tuple to the comparison result format."""
    response, latency, usage = result
    return {
        "response": response,
        "latency": latency,
        "cost": _calculate_cost(model, usage),
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
    }


def compare_models(prompt: str) -> dict:
    """Compare GPT-4o, GPT-4o Mini, and Gemini Flash on one prompt."""
    gpt4o = call_openai(prompt, model=OPENAI_MODEL)
    gpt4o_mini = call_openai(prompt, model=OPENAI_MINI_MODEL)
    gemini_flash = call_gemini(prompt, model=GEMINI_MODEL)

    return {
        "gpt4o": _comparison_entry(OPENAI_MODEL, gpt4o),
        "gpt4o_mini": _comparison_entry(OPENAI_MINI_MODEL, gpt4o_mini),
        "gemini_flash": _comparison_entry(GEMINI_MODEL, gemini_flash),
    }


def streaming_chatbot() -> None:
    """Run a terminal chatbot with Gemini streaming and three-turn history."""
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        current_message = {
            "role": "user",
            "parts": [{"text": user_input}],
        }
        response_parts: list[str] = []
        print("Assistant: ", end="", flush=True)

        response_stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=history + [current_message],
        )
        for chunk in response_stream:
            chunk_text = chunk.text or ""
            response_parts.append(chunk_text)
            print(chunk_text, end="", flush=True)
        print()

        history.extend(
            [
                current_message,
                {
                    "role": "model",
                    "parts": [{"text": "".join(response_parts)}],
                },
            ]
        )
        history = history[-6:]


def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """Run a callable, retrying failures with exponential backoff."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if base_delay < 0:
        raise ValueError("base_delay must be non-negative")

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2**attempt))

    raise RuntimeError("unreachable")


def batch_compare(prompts: list[str]) -> list[dict]:
    """Compare all models for every prompt while preserving input order."""
    results = []
    for prompt in prompts:
        comparison = compare_models(prompt)
        results.append({**comparison, "prompt": prompt})
    return results


def _table_cell(value: Any) -> str:
    """Escape a value so it remains within one Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _truncate(text: str, limit: int = 50) -> str:
    """Truncate text to at most limit characters, including an ellipsis."""
    clean_text = _table_cell(text)
    if len(clean_text) <= limit:
        return clean_text
    return clean_text[: limit - 3] + "..."


def format_comparison_table(results: list[dict]) -> str:
    """Format batch comparison results as a Markdown table."""
    model_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---:|---:|---:|",
    ]

    for result in results:
        prompt = _table_cell(result["prompt"])
        for model_key, display_name in model_names.items():
            stats = result[model_key]
            lines.append(
                f"| {prompt} | {display_name} | {_truncate(stats['response'])} "
                f"| {stats['latency']:.3f}s "
                f"| {stats['input_tokens']}/{stats['output_tokens']} "
                f"| ${stats['cost']:.8f} |"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    test_prompt = (
        "Hãy giải thích sự khác biệt giữa temperature và top_p "
        "bằng tiếng Việt ngắn gọn trong 2 câu."
    )
    try:
        result = compare_models(test_prompt)
        print(format_comparison_table([{**result, "prompt": test_prompt}]))
    except Exception as exc:
        print(f"Skipping live API comparison test: {exc}")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as exc:
        print(f"Chatbot failed to start: {exc}")
