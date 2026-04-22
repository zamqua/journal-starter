import json
import logging
import re
from typing import Any

import openai
from openai.types.chat import ChatCompletionMessageParam

from api.config import get_settings

logger = logging.getLogger("llm_service")


def extract_json_from_markdown(content: str) -> dict[str, Any]:
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, content)

    clean_json_str = match.group(1).strip() if match else content.strip()

    try:
        return json.loads(clean_json_str)
    except json.JSONDecodeError as e:
        # Provide helpful debugging info in the error message
        logger.error(f"Failed to parse string: {clean_json_str}")
        raise ValueError(f"Failed to parse JSON from LLM output: {e!s}") from e


def _default_client() -> openai.AsyncOpenAI:
    """Construct the real OpenAI client from application settings.

    Called lazily from ``analyze_journal_entry`` so tests can inject a
    ``MockAsyncOpenAI`` without ever triggering this code path.
    """
    settings = get_settings()

    return openai.AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


async def analyze_journal_entry(
    entry_id: str,
    entry_text: str,
    client: openai.AsyncOpenAI | None = None,
) -> dict:
    settings = get_settings()

    """Analyze a journal entry using an OpenAI-compatible LLM.
    Args:
        entry_id: ID of the entry being analyzed (pass through to the result).
        entry_text: Combined work + struggle + intention text.
        client: OpenAI client. If None, a default one is constructed from
            application settings. Tests pass in a MockAsyncOpenAI here; production code
            in the router calls this with no ``client`` argument.

    Returns:
        A dict matching AnalysisResponse:
            {
                "entry_id":  str,
                "sentiment": str,   # "positive" | "negative" | "neutral"
                "summary":   str,
                "topics":    list[str],
            }
    """

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You analyze journal entries and return JSON with sentiment, summary, and topics.",
        },
        {"role": "user", "content": f"entry_id: {entry_id}\n\nJournal entry:\n{entry_text}"},
    ]
    if client is None:
        client = _default_client()

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Model returned empty content")

    parsed = extract_json_from_markdown(content)

    return {
        "entry_id": entry_id,
        "sentiment": parsed["sentiment"],
        "summary": parsed["summary"],
        "topics": parsed["topics"],
    }
