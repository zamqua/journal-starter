import json

import openai
from openai.types.chat import ChatCompletionMessageParam

from api.config import get_settings


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

    parsed = json.loads(content)

    return {
        "entry_id": entry_id,
        "sentiment": parsed["sentiment"],
        "summary": parsed["summary"],
        "topics": parsed["topics"],
    }
