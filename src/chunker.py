"""
Splits long text into token-bounded chunks.

"""
import tiktoken


def chunk_text(
    text: str,
    max_tokens: int = 1500,
    overlap_tokens: int = 150,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """
    Split text into chunks of at most `max_tokens`, with `overlap_tokens`
    repeated between consecutive chunks so context isn't lost at the seams.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))

        if end == len(tokens):
            break
        start = end - overlap_tokens  # step back so chunks overlap

    return chunks


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
