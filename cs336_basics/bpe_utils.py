from collections import Counter
import pickle
import regex as re
from collections.abc import Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def count_pre_tokens_for_chunk(args):
    input_path, start, end, special_tokens, pat = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    counts = Counter()

    if special_tokens:
        special_pat = re.compile(
            "|".join(re.escape(st) for st in sorted(special_tokens, key=len, reverse=True))
        )
        pieces = special_pat.split(chunk)
    else:
        pieces = [chunk]

    for piece in pieces:
        for m in re.finditer(pat, piece):
            key = tuple(bytes([b]) for b in m.group(0).encode("utf-8"))
            counts[key] += 1

    return counts


def save_bpe(vocab, merges, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(
            {
                "vocab": vocab,
                "merges": merges,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_bpe(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    return obj["vocab"], obj["merges"]

def iter_split_with_specials(
    text: str,
    special_pat,
) -> Iterator[tuple[str, str | None]]:
    """
    Yields (piece, special_token_after_piece).

    Example:
        text = "abc<|endoftext|>def<|pad|>ghi"

    yields:
        ("abc", "<|endoftext|>")
        ("def", "<|pad|>")
        ("ghi", None)
    """

    if not special_pat:
        yield text, None
        return

    last = 0

    for m in special_pat.finditer(text):
        yield text[last:m.start()], m.group(0)
        last = m.end()

    yield text[last:], None