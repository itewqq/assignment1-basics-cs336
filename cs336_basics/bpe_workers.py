from collections import Counter
import regex as re


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