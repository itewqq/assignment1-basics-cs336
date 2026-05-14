import os
import regex as re
from itertools import pairwise
from pathlib import Path
from copy import deepcopy
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _count_pre_tokens_for_chunk(args):
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
            # byte-level BPE version
            key = tuple(bytes([b]) for b in m.group(0).encode("utf-8"))
            counts[key] += 1

    return counts

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def bpe_tokenizer_train(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]] ]:
    with open(input_path, "rb") as f:
        num_processes = 32
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    tasks = [
        (input_path, start, end, special_tokens, PAT)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        counters = executor.map(_count_pre_tokens_for_chunk, tasks)
    pre_token_cnt_map = Counter()
    for c in counters:
        pre_token_cnt_map.update(c)
    pre_token_cnt_map = dict(pre_token_cnt_map)

    # Init vocab
    vocab = [st.encode("utf-8") for st in special_tokens]
    vocab.extend([i.to_bytes(1, "little") for i in range(256)])
    
    # Merges
    merges = []
    while len(vocab) < vocab_size:
        pair_cnt = Counter()
        for pt_key,pt_cnt in pre_token_cnt_map.items():
            for pair in pairwise(pt_key):
                pair_cnt[pair] += pt_cnt
        if not pair_cnt:
            break
        pair_most = max(pair_cnt.items(), key=lambda x:(x[1], x[0]))[0]
        new_token = b"".join(pair_most)
        vocab.append(new_token)
        merges.append(pair_most)
        pre_token_cnt_map_new = Counter()
        for pt_key,pt_cnt in pre_token_cnt_map.items():
            key_new = []
            i = 0
            while i < len(pt_key):
                if i + 1 < len(pt_key) and (pt_key[i], pt_key[i+1]) == pair_most:
                    key_new.append(b"".join(pair_most))
                    i += 2
                else:
                    key_new.append(pt_key[i])
                    i += 1
            pre_token_cnt_map_new[tuple(key_new)] += pt_cnt

        pre_token_cnt_map = pre_token_cnt_map_new
    vocab = {i:t for i,t in enumerate(vocab)}
    return vocab, merges



if __name__ == "__main__":
    # local test
    current_file_path = Path(__file__).parent.resolve()
    test_file_path = Path.joinpath(current_file_path, "../data/TinyStoriesV2-GPT4-train.txt").absolute()
    vocab, merges = bpe_tokenizer_train(str(test_file_path), 10_000, ["<|endoftext|>"])
    print(list(vocab.items())[:10], merges[:10])