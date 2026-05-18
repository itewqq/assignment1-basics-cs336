import os
from itertools import pairwise
from pathlib import Path
from copy import deepcopy
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO
from line_profiler import profile
import heapq
from dataclasses import dataclass
from cs336_basics.bpe_workers import count_pre_tokens_for_chunk

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

# I think it is a stupid idea to optimize an algorithm in Python, but whatever this lazy heap make us pass the 2 minutes threshold hinted by the assignment doc. 
@dataclass(frozen=True)
class HeapItem:
    count: int
    pair: tuple[bytes, bytes]
    timestamp: int

    def __lt__(self, other: "HeapItem") -> bool:
        # heapq is min-heap, so return True when self should come BEFORE other.

        # 1. larger count wins
        if self.count != other.count:
            return self.count > other.count

        # 2. lexicographically larger bytes-pair wins
        if self.pair != other.pair:
            return self.pair > other.pair

        # 3. larger timestamp wins
        return self.timestamp > other.timestamp

@profile
def bpe_tokenizer_train(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]] ]:
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    tasks = [
        (input_path, start, end, special_tokens, PAT)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        counters = executor.map(count_pre_tokens_for_chunk, tasks)
    pre_token_cnt_map = Counter()
    max_cnt = 0
    for c in counters:
        pre_token_cnt_map.update(c)
    pre_token_cnt_map = dict(pre_token_cnt_map)

    # Init vocab
    vocab = [st.encode("utf-8") for st in special_tokens]
    vocab.extend([i.to_bytes(1, "little") for i in range(256)])
    lazy_heap = [] # lazy heap, timestamp based
    pair_stat = {} # store the cnt, the latest timestamp
    pair_to_pt = {} # pair to pre-token map
    pt_to_tokens = {}
    for pt_tuple,cnt in pre_token_cnt_map.items():
        max_cnt = max(max_cnt, cnt)
        pt_to_tokens[pt_tuple] = list(pt_tuple) # initially just all bytes alone
        for pair in pairwise(pt_tuple):
            if pair in pair_to_pt:
                pair_to_pt[pair].add(pt_tuple)
            else:
                pair_to_pt[pair] = set()
                pair_to_pt[pair].add(pt_tuple) # other wise it will be mergered, weird fucking python
            if pair in pair_stat:
                pair_stat[pair][0] += cnt
            else:
                pair_stat[pair] = [cnt, len(vocab)] # pt cnt, timestamp=len(vocab)

    for pair,stat in pair_stat.items():
        lazy_heap.append( HeapItem(stat[0], pair, stat[1]) ) # cnt, pair, timestamp
    heapq.heapify(lazy_heap) # init the heap

    # Merges
    merges = []
    while len(vocab) < vocab_size:
        best = heapq.heappop(lazy_heap)
        cnt, pair_most, timestamp = best.count, best.pair, best.timestamp
        # print("best:", cnt, pair_most, timestamp)
        if pair_most not in pair_stat or timestamp < pair_stat[pair_most][1]:
            continue # drop outdated value
        

        # print(pair_most[0], pair_most[1])
        new_token = pair_most[0] + pair_most[1] # merge the pair of bytearray to make new token bytearray
        vocab.append(new_token)
        ts_new = len(vocab)
        merges.append(pair_most)
        modified_pairs = set() # re enqueue candidates
        del pair_stat[pair_most] # delete because we have new token to replace them all

        def update_pair_stat(pair, cnt):
            if pair not in pair_stat:
                pair_stat[pair] = [cnt, ts_new]
            else:
                pair_stat[pair][0] += cnt
                pair_stat[pair][1] = ts_new

        # update the infected pre-token
        for infected_pt in pair_to_pt[pair_most]:
            cnt = pre_token_cnt_map[infected_pt]
            cur_tokens = pt_to_tokens[infected_pt]
            pt_tokens_new = []
            i = 0
            mx = len(cur_tokens)

            # if b"".join(infected_pt) == b" the":
            #     print(cnt, pre_token_cnt_map[infected_pt], pair_most, cur_tokens)
            #     input("???")

            while i < mx - 1:
                pair = (cur_tokens[i], cur_tokens[i+1])
                if pair == pair_most:
                    # update current tokens
                    pt_tokens_new.append(new_token)
                    # update previous
                    if i > 0:
                        # old
                        pair_pre = (cur_tokens[i-1], cur_tokens[i])
                        update_pair_stat(pair_pre, -cnt) # pair_pre must exist so it is safe
                        modified_pairs.add(pair_pre)
                        # new
                        pair_pre_new = (cur_tokens[i-1], new_token)
                        update_pair_stat(pair_pre_new, cnt)
                        # add new link
                        if pair_pre_new not in pair_to_pt:
                            pair_to_pt[pair_pre_new] = set()
                        pair_to_pt[pair_pre_new].add(infected_pt) # other wise it will be mergered, weird 
                        modified_pairs.add(pair_pre_new)
                    # update after
                    if i < mx - 2:
                        # old
                        pair_aft = (cur_tokens[i+1], cur_tokens[i+2])
                        update_pair_stat(pair_aft, -cnt)
                        modified_pairs.add(pair_aft)
                        # new
                        pair_aft_new = (new_token, cur_tokens[i+2])
                        update_pair_stat(pair_aft_new, cnt)
                        # add new link
                        if pair_aft_new not in pair_to_pt:
                            pair_to_pt[pair_aft_new] = set()
                        pair_to_pt[pair_aft_new].add(infected_pt) # other wise it will be mergered, weird 
                        modified_pairs.add(pair_aft_new)
                    i += 2
                else:
                    pt_tokens_new.append(cur_tokens[i])
                    i += 1
            if i == mx - 1:
                pt_tokens_new.append(cur_tokens[-1]) # bug 1 fixed: forget the last one when iterating over the old tokens
            pt_to_tokens[infected_pt] = pt_tokens_new
        
        del pair_to_pt[pair_most] # remove those links
        # insert all modified pairs in the heap after all infected pt are handled
        for pair in modified_pairs:
            stat = pair_stat[pair]
            heapq.heappush(lazy_heap, HeapItem(stat[0], pair, stat[1]) )


    vocab = {i:t for i,t in enumerate(vocab)}
    return vocab, merges



if __name__ == "__main__":
    # local test
    current_file_path = Path(__file__).parent.resolve()
    test_file_path = Path.joinpath(current_file_path, "../data/TinyStoriesV2-GPT4-train.txt").absolute()
    vocab, merges = bpe_tokenizer_train(str(test_file_path), 10_000, ["<|endoftext|>"])
    results_dump_path = Path.joinpath(current_file_path, "../out").absolute()
    # # debug for answer
    import pprint
    with open(results_dump_path / "vocab", "w") as f:
        f.write(pprint.pformat(vocab))
    with open(results_dump_path / "merges", "w") as f:
        f.write(pprint.pformat(merges))