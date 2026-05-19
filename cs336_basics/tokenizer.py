import regex as re
from pathlib import Path
from typing import Iterable, Iterator
from cs336_basics.bpe_utils import load_bpe, PAT, iter_split_with_specials
from collections import Counter
from itertools import pairwise

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab # token id -> token bytes
        self.merges = merges
        self.special_tokens = special_tokens
        if special_tokens:
            self.special_pat = re.compile("|".join(re.escape(st) for st in sorted(special_tokens, key=len, reverse=True))
            )
        else:
            self.special_pat = None
        self.vocab_rev = {v:k for k,v in vocab.items()} # token bytes -> token id
        self.pt_to_tokens = {}

    def from_files(cls, bpe_path: str, special_tokens: list[str] | None = None):
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        # initialize pre token list and counter
        token_id_list = []

        for piece, sep in iter_split_with_specials(text, self.special_pat):
            # init for this segment
            pt_cnt = Counter()
            pt_list_input_order = []
            pair_to_pt = {}
            # for each segment divided by the special tokens
            for m in re.finditer(PAT, piece):
                key = tuple(bytes([b]) for b in m.group(0).encode("utf-8"))
                pt_cnt[key] += 1
                pt_list_input_order.append(key)

            for pt in pt_cnt:
                if pt not in self.pt_to_tokens: # performance bug fix #1: cache the pre-token results. 3 times faster after this fix, 1910.749s -> 669.534s on TinyStoriesV2-GPT4-train.txt.
                    self.pt_to_tokens[pt] = list(pt) # raw pre token key is all single bytes
                    for raw_pair in pairwise(pt):
                        if raw_pair not in pair_to_pt:
                            pair_to_pt[raw_pair] = set()
                        pair_to_pt[raw_pair].add(pt) 

            # merge in the same order of training
            for pair_merge in self.merges:
                if pair_merge not in pair_to_pt:
                    continue

                for pt in pair_to_pt[pair_merge]:
                    tokens = self.pt_to_tokens[pt] # tokens list now
                    new_tokens = []
                    new_pairs = set() # new borned pairs after merge
                    mx = len(tokens)
                    i = 0

                    while i < mx - 1:
                        pair = (tokens[i], tokens[i+1])
                        new_token = tokens[i] + tokens[i+1]
                        if pair == pair_merge:
                            if i > 0:
                                pair_pre_new = (tokens[i-1], new_token)
                                new_pairs.add(pair_pre_new)
                            if i < mx - 2:
                                pair_aft_new = (new_token, tokens[i+2])
                                new_pairs.add(pair_aft_new)
                            new_tokens.append(new_token)
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    
                    if i == mx - 1:
                        new_tokens.append(tokens[-1])
                    
                    # print("debug", "new_pair", new_pairs, new_tokens)
                    # add new links for new pair 
                    for new_pair in new_pairs:
                        if new_pair not in pair_to_pt:
                            pair_to_pt[new_pair] = set() # could born in other pts
                        pair_to_pt[new_pair].add(pt)
                    
                    self.pt_to_tokens[pt] = new_tokens # update the tokens storage for this pt

                del pair_to_pt[pair_merge] # drop that link since there is no more pair_merge
            
            token_id_list.extend([self.vocab_rev[token] for pt in pt_list_input_order for token in self.pt_to_tokens[pt]])
            if sep:
                token_id_list.append(self.vocab_rev[sep.encode('utf-8')])

        return token_id_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text) # bug 1 fixed: use yield from, which means yield each token one by one inside the encoded result
    def decode(self, ids: list[int]) -> str:
        return b"".join([ self.vocab[i] for i in ids]).decode("utf-8", errors="replace") 

if __name__ == "__main__":
    import pickle
    import time
    import numpy as np

    current_file_path = Path(__file__).parent.resolve()
    data_path = Path.joinpath(current_file_path, "../data/").absolute()
    out_path = Path.joinpath(current_file_path, "../out/").absolute()

    # tinystory
    bpe_path = out_path / "bpe_dump_tinystory"
    vocab, merges = load_bpe(bpe_path)
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    # train
    with open(data_path / "TinyStoriesV2-GPT4-train.txt", "r") as f:
        text = f.read()
    t0 = time.perf_counter()
    token_id_list = tokenizer.encode(text)
    token_id_list = np.array(token_id_list, dtype="uint16")
    print(f"TinyStories train encode: {time.perf_counter() - t0:.3f}s")
    with open(out_path / "TinyStoriesV2-GPT4-train.tid", "wb") as f:
        pickle.dump({"token_id_list": token_id_list}, f, protocol=pickle.HIGHEST_PROTOCOL)

    # valid
    with open(data_path / "TinyStoriesV2-GPT4-valid.txt", "r") as f:
        text = f.read()
    t0 = time.perf_counter()
    token_id_list = tokenizer.encode(text)
    token_id_list = np.array(token_id_list, dtype="uint16")
    print(f"TinyStories valid encode: {time.perf_counter() - t0:.3f}s")
    with open(out_path / "TinyStoriesV2-GPT4-valid.tid", "wb") as f:
        pickle.dump({"token_id_list": token_id_list}, f, protocol=pickle.HIGHEST_PROTOCOL)

    # owt
    bpe_path = out_path / "bpe_dump_owt"
    vocab, merges = load_bpe(bpe_path)
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    # train
    with open(data_path / "owt_train.txt", "r") as f:
        text = f.read()
    t0 = time.perf_counter()
    token_id_list = tokenizer.encode(text)
    token_id_list = np.array(token_id_list, dtype="uint16")
    print(f"OWT train encode: {time.perf_counter() - t0:.3f}s")
    with open(out_path / "owt_train.tid", "wb") as f:
        pickle.dump({"token_id_list": token_id_list}, f, protocol=pickle.HIGHEST_PROTOCOL)

    # valid
    with open(data_path / "owt_valid.txt", "r") as f:
        text = f.read()
    t0 = time.perf_counter()
    token_id_list = tokenizer.encode(text)
    token_id_list = np.array(token_id_list, dtype="uint16")
    print(f"OWT valid encode: {time.perf_counter() - t0:.3f}s")
    with open(out_path / "owt_valid.tid", "wb") as f:
        pickle.dump({"token_id_list": token_id_list}, f, protocol=pickle.HIGHEST_PROTOCOL)

