import regex as re
import os
from collections import defaultdict
from multiprocessing import Pool
from .pretokenization_example import find_chunk_boundaries


def pretokenize_chunk(args):
    input_path, start, end, special_tokens = args
    local_pretoken_counts = defaultdict(int)
    local_byte_pair_counts = defaultdict(int)
    local_pair_loc = defaultdict(set)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    delimiter_pattern = "|".join(escaped_special_tokens)
    
    with open(input_path, "rb") as text:
        text.seek(start)
        chunk = text.read(end - start).decode("utf-8", errors="ignore")
        chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
        blocks = re.split(delimiter_pattern, chunk)

        for block in blocks:
            for match in re.finditer(PAT, block):
                pretoken_encoded = match.group().encode("utf-8")
                byte_tuple = tuple(bytes([pretoken]) for pretoken in pretoken_encoded)
                local_pretoken_counts[byte_tuple] += 1

        for byte_tuple, freq in local_pretoken_counts.items():
            for k in range(len(byte_tuple)-1):
                byte_pair = (byte_tuple[k], byte_tuple[k+1])
                local_byte_pair_counts[byte_pair] += freq
                local_pair_loc[byte_pair].add(byte_tuple)
        
    return (local_pretoken_counts, local_byte_pair_counts, local_pair_loc)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretoken_counts = defaultdict(int)
    byte_pair_counts = defaultdict(int)
    pair_loc = defaultdict(set)

    with open(input_path, 'rb') as text:
        num_processes = 8
        boundaries = find_chunk_boundaries(text, num_processes, b"<|endoftext|>")

    # multi-processed pretokenization
    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(num_processes) as pool:
        chunk_results = pool.map(pretokenize_chunk, chunk_args)
    for local_pretoken_counts, local_byte_pair_counts, local_pair_loc in chunk_results:
        for pretoken, freq in local_pretoken_counts.items():
            pretoken_counts[pretoken] += freq
        for byte_pair, freq in local_byte_pair_counts.items():
            byte_pair_counts[byte_pair] += freq
        for byte_pair, pretoken_set in local_pair_loc.items():
            pair_loc[byte_pair] |= pretoken_set

    # merging process
    vocab = {k: bytes([k]) for k in range(256)}
    vocab_idx = 256
    for token in special_tokens:
        vocab[vocab_idx] = token.encode("utf-8")
        vocab_idx += 1

    merged_pairs = []
    while len(vocab) < vocab_size:
        max_pair = max(byte_pair_counts, key=lambda pair: (byte_pair_counts[pair], pair))
        merged_pairs.append(max_pair)
        vocab[vocab_idx] = max_pair[0] + max_pair[1]
        vocab_idx += 1

        for byte_tuple in pair_loc[max_pair]:
            freq = pretoken_counts[byte_tuple]
            pretoken_idx = 0
            rebuilt_pretoken = []

            while pretoken_idx < len(byte_tuple):
                if pretoken_idx == len(byte_tuple) - 1:
                    rebuilt_pretoken.append(byte_tuple[pretoken_idx])
                    break
                byte_pair = (byte_tuple[pretoken_idx], byte_tuple[pretoken_idx+1])
                if byte_pair == max_pair:
                    byte_pair_counts[max_pair] -= freq
                    if pretoken_idx > 0:
                        byte_pair_counts[(rebuilt_pretoken[-1], byte_tuple[pretoken_idx])] -= freq
                        byte_pair_counts[(rebuilt_pretoken[-1], max_pair[0] + max_pair[1])] += freq
                    if pretoken_idx < len(byte_tuple) - 2:
                        byte_pair_counts[(byte_tuple[pretoken_idx+1], byte_tuple[pretoken_idx+2])] -= freq
                        byte_pair_counts[(max_pair[0] + max_pair[1], byte_tuple[pretoken_idx+2])] += freq
                    rebuilt_pretoken.append(byte_pair[0] + byte_pair[1])
                    pretoken_idx += 2
                else:
                    rebuilt_pretoken.append(byte_tuple[pretoken_idx])
                    pretoken_idx += 1
            
            rebuilt_pretoken = tuple(rebuilt_pretoken)
            pretoken_counts[rebuilt_pretoken] += freq
            for k in range(len(rebuilt_pretoken) - 1):
                byte_pair = (rebuilt_pretoken[k], rebuilt_pretoken[k+1])
                pair_loc[byte_pair].discard(byte_tuple)
                pair_loc[byte_pair].add(rebuilt_pretoken)
        
        for byte_tuple in pair_loc[max_pair]:
            pretoken_counts.pop(byte_tuple)
        pair_loc.pop(max_pair)

    return (vocab, merged_pairs)