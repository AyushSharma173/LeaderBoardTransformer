from typing import List, Tuple
from cs336_basics.pretokenization_example import find_chunk_boundaries

from collections import defaultdict
import multiprocessing
import regex
import logging
import time

from tqdm import tqdm

# Precompile the GPT-2 pretokenization regex once per process
PATTERN_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = regex.compile(PATTERN_STR)

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    logger.info(f"Starting BPE training: target vocab size = {vocab_size:,}")
    logger.info(f"Input file: {input_path}")
    logger.info(f"Special tokens: {special_tokens}")

    with open(input_path, "rb") as f:
        
        sentinels = [tok.encode("utf-8") for tok in special_tokens]
        if b"<|endoftext|>" not in sentinels:        # always include canonical EOT
            sentinels.append(b"<|endoftext|>")
        boundaries = find_chunk_boundaries(f, 4000, *sentinels)
        logger.info(f"Found {len(boundaries):,} chunk boundaries for processing")
                    
        
        vocab = {i: bytes([i]) for i in range(256)}
        idx = 256
        logger.info(f"Initialized base vocabulary with {len(vocab)} byte tokens")
        
        for special_token in special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in vocab.values():
                vocab[idx] = token_bytes
                idx += 1
                logger.info(f"Added special token: {special_token}")
            else:
                logger.info(f"Special token {special_token} already exists in vocabulary")

        corpus = defaultdict(int)

        chunk_spans = list(zip(boundaries[:-1], boundaries[1:]))
        logger.info(f"Processing {len(chunk_spans):,} chunks...")

        args = [(start, end, input_path, special_tokens) for start, end in chunk_spans]

        # Multiprocessing can be slower than serial on small vocab/inputs; force serial for small sweeps
        chunk_start_time = time.time()
        if len(chunk_spans) < 4 or vocab_size <= 1000:
            logger.info("Using serial processing for chunk tokenization")
            results = []
            for i, a in enumerate(args):
                if i % max(1, len(args) // 10) == 0:  # Log progress every 10%
                    progress = (i / len(args)) * 100
                    logger.info(f"Chunk processing progress: {progress:.1f}% ({i:,}/{len(args):,})")
                results.append(process_chunk(a))
        else:
            logger.info(f"Using multiprocessing with {min(multiprocessing.cpu_count(), len(chunk_spans))} processes")
            with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(chunk_spans))) as pool:
                results = pool.map(process_chunk, args)
        
        chunk_time = time.time() - chunk_start_time
        logger.info(f"Chunk processing completed in {chunk_time:.2f}s")

        
        # Operate directly in bytes-token space for correctness and simplicity
        global_corpus = defaultdict(int)
        total_tokens = 0
        for local_corpus in results:
            for token_bytes, count in local_corpus.items():
                global_corpus[token_bytes] += count
                total_tokens += count
        
        logger.info(f"Corpus statistics:")
        logger.info(f"  - Total unique token sequences: {len(global_corpus):,}")
        logger.info(f"  - Total token occurrences: {total_tokens:,}")
        logger.info(f"  - Average sequence frequency: {total_tokens / len(global_corpus):.2f}")
                

        
        def merge_sequence(seq: Tuple[bytes, ...], pair: Tuple[bytes, bytes]) -> Tuple[bytes, ...]:
            A, B = pair
            out: list[bytes] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == A and seq[i+1] == B:
                    out.append(A + B)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            return tuple(out)
        
        merges = []

        # Build initial global pair counts and inverted index: pair -> set of sequences containing it
        def count_pairs_in_seq(seq: Tuple[bytes, ...]) -> dict[Tuple[bytes, bytes], int]:
            local = defaultdict(int)
            for a, b in zip(seq, seq[1:]):
                local[(a, b)] += 1
            return local

        pair2count: dict[Tuple[bytes, bytes], int] = defaultdict(int)
        pair2seqs: dict[Tuple[bytes, bytes], set] = defaultdict(set)

        for seq, freq in global_corpus.items():
            if len(seq) < 2:
                continue
            local = count_pairs_in_seq(seq)
            for p, c in local.items():
                pair2count[p] += c * freq
                pair2seqs[p].add(seq)

        merges_needed = vocab_size - len(vocab)
        logger.info(f"Starting BPE merging: {merges_needed:,} merges needed")
        logger.info(f"Initial pair count: {len(pair2count):,} unique pairs")
        
        merge_start_time = time.time()
        pbar = tqdm(total=merges_needed, desc="Merging BPE pairs", disable=False)
        merge_times = []
        
        while len(vocab) < vocab_size:
            iteration_start = time.time()
            
            if not pair2count:
                logger.warning("No more pairs to merge - stopping early")
                break

            # Pick pair by (count, lexicographically greater pair) as tie-breaker
            most_frequent_pair, pair_count = max(pair2count.items(), key=lambda x: (x[1], x[0]))

            A, B = most_frequent_pair
            merged_token = A + B

            vocab[idx] = merged_token
            merges.append((A, B))

            # Sequences impacted by this merge
            impacted = list(pair2seqs.get(most_frequent_pair, set()))
            
            # Log progress every 100 merges or at key milestones
            merge_num = len(merges)
            if merge_num % 100 == 0 or merge_num in [1, 10, 50]:
                elapsed = time.time() - merge_start_time
                remaining = merges_needed - merge_num
                if merge_num > 0:
                    avg_time_per_merge = elapsed / merge_num
                    estimated_remaining = avg_time_per_merge * remaining
                    logger.info(f"Merge {merge_num:,}/{merges_needed:,} | "
                              f"Pair: {A.hex()[:8]}...+{B.hex()[:8]}... (freq: {pair_count:,}) | "
                              f"Elapsed: {elapsed:.1f}s | "
                              f"ETA: {estimated_remaining:.1f}s | "
                              f"Vocab size: {len(vocab):,}")
                else:
                    logger.info(f"Merge {merge_num:,}/{merges_needed:,} | "
                              f"Pair: {A.hex()[:8]}...+{B.hex()[:8]}... (freq: {pair_count:,}) | "
                              f"Vocab size: {len(vocab):,}")

            # For each impacted sequence, update counts/indexes and corpus
            for seq in impacted:
                freq = global_corpus.get(seq, 0)
                if freq == 0:
                    # Already merged/consumed into another identical sequence
                    # Skip but ensure indexes don't retain stale pointers
                    continue

                # Remove this sequence's contribution from global pair counts
                old_pairs = count_pairs_in_seq(seq)
                for p, c in old_pairs.items():
                    pair2count[p] -= c * freq
                    if pair2count[p] <= 0:
                        pair2count.pop(p, None)
                # We'll rebuild pair2seqs entries for new sequence below; remove old mapping now
                for p in old_pairs.keys():
                    sset = pair2seqs.get(p)
                    if sset is not None:
                        sset.discard(seq)
                        if not sset:
                            pair2seqs.pop(p, None)

                # Merge this sequence (replace A,B with A+B)
                new_seq = merge_sequence(seq, most_frequent_pair)

                # Update corpus: remove old seq count, add to new_seq (coalesce if exists)
                global_corpus[seq] -= freq
                if global_corpus[seq] <= 0:
                    global_corpus.pop(seq, None)
                global_corpus[new_seq] += freq

                # Add new sequence contribution to global pair counts and indexes
                new_pairs = count_pairs_in_seq(new_seq)
                for p, c in new_pairs.items():
                    pair2count[p] += c * freq
                    if new_seq not in pair2seqs[p]:
                        pair2seqs[p].add(new_seq)

            # The merged pair's set is now stale; clear it (it may reappear due to different sequences)
            pair2seqs.pop(most_frequent_pair, None)

            idx += 1
            pbar.update(1)
            
            # Track merge timing for final statistics
            iteration_time = time.time() - iteration_start
            merge_times.append(iteration_time)

        pbar.close()
        
        # Log final merge statistics
        total_merge_time = time.time() - merge_start_time
        actual_merges = len(merges)
        logger.info(f"BPE merging completed!")
        logger.info(f"  - Total merges performed: {actual_merges:,}/{merges_needed:,}")
        logger.info(f"  - Total merge time: {total_merge_time:.2f}s")
        if merge_times:
            logger.info(f"  - Average time per merge: {sum(merge_times)/len(merge_times):.4f}s")
            logger.info(f"  - Slowest merge: {max(merge_times):.4f}s")
            logger.info(f"  - Fastest merge: {min(merge_times):.4f}s")

    # Log final training summary
    total_time = time.time() - start_time
    logger.info(f"BPE training completed in {total_time:.2f}s")
    logger.info(f"Final vocabulary size: {len(vocab):,} tokens")
    logger.info(f"Total merges: {len(merges):,}")
    
    return vocab, merges


        


def process_chunk(args):
    start, end, input_path, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
    # remove special tokens
    pattern = regex.compile("|".join(regex.escape(tok) for tok in special_tokens))
    # chunk = pattern.sub("", chunk)
    parts = regex.split(pattern, chunk)
    local_corpus = defaultdict(int)
    for part in parts:
        if not part:
            continue
        # run regex pretokenization
        matches = PAT_RE.finditer(part)

        
        for match in matches:
            pretoken = match.group(0)
            # represent each UTF-8 byte as its own bytes object for consistency with training
            pretoken_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
            local_corpus[pretoken_bytes] += 1
    
    return local_corpus




def test_debug():
    vocab = {i: bytes([i]) for i in range(256)}
    print(vocab)
    pass


import json

if __name__ == "__main__":
    # ✅ Path to raw training text
    input_path = "../data/owt_train.txt"

    # ✅ Max vocab size including special tokens and byte vocab
    vocab_size = 32_000

    # ✅ Must include this token for TinyStories
    special_tokens = ["<|endoftext|>"]

    # ✅ Train BPE tokenizer
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # ✅ Save vocab (convert bytes -> hex string so it's JSON serializable)
    print("Saving vocabulary and merges...")
    vocab_json = {str(token_id): token_bytes.hex() for token_id, token_bytes in vocab.items()}
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, indent=2)

    # ✅ Save merges (convert list[tuple[bytes, bytes]] -> list[tuple[str, str]])
    merges_json = [(t1.hex(), t2.hex()) for (t1, t2) in merges]
    with open("merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_json, f, indent=2)
    
    print(f"✅ Training complete! Final vocab size: {len(vocab):,} • Total merges: {len(merges):,}")
    print("📁 Files saved: vocab.json, merges.json")