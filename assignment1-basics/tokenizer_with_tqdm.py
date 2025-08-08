#!/usr/bin/env python3
"""
Tokenizer script with tqdm progress bar
Run: pip install tqdm (or uv add tqdm)
"""

import numpy as np
import time
from cs336_basics.tokenizer import Tokenizer

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not installed. Install with: pip install tqdm")
    print("Falling back to basic progress logging...")
    tqdm = None


def count_lines(filepath):
    """Count total lines in file for progress tracking"""
    print("Counting total lines...")
    with open(filepath, 'r') as f:
        total = sum(1 for _ in f)
    return total


def tokenize_with_tqdm(input_path, output_path, tokenizer):
    """Tokenize file with tqdm progress bar"""
    
    # Count total lines
    total_lines = count_lines(input_path)
    print(f"Total lines to process: {total_lines:,}")
    
    ids = []
    start_time = time.time()
    
    with open(input_path, "r") as f:
        # Wrap the file iterator with tqdm
        if tqdm:
            pbar = tqdm(f, total=total_lines, desc="Tokenizing", 
                       unit="lines", unit_scale=True,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} lines [{elapsed}<{remaining}, {rate_fmt}]')
        else:
            pbar = f
            log_interval = max(1000, total_lines // 100)
            lines_processed = 0
        
        for line in pbar:
            if not line.strip():  # Skip empty lines
                continue
                
            ids.extend(tokenizer.encode(line))
            
            # Update tqdm description with token count
            if tqdm and len(ids) > 0:
                pbar.set_postfix({
                    'tokens': f"{len(ids):,}",
                    'avg_tok/line': f"{len(ids)/(pbar.n+1):.1f}"
                })
            
            # Fallback logging if no tqdm
            elif not tqdm:
                lines_processed += 1
                if lines_processed % log_interval == 0:
                    elapsed = time.time() - start_time
                    progress_pct = (lines_processed / total_lines) * 100
                    lines_per_sec = lines_processed / elapsed if elapsed > 0 else 0
                    print(f"Progress: {lines_processed:,}/{total_lines:,} lines ({progress_pct:.1f}%) | "
                          f"{len(ids):,} tokens | {lines_per_sec:.0f} lines/sec")
        
        if tqdm:
            pbar.close()
    
    # Final summary
    elapsed = time.time() - start_time
    final_tokens = len(ids)
    print(f"\nTokenization complete!")
    print(f"Total tokens: {final_tokens:,}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average: {final_tokens/elapsed:.0f} tokens/sec")
    
    # Save to file
    print("Converting to numpy array and saving...")
    ids = np.array(ids)
    np.save(output_path, ids)
    print(f"Saved to {output_path}")
    
    return ids


if __name__ == "__main__":
    # Load tokenizer
    tokenizer = Tokenizer.from_files("vocab.json", "merges.json", special_tokens=["<|endoftext|>"])
    
    input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    output_path = "../data/TinyStoriesV2-GPT4-train-tok.npy"
    
    # Run tokenization
    ids = tokenize_with_tqdm(input_path, output_path, tokenizer) 