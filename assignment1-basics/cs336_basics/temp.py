import torch, numpy as np
from cs336_basics.main_modeling import TransformerLanguageModel, load_checkpoint, AdamW
from tokenizer import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer.from_files("vocab.json","merges.json",["<|endoftext|>"])

print(f"len(tok.vocab: {len(tok.vocab)}")
model = TransformerLanguageModel(
    d_model=256, num_heads=8, d_ff=1024, max_seq_len=256, theta=10000,
    vocab_size=len(tok.vocab), context_length=128, num_layers=4
).to(device)

optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

iteration = load_checkpoint("../checkpoints/model.pt", model, optimizer=optimizer)
model.eval()

ids = torch.tensor([[tok.encode("Ayush Sharma is a ")[0]]], device=device)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=50)

print("Iteration:", iteration)
print(tok.decode(out[0].tolist()))
