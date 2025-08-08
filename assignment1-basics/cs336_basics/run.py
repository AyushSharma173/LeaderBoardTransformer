
def FLOP_CALCULATIONS():
    # ----- Config you can tweak -----
    S = 1024
    vocab_size = 50257
    models = {
        "gpt2-small":  dict(L=12,  d_model=768,  n_heads=12),
        "gpt2-medium": dict(L=24,  d_model=1024, n_heads=16),
        "gpt2-large":  dict(L=36,  d_model=1280, n_heads=20),
    }
    use_swiglu = True          # matches your code: 3 FFN mats
    d_ff_scale = 6400/1600     # keep same ratio as XL (=> 8/3). d_ff = d_model * d_ff_scale

    # ----- Helpers -----
    def mm(m,n,p): return 2*m*n*p

    def flops_breakdown(S, d_model, n_heads, d_ff, L, vocab):
        d_k = d_model // n_heads
        # projections
        mhsa_proj = 4 * mm(S, d_model, d_model)
        # attention matmuls
        attn_scores = 2 * S * S * d_k * n_heads
        attn_value  = 2 * S * S * d_k * n_heads
        attn_mats   = attn_scores + attn_value
        # FFN
        ffn = 3 * mm(S, d_model, d_ff) if use_swiglu else 2 * mm(S, d_model, d_ff)
        # norms
        rms_per = 2 * S * d_model
        layer_rms = 2 * rms_per
        # totals
        per_layer = mhsa_proj + attn_mats + ffn + layer_rms
        final_rms = 2 * S * d_model
        lm_head   = mm(S, d_model, vocab)  # untied head
        total = L * per_layer + final_rms + lm_head
        parts = {
            "MHSA_proj":   L * mhsa_proj,
            "Attn_matmuls":L * attn_mats,
            "FFN":         L * ffn,
            "Layer_RMS":   L * layer_rms,
            "Final_RMS":   final_rms,
            "LM_head":     lm_head,
        }
        pct = {k: v / total * 100 for k,v in parts.items()}
        return total, parts, pct

    for name, cfg in models.items():
        d_model = cfg["d_model"]; L = cfg["L"]; n_heads = cfg["n_heads"]
        d_ff = int(d_model * d_ff_scale)
        total, parts, pct = flops_breakdown(S, d_model, n_heads, d_ff, L, vocab_size)

        print(f"\n{name.upper()}")
        print(f"Total FLOPs: {total:,.0f}")
        for k in parts:
            print(f"{k:12s}: {parts[k]:,}  ({pct[k]:5.2f}%)")





from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


def ascii_plot(values, height=20, width=80):
    min_val, max_val = min(values), max(values)
    scaled = [
        int((val - min_val) / (max_val - min_val + 1e-8) * (height - 1)) 
        for val in values
    ]
    canvas = [[" " for _ in range(len(values))] for _ in range(height)]
    for i, val in enumerate(scaled):
        canvas[height - 1 - val][i] = "*"
    for row in canvas:
        print("".join(row))
    print(f"Min: {min_val:.4f}   Max: {max_val:.4f}")


def training_sgd_debug(lr=1.0, steps=10):
    print(f"\n--- Learning Rate: {lr} ---")
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    losses = []

    for t in range(steps):
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.item())
        print(f"Step {t}: Loss = {loss.item():.4f}")
        loss.backward()
        opt.step()

    ascii_plot(losses)




if __name__ == "__main__":
    training_sgd_debug(lr=1e0, steps=10)
    training_sgd_debug(lr=1e1, steps=10)
    training_sgd_debug(lr=1e2, steps=10)
    training_sgd_debug(lr=1e3, steps=10)





if __name__ == "__main__":
    training_sgd_debug()



