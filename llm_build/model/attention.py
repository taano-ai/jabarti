import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Single-head causal attention: x -> q,k,v -> scores -> mask -> softmax -> mix v."""

    def __init__(self, d_model: int):
        super().__init__()
        # 3 learned matrices shared by all tokens (your q_w, k_w, v_w).
        # bias=False: attention projections usually have none.
        self.Wq = nn.Linear(d_model, d_model, bias=False)  # what each token looks for
        self.Wk = nn.Linear(d_model, d_model, bias=False)  # what each token offers
        self.Wv = nn.Linear(d_model, d_model, bias=False)  # content each token hands over

    def forward(self, x):
        # x: (batch, seq_len, d_model)

        # STEP 1: same x through three different layers -> each (batch, seq_len, d_model)
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # STEP 2: scores[i][j] = how much token i's query matches token j's key
        # k.transpose(-2, -1) swaps the last two dims (k.T only works for 2D)
        # (batch, seq_len, d) @ (batch, d, seq_len) -> (batch, seq_len, seq_len)
        d = q.shape[-1]
        scores = (q @ k.transpose(-2, -1)) / d ** 0.5   # scaled by sqrt(d)

        # STEP 3: causal mask, block the future (above the diagonal) with -inf
        n = scores.shape[-1]   # seq_len
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        # STEP 4: softmax per row -> positive weights summing to 1
        weights = torch.softmax(scores, dim=-1)

        # STEP 5: each token's new row = weighted mix of value rows
        # (batch, seq_len, seq_len) @ (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return weights @ v