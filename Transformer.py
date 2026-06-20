import torch
import torch.nn as nn
import math


# ──────────────────────────────────────────────
# 1. Scaled Dot-Product Attention
# ──────────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V : (batch, heads, seq_len, d_k)
    mask    : (1, 1, seq_len, seq_len)  — optional causal mask
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, T, T)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = torch.softmax(scores, dim=-1)                          # (B, H, T, T)
    output  = torch.matmul(weights, V)                               # (B, H, T, d_k)
    return output


# ──────────────────────────────────────────────
# 2. Multi-Head Attention
# ──────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # x : (B, T, d_model) → (B, H, T, d_k)
        B, T, _ = x.size()
        x = x.view(B, T, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_Q(x))
        K = self.split_heads(self.W_K(x))
        V = self.split_heads(self.W_V(x))

        attn = scaled_dot_product_attention(Q, K, V, mask)   # (B, H, T, d_k)

        # Merge heads back → (B, T, d_model)
        B, H, T, d_k = attn.size()
        attn = attn.transpose(1, 2).contiguous().view(B, T, H * d_k)

        return self.W_O(attn)


# ──────────────────────────────────────────────
# 3. Position-wise Feed-Forward Network
# ──────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# 4. Sinusoidal Positional Encoding
# ──────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)                        # (max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()       # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer('pe', pe.unsqueeze(0))               # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ──────────────────────────────────────────────
# 5. Transformer Block
# ──────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One Transformer block:
        x → MultiHeadAttention → Add & Norm → FeedForward → Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, num_heads)
        self.ff    = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))   # attention + residual + norm
        x = self.norm2(x + self.ff(x))            # FFN + residual + norm
        return x


# ──────────────────────────────────────────────
# 6. Full Transformer (stacked blocks)
# ──────────────────────────────────────────────

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=512,
                 num_layers=4, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len)
        self.blocks    = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, vocab_size)

    def causal_mask(self, seq_len, device):
        # Lower-triangular mask: token i can only attend to positions ≤ i
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)                  # (1, 1, T, T)

    def forward(self, x):
        # x : (B, T)  — token indices
        mask = self.causal_mask(x.size(1), x.device)
        x = self.pos_enc(self.embedding(x))                   # (B, T, d_model)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(x)                                    # (B, T, vocab_size)



if __name__ == "__main__":
    VOCAB_SIZE = 1000
    BATCH_SIZE = 2
    SEQ_LEN    = 16

    model  = Transformer(vocab_size=VOCAB_SIZE, d_model=128, num_heads=4,
                         d_ff=512, num_layers=4)
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    logits = model(tokens)

    print(f"Input  shape : {tokens.shape}")     # (2, 16)
    print(f"Output shape : {logits.shape}")     # (2, 16, 1000)
    print(f"Total params : {sum(p.numel() for p in model.parameters()):,}")