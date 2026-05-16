import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import time
import csv
import statistics
from datetime import datetime
from pathlib import Path

from data_loader import FineWebShardData

# hyperparameters
batch_size = 32  #  paraller predictions
block_size = 64  # max context length for predictions
max_iters = 10000
eval_interval_procentage = 0.25
eval_interval = max_iters * eval_interval_procentage
adamw_lr, muon_lr = 3e-4, 8e-4
warmup_fraction = 0.02
min_lr_ratio = 0.1
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required. Install a CUDA/ROCm-enabled PyTorch build and rerun."
    )
device = torch.device("cuda")
eval_iters = 10
n_embd = 384
n_head = 8
n_kv_head = 2
n_layer = 6
moe_layer_set = {i for i in range(1, n_layer + 1) if i % 2 == 0}
dropout = 0.2
use_moe_recurrent = True
capacity_factor = 1.25
num_experts = 4
aux_loss_weight = 0.01
pre_layers = 2
post_layers = 2
xT_loops = 4
lora_rank = 4
lti_init_decay = 0.95
warmup_steps = max(1, int(math.ceil(max_iters * warmup_fraction)))
amp_dtype = torch.bfloat16
rope_base = 10000
# ------------

print(
    f"runtime: cuda_device={torch.cuda.get_device_name(device)}, autocast_dtype={amp_dtype}"
)


def is_power_of_two(x):
    return x > 0 and (x & (x - 1) == 0)


def synchronize_for_timing():
    """Synchronize device work before reading wall-clock timing."""
    torch.cuda.synchronize(device)


def measure_step_metrics(step_start_time, tokens_processed):
    """Return per-step latency in milliseconds and throughput in tokens/sec."""
    synchronize_for_timing()
    dt_seconds = time.perf_counter() - step_start_time
    dt_ms = dt_seconds * 1000.0
    tokens_per_second = (
        tokens_processed / dt_seconds if dt_seconds > 0 else float("inf")
    )
    return dt_ms, tokens_per_second


def build_rope_cache(seq_len, head_dim, base=10000, device=device, dtype=amp_dtype):
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE.")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
    positions = torch.arange(seq_len, device=device)
    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos()
    sin = angles.sin()
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


def apply_rope(x, cos, sin):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)


assert is_power_of_two(n_head), "n_head must be a power of 2"
assert is_power_of_two(n_kv_head), "n_kv_head must be a power of 2"
assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"
assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

torch.manual_seed(1337)
torch.set_float32_matmul_precision("high")

script_dir = Path(__file__).resolve().parent

data = FineWebShardData(batch_size=batch_size, block_size=block_size, device=device)
decode = data.decode
get_batch = data.get_batch
vocab_size = data.vocab_size


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class GroupedQueryAttention(nn.Module):
    """grouped-query attention with shared KV heads"""

    def __init__(self, num_query_heads, num_kv_heads, head_size):
        super().__init__()
        assert is_power_of_two(num_query_heads), "num_query_heads must be a power of 2"
        assert is_power_of_two(num_kv_heads), "num_kv_heads must be a power of 2"
        assert (
            num_query_heads % num_kv_heads == 0
        ), "num_query_heads must be divisible by num_kv_heads"
        assert head_size % 2 == 0, "head_size must be even for RoPE"

        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.query_heads_per_kv = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(n_embd, num_query_heads * head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, num_kv_heads * head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, num_kv_heads * head_size, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.rope_base = rope_base
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = (
            self.q_proj(x)
            .view(B, T, self.num_query_heads, self.head_size)
            .transpose(1, 2)
        )
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)

        if (
            self.rope_cos.numel() == 0
            or self.rope_cos.size(0) < T
            or self.rope_cos.device != q.device
            or self.rope_cos.dtype != q.dtype
        ):
            cos, sin = build_rope_cache(
                seq_len=T,
                head_dim=self.head_size,
                base=self.rope_base,
                device=q.device,
                dtype=q.dtype,
            )
            self.rope_cos = cos
            self.rope_sin = sin

        cos = self.rope_cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin[:T].unsqueeze(0).unsqueeze(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.query_heads_per_kv > 1:
            k = k.repeat_interleave(self.query_heads_per_kv, dim=1)
            v = v.repeat_interleave(self.query_heads_per_kv, dim=1)

        attn_dropout = self.attn_dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=attn_dropout,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, n_embd)
        out = self.resid_dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MoEFeedForward(nn.Module):
    """Mixture of Experts feedforward layer"""

    def __init__(self, n_embd, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(n_embd, num_experts)
        self.experts = nn.ModuleList([FeedFoward(n_embd) for _ in range(num_experts)])

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        x_flat = x.view(B * T, C)  # (tokens, C), tokens = B*T

        router_logits = self.router(x_flat)  # (tokens, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (tokens, E)

        top1_val, top1_idx = router_probs.max(dim=-1)  # (tokens,), (tokens,)
        top1_mask = F.one_hot(
            top1_idx, num_classes=self.num_experts
        ).float()  # (tokens, E)

        capacity = int(math.ceil(capacity_factor * (B * T) / self.num_experts))

        positions = torch.cumsum(top1_mask, dim=0) - 1  # (tokens, E)
        dispatch_mask = (positions < capacity) * top1_mask  # (tokens, E)

        importance = router_probs.mean(dim=0)  # (E,)
        load = dispatch_mask.mean(dim=0)  # (E,)
        aux_loss = (importance * load).sum() * self.num_experts  # scalar
        out_flat = torch.zeros_like(x_flat)  # (tokens, C)

        for e in range(self.num_experts):
            expert_mask = dispatch_mask[:, e].bool()  # (tokens,)
            if expert_mask.any():
                expert_in = x_flat[expert_mask]  # (tokens_e, C)
                expert_out = self.experts[e](expert_in)  # (tokens_e, C)
                out_flat[expert_mask] = expert_out * top1_val[expert_mask].unsqueeze(
                    -1
                )  # (tokens_e, C)

        out = out_flat.view(B, T, C)  # (B, T, C)
        return out, aux_loss


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, n_kv_head, use_moe):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.use_moe = use_moe
        head_size = n_embd // n_head
        self.sa = GroupedQueryAttention(n_head, n_kv_head, head_size)
        if self.use_moe:
            self.ffwd = MoEFeedForward(n_embd, num_experts=num_experts)
        else:
            self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        ffwd_out = self.ffwd(self.ln2(x))
        if self.use_moe:
            ffwd_out, aux_loss = ffwd_out
        else:
            aux_loss = 0.0
        x = x + ffwd_out
        return x, aux_loss
    
class BlockLoRA(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, use_moe):
        super().__init__()
        self.use_moe = use_moe
        head_size = n_embd // n_head
        self.sa = GroupedQueryAttentionLoRA(n_head, n_kv_head, head_size)
        if self.use_moe:
            self.ffwd = MoEFeedForwardLoRA(n_embd, num_experts=num_experts)
        else:
            self.ffwd = FeedForwardLoRA(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, adapter_id):
        x = x + self.sa(self.ln1(x), adapter_id)
        ffwd_out = self.ffwd(self.ln2(x), adapter_id)
        if self.use_moe:
            ffwd_out, aux_loss = ffwd_out
        else:
            aux_loss = 0.0
        x = x + ffwd_out
        return x, aux_loss
    
class RecurrentBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, use_moe):
        super().__init__()
        self.core = BlockLoRA(n_embd, n_head, n_kv_head, use_moe)
        self.lti = LTIinjection(n_embd)
        self.halt_proj = nn.Linear(n_embd, 1)
        self.max_loops = xT_loops  # ACT upper bound

    def forward(self, x):
        B, T, C = x.shape
        aux_total = 0.0

        p_t = x.new_zeros(B)             # cumulative halting probability
        n_updates = x.new_zeros(B)
        weighted_sum = x.new_zeros(B, T, C)

        for i in range(self.max_loops):
            x_new, aux = self.core(x, adapter_id=i)
            aux_total += aux

            # LTI injection
            lti_out = self.lti(x_new)
            x_new = x_new + lti_out

            # halting probability per sample
            p = torch.sigmoid(self.halt_proj(x_new.mean(dim=1))).squeeze(-1)  # [B]

            # ACT halting logic (correct)
            still_running = (p_t < 1.0).float()
            p = p * still_running

            remainder = 1.0 - p_t
            update = torch.min(p, remainder)

            weighted_sum = weighted_sum + update[:, None, None] * x_new
            p_t = p_t + update
            n_updates = n_updates + still_running

            if (p_t >= 1.0).all():
                break

            x = x_new

        return weighted_sum, aux_total
class LoRALinear(nn.Module):
    def __init__(self, in_n_embd, out_n_embd, r, num_loops, alpha=1.0, bias=False):
        super().__init__()
        self.base = nn.Linear(in_n_embd, out_n_embd, bias=bias) # frozen base weights 

        
        self.r = r # scaling factor for LoRA updates
        self.scale = alpha / r
        # LoRA parameters for each num_loops
        self.lora_A = nn.Parameter(torch.zeros(num_loops, r, in_n_embd)) # (num_loops, r, in_n_embd)
        self.lora_B = nn.Parameter(torch.zeros(num_loops, out_n_embd, r)) # (num_loops, out_n_embd, r)

        # Simple init: small random for A, zeros for B.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) # weight init that stabilizes training
        nn.init.zeros_(self.lora_B)

    def forward(self, x, adapter_id):
        out = self.base(x) # Wx, frozen weights

        # fall back to base behavior if no adapter_id is provided
        if adapter_id is None:
            return out

        # Select LoRA weights for this loop.
        A = self.lora_A[adapter_id]  # (r, in_n_embd)
        B = self.lora_B[adapter_id]  # (out_n_embd, r)

        delta = F.linear(F.linear(x, A), B) * self.scale # (B @ (A @ x)) * scale
        return out + delta # Wx + B @ A @ x

class GroupedQueryAttentionLoRA(nn.Module):
    def __init__(self, num_query_heads, num_kv_heads, head_size):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.query_heads_per_kv = num_query_heads // num_kv_heads

        self.q_proj = LoRALinear(n_embd, num_query_heads * head_size, r=lora_rank, num_loops=xT_loops)
        self.k_proj = LoRALinear(n_embd, num_kv_heads * head_size, r=lora_rank, num_loops=xT_loops)
        self.v_proj = LoRALinear(n_embd, num_kv_heads * head_size, r=lora_rank, num_loops=xT_loops)
        self.proj = LoRALinear(n_embd, n_embd, r=lora_rank, num_loops=xT_loops)
        self.attn_dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.rope_base = rope_base
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

    def forward(self, x, adapter_id):
        B, T, _ = x.shape
        q = self.q_proj(x, adapter_id).view(B, T, self.num_query_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(x, adapter_id).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(x, adapter_id).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)

        if (
            self.rope_cos.numel() == 0
            or self.rope_cos.size(0) < T
            or self.rope_cos.device != q.device
            or self.rope_cos.dtype != q.dtype
        ):
            cos, sin = build_rope_cache(
                seq_len=T,
                head_dim=self.head_size,
                base=self.rope_base,
                device=q.device,
                dtype=q.dtype,
            )
            self.rope_cos = cos
            self.rope_sin = sin

        cos = self.rope_cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin[:T].unsqueeze(0).unsqueeze(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.query_heads_per_kv > 1:
            k = k.repeat_interleave(self.query_heads_per_kv, dim=1)
            v = v.repeat_interleave(self.query_heads_per_kv, dim=1)

        attn_dropout = self.attn_dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=attn_dropout,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, n_embd)
        out = self.resid_dropout(self.proj(out, adapter_id))
        return out
    
class FeedForwardLoRA(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            LoRALinear(n_embd, 4 * n_embd, r=lora_rank, num_loops=xT_loops),
            nn.ReLU(),
            LoRALinear(4 * n_embd, n_embd, r=lora_rank, num_loops=xT_loops),
            nn.Dropout(dropout),
        )

    def forward(self, x, adapter_id):
        for layer in self.net:
            if isinstance(layer, LoRALinear):
                x = layer(x, adapter_id)
            else:
                x = layer(x)
        return x

class MoEFeedForwardLoRA(nn.Module):
    def __init__(self, n_embd, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(n_embd, num_experts)
        self.experts = nn.ModuleList([FeedForwardLoRA(n_embd) for _ in range(num_experts)])

    def forward(self, x, adapter_id):
        # x: (B, T, C)
        B, T, C = x.shape
        x_flat = x.view(B * T, C)  # (tokens, C), tokens = B*T

        router_logits = self.router(x_flat)  # (tokens, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (tokens, E)

        top1_val, top1_idx = router_probs.max(dim=-1)  # (tokens,), (tokens,)
        top1_mask = F.one_hot(
            top1_idx, num_classes=self.num_experts
        ).float()  # (tokens, E)

        capacity = int(math.ceil(capacity_factor * (B * T) / self.num_experts))

        positions = torch.cumsum(top1_mask, dim=0) - 1  # (tokens, E)
        dispatch_mask = (positions < capacity) * top1_mask  # (tokens, E)

        importance = router_probs.mean(dim=0)  # (E,)
        load = dispatch_mask.mean(dim=0)  # (E,)
        aux_loss = (importance * load).sum() * self.num_experts  # scalar
        out_flat = torch.zeros_like(x_flat)  # (tokens, C)

        for e in range(self.num_experts):
            expert_mask = dispatch_mask[:, e].bool()  # (tokens,)
            if expert_mask.any():
                expert_in = x_flat[expert_mask]  # (tokens_e, C)
                expert_out = self.experts[e](expert_in, adapter_id)  # (tokens_e, C)
                out_flat[expert_mask] = expert_out * top1_val[expert_mask].unsqueeze(
                    -1
                )  # (tokens_e, C)

        out = out_flat.view(B, T, C)  # (B, T, C)
        return out, aux_loss 


def lti_scan_loop(x, a, B,h0=None):
    Bsize, T, C = x.shape
    H = a.numel()

    if h0 is None:
        h = x.new_zeros(Bsize, H)
    else:
        h = h0

    outputs = []
    for t in range(T):
        x_t = x[:, t, :]          # [B, C]
        h = h * a + x_t @ B.T     # [B, H]
        outputs.append(h)

    return torch.stack(outputs, dim=1)  # [B, T, H]

class LTIinjection(nn.Module):
    def __init__(self, n_embd,init_decay = lti_init_decay):
        super().__init__()
        raw_init = math.log(init_decay / (1 - init_decay))
        self.raw_a = nn.Parameter(torch.full((n_embd,), raw_init))
        self.B = nn.Parameter(torch.randn(n_embd, n_embd) * 0.01) # (n_embd, n_embd) * decay 
    def forward(self, x, h0=None):
        a = torch.sigmoid(self.raw_a) # (n_embd,) in (0, 1)
        return lti_scan_loop(x, a, self.B, h0=h0) # (B, T, n_embd)


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList()
        for i in range(pre_layers):
            self.blocks.append(Block(n_embd, n_head=n_head, n_kv_head=n_kv_head, use_moe=False))
        self.blocks.append(RecurrentBlock(n_embd, n_head=n_head, n_kv_head=n_kv_head, use_moe=use_moe_recurrent))
        for k in range(post_layers):
            self.blocks.append(Block(n_embd, n_head=n_head, n_kv_head=n_kv_head, use_moe=False))
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        x = tok_emb  # (B,T,C)
        aux_loss_total = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)  # (B,T,C)
            aux_loss_total += aux_loss
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            loss = loss + aux_loss_total * aux_loss_weight
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel().to(device)
model = torch.compile(model)
m = model


def should_use_muon(name, param):
    if param.ndim != 2:
        return False
    if "ffwd" not in name:
        return False
    skip_names = (
        "token_embedding_table",
        "lm_head",
        "embedding",
        "ln",
        "layernorm",
        "lora_",
        "router",
    )
    return not any(skip in name for skip in skip_names)


def partition_optimizer_params(model):
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if should_use_muon(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def get_lr_scale(step, warmup_steps, total_steps, min_lr_ratio):
    if total_steps <= warmup_steps:
        return 1.0
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def apply_lr(optimizer, base_lr, scale):
    lr = base_lr * scale
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# create PyTorch optimizers
muon_params, adamw_params = partition_optimizer_params(model)

muon_optimizer = torch.optim.Muon(
    muon_params,
    lr=muon_lr,
    ns_steps=2,
    weight_decay=0.1,
    momentum=0.95,
    nesterov=True,
    adjust_lr_fn="match_rms_adamw",
)
adamw_optimizer = torch.optim.AdamW(
    adamw_params,
    lr=adamw_lr,
    weight_decay=0.0,
    foreach=False,
)

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
arch_mode = "mha" if n_kv_head == n_head else "gqa"
lr_tag = f"{adamw_lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
run_name = (
    f"{arch_mode}_nh{n_head}_nkv{n_kv_head}_nl{n_layer}_ne{n_embd}_"
    f"bs{batch_size}_ctx{block_size}_lr{lr_tag}_it{max_iters}"
)

runs_dir = script_dir / "runs"
raw_runs_dir = runs_dir / "raw"
summary_runs_dir = runs_dir / "summary"
raw_runs_dir.mkdir(parents=True, exist_ok=True)
summary_runs_dir.mkdir(parents=True, exist_ok=True)

raw_log_path = raw_runs_dir / f"{run_timestamp}_{run_name}.csv"
summary_path = summary_runs_dir / f"{run_timestamp}_{run_name}_summary.txt"

fieldnames = [
    "step",
    "wall_time_s",
    "dt_ms",
    "tokens_per_s",
    "train_loss",
    "val_loss",
    "tokens_step",
    "tokens_seen",
    "warmup",
    "run_name",
]

tokens_seen = 0
steady_dt_ms = []
steady_tokens_per_s = []
run_start_time = time.perf_counter()
last_eval_train_loss = None
last_eval_val_loss = None

with raw_log_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        eval_train_loss = None
        eval_val_loss = None
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            eval_train_loss = float(losses["train"])
            eval_val_loss = float(losses["val"])
            last_eval_train_loss = eval_train_loss
            last_eval_val_loss = eval_val_loss
            print(
                f"step {iter}: train loss {eval_train_loss:.4f}, val loss {eval_val_loss:.4f}"
            )

        # sample a batch of data
        xb, yb = get_batch("train")

        synchronize_for_timing()
        step_start_time = time.perf_counter()

        # evaluate the los
        lr_scale = get_lr_scale(iter, warmup_steps, max_iters, min_lr_ratio)
        apply_lr(muon_optimizer, muon_lr, lr_scale)
        apply_lr(adamw_optimizer, adamw_lr, lr_scale)
        muon_optimizer.zero_grad(set_to_none=True)
        adamw_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            logits, loss = model(xb, yb)

        loss.backward()
        muon_optimizer.step()
        adamw_optimizer.step()

        tokens_step = yb.numel()
        tokens_seen += tokens_step
        dt_ms, tokens_per_second = measure_step_metrics(step_start_time, tokens_step)

        is_warmup = iter < warmup_steps
        if not is_warmup:
            steady_dt_ms.append(dt_ms)
            steady_tokens_per_s.append(tokens_per_second)

        wall_time_s = time.perf_counter() - run_start_time
        writer.writerow(
            {
                "step": iter,
                "wall_time_s": f"{wall_time_s:.6f}",
                "dt_ms": f"{dt_ms:.6f}",
                "tokens_per_s": f"{tokens_per_second:.6f}",
                "train_loss": f"{loss.item():.6f}",
                "val_loss": f"{eval_val_loss:.6f}" if eval_val_loss is not None else "",
                "tokens_step": tokens_step,
                "tokens_seen": tokens_seen,
                "warmup": int(is_warmup),
                "run_name": run_name,
            }
        )
        csv_file.flush()

        print(f"step {iter}: dt {dt_ms:.2f} ms | tok/s {tokens_per_second:,.0f}")

if steady_tokens_per_s:
    mean_tokens_per_s = statistics.mean(steady_tokens_per_s)
    median_dt_ms = statistics.median(steady_dt_ms)
    mean_tokens_per_s_text = f"{mean_tokens_per_s:.6f}"
    median_dt_ms_text = f"{median_dt_ms:.6f}"
else:
    mean_tokens_per_s_text = "nan"
    median_dt_ms_text = "nan"

summary_lines = [
    f"run_name={run_name}",
    f"timestamp={run_timestamp}",
    f"arch_mode={arch_mode}",
    f"n_head={n_head}",
    f"n_kv_head={n_kv_head}",
    f"n_layer={n_layer}",
    f"n_embd={n_embd}",
    f"batch_size={batch_size}",
    f"block_size={block_size}",
    f"adamw_lr={adamw_lr}",
    f"muon_lr={muon_lr}",
    f"max_iters={max_iters}",
    f"warmup_fraction={warmup_fraction}",
    f"warmup_steps={warmup_steps}",
    f"min_lr_ratio={min_lr_ratio}",
    f"autocast_dtype={amp_dtype}",
    f"torch_compile=enabled",
    f"tokens_seen={tokens_seen}",
    f"mean_tokens_per_s_excluding_warmup={mean_tokens_per_s_text}",
    f"median_dt_ms_excluding_warmup={median_dt_ms_text}",
    f"last_eval_train_loss={last_eval_train_loss if last_eval_train_loss is not None else 'nan'}",
    f"last_eval_val_loss={last_eval_val_loss if last_eval_val_loss is not None else 'nan'}",
    f"raw_log_path={raw_log_path}",
]
summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print(f"saved run log: {raw_log_path}")
print(f"saved run summary: {summary_path}")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
model.eval()
with torch.no_grad():
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
