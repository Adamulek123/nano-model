import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm

import time
import csv
import statistics
from datetime import datetime
from pathlib import Path
# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 100
eval_interval = 50
learning_rate = 3e-4
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required. Install a CUDA/ROCm-enabled PyTorch build and rerun.")
device = torch.device("cuda")
eval_iters = 50
n_embd = 384
n_head = 8
n_kv_head = 2
n_layer = 6
dropout = 0.2
warmup_steps = 5 # exclude startup transients from summary stats
amp_dtype = torch.bfloat16
# ------------

print(f"runtime: cuda_device={torch.cuda.get_device_name(device)}, autocast_dtype={amp_dtype}")

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
    tokens_per_second = tokens_processed / dt_seconds if dt_seconds > 0 else float('inf')
    return dt_ms, tokens_per_second

assert is_power_of_two(n_head), "n_head must be a power of 2"
assert is_power_of_two(n_kv_head), "n_kv_head must be a power of 2"
assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"
assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

torch.manual_seed(1337)
torch.set_float32_matmul_precision('high')

script_dir = Path(__file__).resolve().parent
fineweb_dir = script_dir / "fineweb-edu-100M"
spm_input_path = fineweb_dir / "spm_input.txt"
tokenizer_model_path = fineweb_dir / "slm_unigram_24k.model"

if not spm_input_path.exists():
    raise FileNotFoundError(
        f"Missing {spm_input_path}. Run parquet_to_text.py first to create spm_input.txt."
    )
if not tokenizer_model_path.exists():
    raise FileNotFoundError(
        f"Missing {tokenizer_model_path}. Run parquet_to_text.py first to create slm_unigram_24k.model."
    )

with spm_input_path.open("r", encoding="utf-8") as f:
    text = f.read()

sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model_path))
def encode(s):
    return sp.encode_as_ids(s)

def decode(ids):
    return sp.decode_ids(ids)

vocab_size = sp.get_piece_size()

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
print(f"train has {len(train_data)} tokens, val has {len(val_data)} tokens")
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
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
    """ grouped-query attention with shared KV heads """

    def __init__(self, num_query_heads, num_kv_heads, head_size):
        super().__init__()
        assert is_power_of_two(num_query_heads), "num_query_heads must be a power of 2"
        assert is_power_of_two(num_kv_heads), "num_kv_heads must be a power of 2"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

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

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_query_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)
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
    """ a simple linear layer followed by a non-linearity """

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

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, n_kv_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = GroupedQueryAttention(n_head, n_kv_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, n_kv_head=n_kv_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel().to(device)
model = torch.compile(model)
m = model


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, foreach=False)

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
arch_mode = "mha" if n_kv_head == n_head else "gqa"
lr_tag = f"{learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")
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
            eval_train_loss = float(losses['train'])
            eval_val_loss = float(losses['val'])
            last_eval_train_loss = eval_train_loss
            last_eval_val_loss = eval_val_loss
            print(f"step {iter}: train loss {eval_train_loss:.4f}, val loss {eval_val_loss:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        synchronize_for_timing()
        step_start_time = time.perf_counter()

        # evaluate the loss
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            logits, loss = model(xb, yb)

        loss.backward()
        optimizer.step()

        tokens_step = yb.numel()
        tokens_seen += tokens_step
        dt_ms, tokens_per_second = measure_step_metrics(step_start_time, tokens_step)

        is_warmup = iter < warmup_steps
        if not is_warmup:
            steady_dt_ms.append(dt_ms)
            steady_tokens_per_s.append(tokens_per_second)

        wall_time_s = time.perf_counter() - run_start_time
        writer.writerow({
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
        })
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
    f"learning_rate={learning_rate}",
    f"max_iters={max_iters}",
    f"warmup_steps={warmup_steps}",
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
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
