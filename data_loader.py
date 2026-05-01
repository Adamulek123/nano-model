import torch
import sentencepiece as spm
from pathlib import Path


def is_power_of_two(x):
    return x > 0 and (x & (x - 1) == 0)


class FineWebShardData:
    def __init__(self, batch_size, block_size, device, shard_reload_interval=200):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.shard_reload_interval = shard_reload_interval

        self.script_dir = Path(__file__).resolve().parent
        self.fineweb_dir = self.script_dir / "fineweb-edu-100M"
        self.shards_dir = self.fineweb_dir / "text_shards"
        self.tokenizer_model_path = self.fineweb_dir / "slm_unigram_24k.model"

        if not self.shards_dir.exists():
            raise FileNotFoundError(
                f"Missing {self.shards_dir}. Run sharding_script.py first to create text shards."
            )
        if not self.tokenizer_model_path.exists():
            raise FileNotFoundError(
                f"Missing {self.tokenizer_model_path}. Run parquet_to_text.py first to create slm_unigram_24k.model."
            )

        self.sp = spm.SentencePieceProcessor(model_file=str(self.tokenizer_model_path))

        self.shard_paths = sorted(self.shards_dir.glob("*.txt"))
        if len(self.shard_paths) < 2:
            raise ValueError(
                f"Need at least 2 text shards in {self.shards_dir}, found {len(self.shard_paths)}"
            )

        self.n_train_shards = max(1, int(0.9 * len(self.shard_paths)))
        self.n_train_shards = min(self.n_train_shards, len(self.shard_paths) - 1)
        self.train_shards = self.shard_paths[: self.n_train_shards]
        self.val_shards = self.shard_paths[self.n_train_shards :]

        print(
            f"using {len(self.train_shards)} train shard(s) and {len(self.val_shards)} val shard(s) "
            f"from {self.shards_dir}"
        )

        self.shard_state = {
            "train": {"tokens": None, "batches_since_reload": shard_reload_interval},
            "val": {"tokens": None, "batches_since_reload": shard_reload_interval},
        }

        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text):
        return self.sp.encode_as_ids(text)

    def decode(self, ids):
        return self.sp.decode_ids(ids)

    def load_random_shard_tokens(self, split):
        shard_pool = self.train_shards if split == "train" else self.val_shards
        while True:
            shard_idx = torch.randint(len(shard_pool), (1,)).item()
            shard_path = shard_pool[shard_idx]
            text = shard_path.read_text(encoding="utf-8")
            token_ids = self.encode(text)
            if len(token_ids) > self.block_size + 1:
                return torch.tensor(token_ids, dtype=torch.long)

    def get_batch(self, split):
        state = self.shard_state[split]
        if (
            state["tokens"] is None
            or state["batches_since_reload"] >= self.shard_reload_interval
        ):
            state["tokens"] = self.load_random_shard_tokens(split)
            state["batches_since_reload"] = 0

        data = state["tokens"]
        state["batches_since_reload"] += 1

        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
