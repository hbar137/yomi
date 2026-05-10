"""Fine-tune cl-tohoku/bert-base-japanese-char-v3 with per-heteronym
classification heads on (sentence, span, reading) triples.

Each heteronym (a surface that appears with ≥2 distinct readings in the
corpus) gets its own classification head over its candidate readings. We pool
the BERT hidden states across the span tokens, project through that
heteronym's head, and minimize cross-entropy.

Reads:
    data/heteronyms.json          -- defines heads + class indices
    data/{train,val}.jsonl        -- training + validation examples

Writes:
    models/<run_name>/best/       -- best-val-accuracy checkpoint
    models/<run_name>/last/       -- last-epoch checkpoint
    models/<run_name>/heteronyms.json  -- frozen copy for inference

Run with:
    python -m yomi.train --run-name v1 --epochs 3 --batch-size 32

Designed to run on a single GPU (A40 / A100). Reasonable defaults for ~1M
training examples; tune via flags. Mixed precision is on by default — disable
with --no-fp16 when debugging.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

DEFAULT_BASE = "cl-tohoku/bert-base-japanese-char-v3"


# --- data --------------------------------------------------------------------

@dataclass
class Example:
    sentence: str
    span_start: int   # char index in sentence
    span_end: int
    surface: str
    reading: str
    work_id: str
    source: str


def load_jsonl(path: Path) -> list[Example]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(Example(
                sentence=d["sentence"],
                span_start=d["span_start"],
                span_end=d["span_end"],
                surface=d["surface"],
                reading=d["reading"],
                work_id=d["work_id"],
                source=d.get("source", "unknown"),
            ))
    return out


class HeteronymTable:
    """Maps surface -> ordered list of reading classes; reverse lookup gives
    the class index used as the cross-entropy target."""

    def __init__(self, table: dict[str, dict[str, int]]) -> None:
        # Stable order: descending by frequency, then lexicographic.
        self.readings: dict[str, list[str]] = {}
        for surface, counts in table.items():
            ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            self.readings[surface] = [r for r, _ in ordered]
        self._idx: dict[str, dict[str, int]] = {
            s: {r: i for i, r in enumerate(rs)}
            for s, rs in self.readings.items()
        }

    @classmethod
    def load(cls, path: Path) -> "HeteronymTable":
        with path.open(encoding="utf-8") as f:
            return cls(json.load(f))

    def reading_index(self, surface: str, reading: str) -> int:
        return self._idx[surface][reading]

    def n_readings(self, surface: str) -> int:
        return len(self.readings[surface])

    def surfaces(self) -> list[str]:
        return list(self.readings.keys())


# --- dataset / collator -----------------------------------------------------

class HeteronymDataset(Dataset):
    """Lazy-tokenized dataset. We tokenize on-the-fly so we keep the
    tokenizer in only one place; for ~1M examples each ~50 chars long the
    tokenizer overhead is small relative to the BERT forward pass."""

    def __init__(self, examples: list[Example], tokenizer, hetero: HeteronymTable,
                 max_length: int = 128):
        self.examples = [
            e for e in examples
            if e.surface in hetero.readings and e.reading in hetero._idx[e.surface]
        ]
        dropped = len(examples) - len(self.examples)
        if dropped:
            print(f"  dropped {dropped} examples whose surface/reading "
                  f"isn't in the heteronym table", file=sys.stderr)
        self.tokenizer = tokenizer
        self.hetero = hetero
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        e = self.examples[idx]
        enc = self.tokenizer(
            e.sentence,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        # cl-tohoku/bert-base-japanese-char-v3 is strictly character-level
        # (1 input char -> 1 token, [UNK] for unrepresentable chars). The
        # tokenizer is the slow Python implementation, which doesn't expose
        # return_offsets_mapping. We compute offsets manually: token at
        # index i corresponds to char at position (i - n_leading_special).
        # In practice [CLS] is the only leading special token.
        n_special_leading = 1  # [CLS]
        token_start = token_end = None
        n_tokens = len(enc["input_ids"])
        # Iterate non-special tokens (skip [CLS] at 0 and [SEP] at end).
        for i in range(n_special_leading, n_tokens - 1):
            char_pos = i - n_special_leading
            if token_start is None and char_pos == e.span_start:
                token_start = i
            if char_pos + 1 == e.span_end:
                token_end = i + 1
                break
        if token_start is None or token_end is None or token_end <= token_start:
            # Span got truncated out; fallback to first non-special token so
            # the batch shape stays valid. We mask these via `valid_mask` so
            # they don't contribute to loss.
            valid = False
            token_start, token_end = 1, 2
        else:
            valid = True
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_start": token_start,
            "token_end": token_end,
            "surface": e.surface,
            "label": self.hetero.reading_index(e.surface, e.reading),
            "valid": valid,
        }


def collate(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        attn[i, :L] = torch.tensor(b["attention_mask"], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "token_starts": torch.tensor([b["token_start"] for b in batch], dtype=torch.long),
        "token_ends": torch.tensor([b["token_end"] for b in batch], dtype=torch.long),
        "surfaces": [b["surface"] for b in batch],
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "valid": torch.tensor([b["valid"] for b in batch], dtype=torch.bool),
    }


# --- model -------------------------------------------------------------------

# nn.ModuleDict keys can't contain "." or be empty. Sanitize surfaces.
def safe_key(surface: str) -> str:
    # Keep things simple: hex encoding is unambiguous and reversible.
    return "h_" + surface.encode("utf-8").hex()


class YomiBert(nn.Module):
    def __init__(self, base_model_name: str, hetero: HeteronymTable):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        hidden = self.bert.config.hidden_size
        self.heads = nn.ModuleDict()
        for surface in hetero.surfaces():
            self.heads[safe_key(surface)] = nn.Linear(
                hidden, hetero.n_readings(surface),
            )
        self.hetero = hetero

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_starts: torch.Tensor,
        token_ends: torch.Tensor,
        surfaces: list[str],
        labels: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> dict:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # (B, L, H)

        # Pool span hidden states. Mean over tokens in [start, end). We loop
        # per-example because spans differ; for typical batch sizes (~32) this
        # is negligible vs the BERT forward.
        B = h.size(0)
        losses = []
        preds = []
        for i in range(B):
            s, e = token_starts[i].item(), token_ends[i].item()
            pooled = h[i, s:e].mean(dim=0)  # (H,)
            logits = self.heads[safe_key(surfaces[i])](pooled)  # (n_readings,)
            preds.append(int(logits.argmax(dim=0).item()))
            if labels is not None and (valid is None or bool(valid[i].item())):
                losses.append(F.cross_entropy(logits.unsqueeze(0),
                                              labels[i:i + 1]))

        result: dict = {"preds": preds}
        if losses:
            result["loss"] = torch.stack(losses).mean()
        else:
            result["loss"] = h.new_zeros(())
        return result


# --- training loop ----------------------------------------------------------

def evaluate(model: YomiBert, loader: DataLoader, device: torch.device,
             hetero: HeteronymTable) -> dict:
    model.eval()
    n_total = n_correct = 0
    per_surface_total: dict[str, int] = defaultdict(int)
    per_surface_correct: dict[str, int] = defaultdict(int)
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            tstarts = batch["token_starts"].to(device)
            tends = batch["token_ends"].to(device)
            labels = batch["labels"].to(device)
            valid = batch["valid"]

            out = model(input_ids, attn, tstarts, tends, batch["surfaces"])
            for i, p in enumerate(out["preds"]):
                if not bool(valid[i].item()):
                    continue
                surf = batch["surfaces"][i]
                n_total += 1
                per_surface_total[surf] += 1
                if p == int(labels[i].item()):
                    n_correct += 1
                    per_surface_correct[surf] += 1
    macro_accs = [
        per_surface_correct[s] / per_surface_total[s]
        for s in per_surface_total if per_surface_total[s] > 0
    ]
    return {
        "micro_acc": n_correct / max(n_total, 1),
        "macro_acc": sum(macro_accs) / max(len(macro_accs), 1),
        "n": n_total,
    }


def save_checkpoint(model: YomiBert, tokenizer, hetero_path: Path,
                    out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.bert.save_pretrained(out_dir / "bert")
    tokenizer.save_pretrained(out_dir / "bert")
    torch.save(model.heads.state_dict(), out_dir / "heads.pt")
    # Copy heteronyms.json next to the model so inference can find it.
    (out_dir / "heteronyms.json").write_text(
        hetero_path.read_text(encoding="utf-8"), encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--heteronyms", type=Path,
                    default=Path("data/heteronyms.json"))
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-pct", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every-frac", type=float, default=0.5,
                    help="run val every X fraction of an epoch (0 = only at end)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="cap total training steps (0 = no cap). Useful for "
                         "CPU smoke tests: --max-steps 20 verifies the data "
                         "loader, model build, and checkpoint write in ~1 min.")
    ap.add_argument("--wandb", action="store_true",
                    help="log to Weights & Biases. Requires WANDB_API_KEY in env.")
    ap.add_argument("--wandb-project", default="yomi")
    ap.add_argument("--eval-only", type=Path, default=None,
                    help="path to a checkpoint dir (e.g. models/v1/best) to "
                         "load and evaluate, then exit. Skips training.")
    ap.add_argument("--eval-set", choices=("val", "test"), default="val",
                    help="which split to use under --eval-only (default val)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", file=sys.stderr)

    # ---- eval-only short-circuit ------------------------------------------
    if args.eval_only is not None:
        ckpt = args.eval_only
        hetero = HeteronymTable.load(ckpt / "heteronyms.json")
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt / "bert"), use_fast=True)
        pad_id = tokenizer.pad_token_id
        eval_path = args.data_dir / f"{args.eval_set}.jsonl"
        eval_examples = load_jsonl(eval_path)
        eval_ds = HeteronymDataset(eval_examples, tokenizer, hetero, args.max_length)
        eval_dl = DataLoader(
            eval_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=lambda b: collate(b, pad_id),
        )
        model = YomiBert(str(ckpt / "bert"), hetero).to(device)
        model.heads.load_state_dict(torch.load(ckpt / "heads.pt", map_location=device))
        metrics = evaluate(model, eval_dl, device, hetero)
        print(f"\n[{args.eval_set} on {ckpt}] micro={metrics['micro_acc']:.4f} "
              f"macro={metrics['macro_acc']:.4f} n={metrics['n']}",
              file=sys.stderr)
        return

    hetero = HeteronymTable.load(args.heteronyms)
    print(f"heteronyms: {len(hetero.surfaces())} surfaces", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    pad_id = tokenizer.pad_token_id

    train_examples = load_jsonl(args.data_dir / "train.jsonl")
    val_examples = load_jsonl(args.data_dir / "val.jsonl")
    print(f"train: {len(train_examples)}  val: {len(val_examples)}", file=sys.stderr)

    train_ds = HeteronymDataset(train_examples, tokenizer, hetero, args.max_length)
    val_ds = HeteronymDataset(val_examples, tokenizer, hetero, args.max_length)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda b: collate(b, pad_id),
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda b: collate(b, pad_id),
    )

    model = YomiBert(args.base_model, hetero).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    if args.max_steps and args.max_steps < total_steps:
        total_steps = args.max_steps
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(total_steps * args.warmup_pct),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_fp16 and device.type == "cuda"))

    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    eval_every = max(int(len(train_dl) * args.eval_every_frac), 1) if args.eval_every_frac > 0 else 0

    wandb_run = None
    if args.wandb:
        import wandb  # local import — only needed when flag is set
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()},
        )

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_dl:
            step += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            tstarts = batch["token_starts"].to(device, non_blocking=True)
            tends = batch["token_ends"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            valid = batch["valid"]

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(input_ids, attn, tstarts, tends, batch["surfaces"],
                            labels=labels, valid=valid)
                loss = out["loss"]

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            if step % args.log_every == 0:
                avg = running_loss / args.log_every
                running_loss = 0.0
                rate = step / max(time.time() - t0, 1e-6)
                eta_sec = (total_steps - step) / max(rate, 1e-6)
                print(f"[ep{epoch} step {step}/{total_steps}] loss={avg:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} "
                      f"{rate:.1f}st/s eta={eta_sec/60:.1f}m", file=sys.stderr)
                if wandb_run is not None:
                    wandb_run.log({"train/loss": avg,
                                   "train/lr": scheduler.get_last_lr()[0],
                                   "train/step": step})

            if eval_every and step % eval_every == 0:
                metrics = evaluate(model, val_dl, device, hetero)
                print(f"  [val @ step {step}] micro={metrics['micro_acc']:.4f} "
                      f"macro={metrics['macro_acc']:.4f} n={metrics['n']}",
                      file=sys.stderr)
                if wandb_run is not None:
                    wandb_run.log({"val/micro_acc": metrics["micro_acc"],
                                   "val/macro_acc": metrics["macro_acc"],
                                   "val/step": step})
                if metrics["macro_acc"] > best_val:
                    best_val = metrics["macro_acc"]
                    save_checkpoint(model, tokenizer, args.heteronyms,
                                    run_dir / "best")
                    print(f"  saved best -> {run_dir / 'best'}", file=sys.stderr)
                model.train()

            if args.max_steps and step >= args.max_steps:
                break  # exit inner batch loop

        # If max_steps cap was hit, also break the epoch loop.
        if args.max_steps and step >= args.max_steps:
            break

        # End-of-epoch eval.
        metrics = evaluate(model, val_dl, device, hetero)
        print(f"== epoch {epoch} val: micro={metrics['micro_acc']:.4f} "
              f"macro={metrics['macro_acc']:.4f} n={metrics['n']} ==",
              file=sys.stderr)
        if wandb_run is not None:
            wandb_run.log({"val/micro_acc": metrics["micro_acc"],
                           "val/macro_acc": metrics["macro_acc"],
                           "val/epoch": epoch})
        if metrics["macro_acc"] > best_val:
            best_val = metrics["macro_acc"]
            save_checkpoint(model, tokenizer, args.heteronyms, run_dir / "best")

    save_checkpoint(model, tokenizer, args.heteronyms, run_dir / "last")
    print(f"\ndone. best val macro acc: {best_val:.4f}", file=sys.stderr)
    if wandb_run is not None:
        wandb_run.summary["best_val_macro_acc"] = best_val
        wandb_run.finish()


if __name__ == "__main__":
    main()
