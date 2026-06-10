#!/usr/bin/env python3
"""Fine-tune the local sentence encoder on SPECTER scientific citation triplets.

Why: the production retriever uses ``all-MiniLM-L6-v2`` (384-dim), a general-purpose
encoder. SPECTER triplets ``(anchor paper, cited paper, non-cited paper)`` teach the
encoder the *scientific* notion of relatedness — "these two papers cite each other"
— which is exactly the signal a literature-review retriever needs. We keep the same
384-dim output so the fine-tuned checkpoint is a **drop-in** replacement: no
pgvector schema change, no migration, just point ``EMBEDDING_MODEL`` at the output.

Objective: ``MultipleNegativesRankingLoss`` (in-batch InfoNCE) with the explicit
SPECTER negative as a hard negative. Reports held-out triplet accuracy — the share
of test triplets where ``cos(anchor, positive) > cos(anchor, negative)`` — for the
base model (before) and the fine-tuned model (after), so the gain is measured, not
asserted.

Run (in the GPU overlay venv):
    python training/finetune_encoder.py --max-train 80000 --epochs 1 --batch-size 96
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss

HERE = Path(__file__).resolve().parent
DEFAULT_TRIPLETS = HERE / "data" / "specter_train_triples.jsonl.gz"
DEFAULT_OUT = HERE.parent / "backend" / "models" / "minilm-specter-ft"


def load_triplets(path: Path, limit: int, seed: int) -> list[list[str]]:
    """Load up to ``limit`` ``[anchor, positive, negative]`` triplets, shuffled."""
    triplets: list[list[str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if isinstance(rec, list) and len(rec) == 3 and all(rec):
                triplets.append([rec[0].strip(), rec[1].strip(), rec[2].strip()])
    random.Random(seed).shuffle(triplets)
    return triplets[:limit]


def to_dataset(triplets: list[list[str]]) -> Dataset:
    """Convert triplets to a HF ``Dataset`` with anchor/positive/negative columns."""
    return Dataset.from_dict(
        {
            "anchor": [t[0] for t in triplets],
            "positive": [t[1] for t in triplets],
            "negative": [t[2] for t in triplets],
        }
    )


def build_evaluator(eval_triplets: list[list[str]], name: str) -> TripletEvaluator:
    """Triplet-accuracy evaluator over the held-out split."""
    return TripletEvaluator(
        anchors=[t[0] for t in eval_triplets],
        positives=[t[1] for t in eval_triplets],
        negatives=[t[2] for t in eval_triplets],
        name=name,
    )


def main() -> None:
    """Parse args, fine-tune the encoder, and report before/after triplet accuracy."""
    parser = argparse.ArgumentParser(description="Fine-tune MiniLM on SPECTER triplets")
    parser.add_argument("--triplets", type=Path, default=DEFAULT_TRIPLETS)
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-train", type=int, default=80000)
    parser.add_argument("--eval-size", type=int, default=3000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = args.max_train + args.eval_size
    print(f"Loading up to {total} triplets from {args.triplets} ...")
    triplets = load_triplets(args.triplets, total, args.seed)
    eval_triplets = triplets[: args.eval_size]
    train_triplets = triplets[args.eval_size :]
    print(f"  train={len(train_triplets)}  eval(held-out)={len(eval_triplets)}")

    train_ds = to_dataset(train_triplets)
    evaluator = build_evaluator(eval_triplets, "specter-heldout")

    print(f"Loading base model {args.base_model} ...")
    model = SentenceTransformer(args.base_model, device="cuda")
    model.max_seq_length = args.max_seq_length
    assert model.get_sentence_embedding_dimension() == 384, "expected 384-dim drop-in"

    before = evaluator(model)
    before_acc = before[f"{evaluator.name}_cosine_accuracy"]
    print(f"\nBEFORE fine-tune — held-out triplet accuracy: {before_acc:.4f}")

    loss = MultipleNegativesRankingLoss(model)
    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.out / "_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=train_args, train_dataset=train_ds, loss=loss, evaluator=evaluator
    )
    print("\nTraining ...")
    trainer.train()

    after = evaluator(model)
    after_acc = after[f"{evaluator.name}_cosine_accuracy"]

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))

    delta = after_acc - before_acc
    print("\n===== SPECTER held-out triplet accuracy =====")
    print(f"  base (all-MiniLM-L6-v2): {before_acc:.4f}")
    print(f"  fine-tuned             : {after_acc:.4f}")
    print(f"  delta                  : {delta:+.4f}")
    print(f"\nSaved fine-tuned 384-dim encoder to {args.out}")
    (args.out / "finetune_metrics.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "train_triplets": len(train_triplets),
                "eval_triplets": len(eval_triplets),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "triplet_accuracy_before": round(before_acc, 4),
                "triplet_accuracy_after": round(after_acc, 4),
                "triplet_accuracy_delta": round(delta, 4),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
