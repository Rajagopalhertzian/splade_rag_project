#!/usr/bin/env python
"""
Fine-tune SPLADE on custom data.

Usage:
    python scripts/train_splade.py --train_data ./data/train.jsonl --output ./models/splade-finetuned
    python scripts/train_splade.py --config config/train_config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from splade_rag.encoder import SPLADEModel, SPLADEEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SPLADEDataset(Dataset):
    """Dataset for SPLADE contrastive training."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Expected format: {"query": "...", "positive": "...", "negatives": ["...", "..."]}
                if "query" in item and "positive" in item:
                    self.data.append(item)

        logger.info(f"Loaded {len(self.data)} training examples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        positive = item["positive"]
        negatives = item.get("negatives", [])

        # Tokenize
        query_enc = self.tokenizer(
            query, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        pos_enc = self.tokenizer(
            positive, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )

        neg_encs = []
        for neg in negatives[:7]:  # Max 7 negatives
            neg_enc = self.tokenizer(
                neg, padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )
            neg_encs.append(neg_enc)

        # Pad negatives to fixed number
        while len(neg_encs) < 7:
            neg_encs.append(query_enc)  # Duplicate query as placeholder

        return {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids": torch.stack([e["input_ids"].squeeze(0) for e in neg_encs]),
            "neg_attention_mask": torch.stack([e["attention_mask"].squeeze(0) for e in neg_encs]),
        }


def train_step(model, batch, optimizer, device, lambda_q=0.001, lambda_d=0.01):
    """Single training step."""
    model.train()

    # Move to device
    q_ids = batch["query_input_ids"].to(device)
    q_mask = batch["query_attention_mask"].to(device)
    p_ids = batch["pos_input_ids"].to(device)
    p_mask = batch["pos_attention_mask"].to(device)
    n_ids = batch["neg_input_ids"].to(device)  # [batch, num_neg, seq_len]
    n_mask = batch["neg_attention_mask"].to(device)

    batch_size = q_ids.shape[0]
    num_neg = n_ids.shape[1]

    # Flatten negatives for batch encoding
    n_ids_flat = n_ids.view(-1, n_ids.shape[-1])
    n_mask_flat = n_mask.view(-1, n_mask.shape[-1])

    # Encode all
    q_vec = model(q_ids, q_mask)
    p_vec = model(p_ids, p_mask)
    n_vec_flat = model(n_ids_flat, n_mask_flat)
    n_vec = n_vec_flat.view(batch_size, num_neg, -1)

    # Contrastive loss
    contrastive_loss = model.contrastive_loss(q_vec, p_vec, n_vec)

    # Sparsity loss
    sparsity_loss = model.flops_loss(q_vec, p_vec, lambda_q, lambda_d)

    total_loss = contrastive_loss + sparsity_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "contrastive_loss": contrastive_loss.item(),
        "sparsity_loss": sparsity_loss.item(),
        "query_terms": (q_vec > 0).float().sum(dim=1).mean().item(),
        "doc_terms": (p_vec > 0).float().sum(dim=1).mean().item()
    }


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_contrastive = 0
    total_sparsity = 0
    total_q_terms = 0
    total_d_terms = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            metrics = train_step(model, batch, None, device)
            total_loss += metrics["total_loss"]
            total_contrastive += metrics["contrastive_loss"]
            total_sparsity += metrics["sparsity_loss"]
            total_q_terms += metrics["query_terms"]
            total_d_terms += metrics["doc_terms"]
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "contrastive_loss": total_contrastive / num_batches,
        "sparsity_loss": total_sparsity / num_batches,
        "avg_query_terms": total_q_terms / num_batches,
        "avg_doc_terms": total_d_terms / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SPLADE model")
    parser.add_argument("--train_data", type=str, required=True, help="Training data (JSONL)")
    parser.add_argument("--val_data", type=str, help="Validation data (JSONL)")
    parser.add_argument("--output", type=str, required=True, help="Output model directory")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased", help="Base BERT model")
    parser.add_argument("--aggregation", type=str, default="max", choices=["max", "sum", "log1p_sum"])
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--lambda_q", type=float, default=0.001, help="Query FLOPS weight")
    parser.add_argument("--lambda_d", type=float, default=0.01, help="Document FLOPS weight")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Gradient accumulation steps")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.base_model)

    # Create datasets
    train_dataset = SPLADEDataset(args.train_data, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_loader = None
    if args.val_data:
        val_dataset = SPLADEDataset(args.val_data, tokenizer, args.max_length)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    model = SPLADEModel(
        model_name=args.base_model,
        aggregation=args.aggregation
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float('inf')

    logger.info(f"Starting training for {args.epochs} epochs ({total_steps} steps)")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_contrastive = 0
        epoch_sparsity = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(progress):
            metrics = train_step(model, batch, optimizer, device, args.lambda_q, args.lambda_d)

            epoch_loss += metrics["total_loss"]
            epoch_contrastive += metrics["contrastive_loss"]
            epoch_sparsity += metrics["sparsity_loss"]

            # Gradient accumulation
            if (step + 1) % args.gradient_accumulation == 0:
                scheduler.step()
                global_step += 1

            # Update progress bar
            progress.set_postfix({
                "loss": metrics["total_loss"],
                "cont": metrics["contrastive_loss"],
                "sparse": metrics["sparsity_loss"],
                "q_terms": metrics["query_terms"],
                "d_terms": metrics["doc_terms"]
            })

            # Evaluation
            if val_loader and global_step % args.eval_steps == 0 and global_step > 0:
                val_metrics = evaluate(model, val_loader, device)
                logger.info(f"Step {global_step} - Val Loss: {val_metrics['loss']:.4f}, "
                          f"Q-terms: {val_metrics['avg_query_terms']:.1f}, "
                          f"D-terms: {val_metrics['avg_doc_terms']:.1f}")

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    # Save best model
                    torch.save(model.state_dict(), output_path / "pytorch_model.bin")
                    tokenizer.save_pretrained(output_path)
                    logger.info(f"Saved best model to {output_path}")

            # Save checkpoint
            if global_step % args.save_steps == 0 and global_step > 0:
                ckpt_path = output_path / f"checkpoint-{global_step}"
                ckpt_path.mkdir(exist_ok=True)
                torch.save(model.state_dict(), ckpt_path / "pytorch_model.bin")
                tokenizer.save_pretrained(ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

        # End of epoch
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, "
                  f"Contrastive: {epoch_contrastive/len(train_loader):.4f}, "
                  f"Sparsity: {epoch_sparsity/len(train_loader):.4f}")

    # Save final model
    torch.save(model.state_dict(), output_path / "pytorch_model.bin")
    tokenizer.save_pretrained(output_path)
    logger.info(f"Training complete. Model saved to {output_path}")


if __name__ == "__main__":
    main()