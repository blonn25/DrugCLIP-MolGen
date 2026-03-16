#!/usr/bin/env python3
"""Training entrypoint for embedding-to-SELFIES transformer decoder.

This script loads molecule embeddings and tokenized SELFIES targets from H5,
trains with teacher forcing and cross-entropy, and writes checkpoints plus
training artifacts (history, losses, and optional plot).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from selfies_decoder import EmbeddingToSelfiesDecoder, SelfiesTokenizer


def save_loss_plot(output_path: Path, train_losses: np.ndarray, val_losses: np.ndarray) -> bool:
    """Save train/validation loss curves to PNG.

    Args:
        output_path: Destination PNG path.
        train_losses: Epoch-wise training losses.
        val_losses: Epoch-wise validation losses.

    Returns:
        True if figure is saved, False if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Warning: matplotlib is not installed; skipping loss plot generation. "
            "Install with: conda run -n drugclip python -m pip install matplotlib"
        )
        return False

    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train_loss", linewidth=2)
    ax.plot(epochs, val_losses, label="val_loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Decoder Training Progress")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


class MolToSelfiesDataset(Dataset):
    """Lazy H5-backed dataset pairing molecule embeddings with token IDs."""

    def __init__(self, embedding_h5: Path, token_h5: Path) -> None:
        """Open metadata and validate row alignment across both H5 files.

        Args:
            embedding_h5: H5 file containing `mol_reps`.
            token_h5: H5 file containing `token_ids` and `vocab`.
        """
        self.embedding_h5_path = str(embedding_h5)
        self.token_h5_path = str(token_h5)
        self._emb_f = None
        self._tok_f = None
        self._emb_ds = None
        self._tok_ds = None

        with h5py.File(self.embedding_h5_path, "r") as emb_f:
            self.n_rows = int(emb_f["mol_reps"].shape[0])
            self.embedding_dim = int(emb_f["mol_reps"].shape[1])

        with h5py.File(self.token_h5_path, "r") as tok_f:
            self.n_rows_tokens = int(tok_f["token_ids"].shape[0])
            self.seq_len = int(tok_f["token_ids"].shape[1])
            self.base_vocab = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in tok_f["vocab"][:]]

        if self.n_rows != self.n_rows_tokens:
            raise ValueError(
                f"Row mismatch: embeddings={self.n_rows}, token_ids={self.n_rows_tokens}."
            )

    def __len__(self) -> int:
        """Return number of aligned training rows."""
        return self.n_rows

    def _lazy_open(self) -> None:
        """Open H5 handles lazily per dataloader worker process."""
        if self._emb_f is None:
            self._emb_f = h5py.File(self.embedding_h5_path, "r")
            self._emb_ds = self._emb_f["mol_reps"]
        if self._tok_f is None:
            self._tok_f = h5py.File(self.token_h5_path, "r")
            self._tok_ds = self._tok_f["token_ids"]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch one (embedding, token_ids) pair as numpy arrays.

        Args:
            idx: Row index.

        Returns:
            Tuple `(mol_embedding, token_ids)`.
        """
        self._lazy_open()
        mol = self._emb_ds[idx].astype(np.float32)
        token_ids = self._tok_ds[idx].astype(np.int64)

        return mol, token_ids


def build_decoder_inputs(
    token_ids: torch.Tensor,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build teacher-forcing inputs and next-token targets.

    Decoder input format: `[BOS] + y_0..y_{T-1}`
    Target format: `y_0..y_{T-1} + [EOS]`

    Args:
        token_ids: Padded token IDs of shape (B, T).
        pad_id: Padding token ID.
        bos_id: Beginning-of-sequence token ID.
        eos_id: End-of-sequence token ID.

    Returns:
        Tuple `(decoder_inputs, targets)` each with shape (B, T+1).
    """
    bsz, seq_len = token_ids.shape
    device = token_ids.device

    decoder_in = torch.full((bsz, seq_len + 1), pad_id, dtype=torch.long, device=device)
    targets = torch.full((bsz, seq_len + 1), pad_id, dtype=torch.long, device=device)

    decoder_in[:, 0] = bos_id

    for i in range(bsz):
        row = token_ids[i]
        # Determine true (non-pad) token length for this sample.
        non_pad = (row != pad_id).nonzero(as_tuple=False).flatten()
        if non_pad.numel() == 0:
            true_len = 0
        else:
            true_len = int(non_pad[-1].item()) + 1

        if true_len > 0:
            # Shift right for decoder inputs; keep original for prediction targets.
            decoder_in[i, 1 : true_len + 1] = row[:true_len]
            targets[i, :true_len] = row[:true_len]
        # Supervise explicit stop at end of non-pad region.
        targets[i, true_len] = eos_id

    return decoder_in, targets


def collate_fn(tokenizer: SelfiesTokenizer):
    """Create collate function that packs batch tensors for training.

    Args:
        tokenizer: Tokenizer providing special token IDs.

    Returns:
        Callable that collates dataset samples into model-ready tensors.
    """

    def _collate(batch):
        """Collate list of numpy samples into torch tensors.

        Args:
            batch: List of `(embedding, token_ids)` numpy tuples.

        Returns:
            Tuple `(mols, decoder_inputs, targets)` as torch tensors.
        """
        mols = torch.from_numpy(np.stack([x[0] for x in batch], axis=0)).float()
        token_ids = torch.from_numpy(np.stack([x[1] for x in batch], axis=0)).long()
        decoder_in, targets = build_decoder_inputs(
            token_ids,
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
        )
        return mols, decoder_in, targets

    return _collate


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    pad_id: int,
    epoch: int,
    mode: str,
    log_every: int,
) -> Dict[str, float]:
    """Run one train or validation epoch and return aggregate metrics.

    Args:
        model: Decoder model.
        loader: DataLoader for current split.
        optimizer: Optimizer for training mode, or None for validation.
        device: Torch device for execution.
        pad_id: Padding token ID for loss masking.
        epoch: 1-based epoch index for logging.
        mode: Logging label (`train` or `valid`).
        log_every: Batch logging interval.

    Returns:
        Dict with aggregate `loss`, `perplexity`, and `tokens`.
    """
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    total_steps = len(loader)

    for step_idx, (mols, decoder_in, targets) in enumerate(loader, start=1):
        mols = mols.to(device)
        decoder_in = decoder_in.to(device)
        targets = targets.to(device)

        # Predict logits over vocab for every sequence position.
        logits = model(mols, decoder_in)
        # Ignore pad positions when computing token-level cross-entropy.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
        )

        valid_tokens = int((targets != pad_id).sum().item())
        total_loss += float(loss.item()) * valid_tokens
        total_tokens += valid_tokens

        if log_every > 0 and (step_idx % log_every == 0 or step_idx == total_steps):
            running_loss = total_loss / max(total_tokens, 1)
            elapsed = time.time() - start_time
            print(
                f"[{mode}] epoch={epoch:03d} step={step_idx}/{total_steps} "
                f"batch_loss={loss.item():.4f} running_loss={running_loss:.4f} "
                f"elapsed={elapsed:.1f}s"
            )

        if train_mode:
            # Standard optimizer step for train mode.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = float(np.exp(min(mean_loss, 20.0)))
    return {"loss": mean_loss, "perplexity": ppl, "tokens": float(total_tokens)}


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments for training.

    Returns:
        Parsed argparse namespace containing training configuration.
    """
    parser = argparse.ArgumentParser(
        description="Train a transformer decoder from molecular embeddings to SELFIES token IDs."
    )
    parser.add_argument(
        "--embedding-h5",
        required=True,
        help="Path to H5 file containing input molecule embeddings (e.g., mol_reps).",
    )
    parser.add_argument(
        "--token-h5",
        required=True,
        help="Path to H5 file containing token_ids and vocab for SELFIES targets.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for checkpoints and training artifacts (history/loss files/plot).",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of full training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training and validation batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay coefficient.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Fraction of rows reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split and training reproducibility.",
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Transformer hidden size used in decoder layers.",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of multi-head attention heads.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer decoder layers.",
    )
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=2048,
        help="Feed-forward sublayer dimension inside each decoder block.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability used in transformer layers.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count (0 is often most stable on macOS/MPS).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print training/validation progress every N batches (default: 100).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on number of rows used for training/validation (0 means all).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. 'auto' selects CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help="Optional path to a training checkpoint (decoder_last.pt) to resume from.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """Resolve target compute device from arg and availability.

    Args:
        device_arg: Requested device string (`auto`, `cpu`, `cuda`, `mps`).

    Returns:
        Torch device selected for training.
    """
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but CUDA is not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS, but MPS is not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    """Main training routine: data setup, loop, checkpointing, artifacts."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset and tokenizer from H5 sources.
    base_dataset = MolToSelfiesDataset(Path(args.embedding_h5), Path(args.token_h5))
    tokenizer = SelfiesTokenizer(base_dataset.base_vocab)
    dataset = base_dataset

    # Optional fast-iteration mode for debugging/trial runs.
    if args.max_samples > 0 and args.max_samples < len(dataset):
        subset_indices = list(range(args.max_samples))
        dataset = torch.utils.data.Subset(dataset, subset_indices)

    if args.log_every < 1:
        raise ValueError("--log-every must be >= 1")

    # Split once with fixed seed for deterministic train/val partitioning.
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Build dataloaders with custom collate for decoder input/target shifting.
    collate = collate_fn(tokenizer)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Initialize model and optimizer.
    model = EmbeddingToSelfiesDecoder(
        embedding_dim=base_dataset.embedding_dim,
        vocab_size=tokenizer.size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=base_dataset.seq_len + 2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    history = []
    train_losses = []
    val_losses = []
    start_epoch = 1

    train_losses_path = output_dir / "train_losses.npy"
    val_losses_path = output_dir / "val_losses.npy"
    loss_plot_path = output_dir / "loss_curve.png"
    plot_saved = False

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume-from path does not exist: {resume_path}")

        ckpt = torch.load(resume_path, map_location=device)

        # Ensure resumed checkpoint matches the current model definition.
        ckpt_config = ckpt.get("config", {})
        expected_config = {
            "embedding_dim": base_dataset.embedding_dim,
            "vocab_size": tokenizer.size,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "max_seq_len": base_dataset.seq_len + 2,
        }
        for key, expected_value in expected_config.items():
            loaded_value = ckpt_config.get(key)
            if loaded_value != expected_value:
                raise ValueError(
                    "Checkpoint/model config mismatch for "
                    f"'{key}': checkpoint={loaded_value}, current={expected_value}."
                )

        model.load_state_dict(ckpt["model_state_dict"])

        optimizer_state = ckpt.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        else:
            print(
                "Warning: optimizer_state_dict missing in checkpoint; "
                "resuming with a fresh optimizer."
            )

        history = list(ckpt.get("history", []))
        if history:
            train_losses = [float(row["train_loss"]) for row in history]
            val_losses = [float(row["val_loss"]) for row in history]
            best_val = min(val_losses)

        last_completed_epoch = int(ckpt.get("epoch", len(history)))
        start_epoch = last_completed_epoch + 1
        print(
            f"Resumed from {resume_path} (completed_epoch={last_completed_epoch}, "
            f"next_epoch={start_epoch})."
        )

    if start_epoch > args.epochs:
        print(
            f"No training needed: checkpoint already at epoch {start_epoch - 1} "
            f"and --epochs={args.epochs}."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        # Run one full training epoch then one validation epoch.
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            tokenizer.pad_id,
            epoch=epoch,
            mode="train",
            log_every=args.log_every,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            None,
            device,
            tokenizer.pad_id,
            epoch=epoch,
            mode="valid",
            log_every=args.log_every,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_ppl": train_stats["perplexity"],
            "val_loss": val_stats["loss"],
            "val_ppl": val_stats["perplexity"],
        }
        history.append(row)
        train_losses.append(row["train_loss"])
        val_losses.append(row["val_loss"])

        # Persist scalar artifacts after every epoch for monitoring/restart safety.
        np.save(train_losses_path, np.array(train_losses, dtype=np.float32))
        np.save(val_losses_path, np.array(val_losses, dtype=np.float32))
        plot_saved = save_loss_plot(
            loss_plot_path,
            train_losses=np.array(train_losses, dtype=np.float32),
            val_losses=np.array(val_losses, dtype=np.float32),
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.4f} train_ppl={row['train_ppl']:.2f} | "
            f"val_loss={row['val_loss']:.4f} val_ppl={row['val_ppl']:.2f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "embedding_dim": base_dataset.embedding_dim,
                "vocab_size": tokenizer.size,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "max_seq_len": base_dataset.seq_len + 2,
            },
            "tokenizer_tokens": tokenizer.info.tokens,
            "pad_id": tokenizer.pad_id,
            "bos_id": tokenizer.bos_id,
            "eos_id": tokenizer.eos_id,
            "seq_len": base_dataset.seq_len,
            "embedding_h5": args.embedding_h5,
            "token_h5": args.token_h5,
            "epoch": epoch,
            "history": history,
        }

        # Always update "last" checkpoint.
        last_path = output_dir / "decoder_last.pt"
        torch.save(ckpt, last_path)

        # Update "best" checkpoint only when validation improves.
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_path = output_dir / "decoder_best.pt"
            torch.save(ckpt, best_path)

    # Save compact JSON history for downstream analysis.
    with (output_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved train losses: {train_losses_path}")
    print(f"Saved val losses: {val_losses_path}")
    if plot_saved:
        print(f"Saved loss plot: {loss_plot_path}")
    else:
        print("Loss plot not saved (matplotlib unavailable).")
    print(f"Training complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
