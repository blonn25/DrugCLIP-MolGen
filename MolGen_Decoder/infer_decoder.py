#!/usr/bin/env python3
"""Inference entrypoint for embedding-to-SELFIES decoder.

Loads a trained checkpoint, runs autoregressive decoding for selected rows of
embedding H5 data, and writes JSONL records containing token IDs, SELFIES,
and decoded SMILES.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import h5py
import numpy as np
import selfies as sf
import torch
from torch.utils.data import DataLoader, TensorDataset

from selfies_decoder import EmbeddingToSelfiesDecoder, SelfiesTokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for decoder inference.

    Returns:
        Parsed argparse namespace containing inference configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with a trained embedding->SELFIES decoder and optionally "
            "convert generated token IDs to SELFIES and SMILES."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained decoder checkpoint (.pt), typically decoder_best.pt.",
    )
    parser.add_argument(
        "--embedding-h5",
        required=True,
        help="Path to H5 file containing input embeddings for decoding.",
    )
    parser.add_argument(
        "--embedding-key",
        default="mol_reps",
        help=(
            "Preferred dataset key in embedding H5. If missing, loader will "
            "fallback to 'mol_reps' then 'pocket_reps'."
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output JSONL path for decoded records (one JSON object per line).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help=(
            "Target decode batch size. DataLoader batch size is adjusted as "
            "max(1, batch_size // embedding_repeat) before repeat expansion."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum number of generated tokens per sample (excluding BOS).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help=(
            "Number of decoding samples to generate per input embedding. "
            "All samples are output in the same JSON object for each input row."
        ),
    )
    parser.add_argument(
        "--embedding-repeat",
        type=int,
        default=1,
        help=(
            "Number of repeated copies to create per input embedding before decoding. "
            "For N input rows, effective decode rows become N * embedding_repeat."
        ),
    )
    parser.add_argument(
        "--embedding-noise-strength",
        type=float,
        default=0.0,
        help=(
            "Gaussian noise strength in [0, 1] applied independently to every effective "
            "decode embedding (including repeats). 0 means no noise."
        ),
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=-1,
        help=(
            "Optional RNG seed for embedding noise. Use >=0 for reproducible noise; "
            "negative values use non-deterministic noise."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature; values >1 increase diversity, <1 make outputs sharper.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="If >0 with --stochastic-sampling, sample only from top-k logits at each step.",
    )
    parser.add_argument(
        "--stochastic-sampling",
        "--do-sample",  # old alias, renamed for clarity
        dest="stochastic_sampling",
        action="store_true",
        help=(
            "Enable stochastic token sampling; otherwise uses greedy argmax decoding. "
            "`--do-sample` is retained as a backward-compatible alias."
        ),
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Inclusive start row index in embedding dataset.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=0,
        help="Exclusive end row index; 0 means decode through end of dataset.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. 'auto' selects CUDA, then MPS, then CPU.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """Resolve runtime device from explicit setting or availability.

    Args:
        device_arg: Requested device string (`auto`, `cpu`, `cuda`, `mps`).

    Returns:
        Torch device selected for inference.
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


def load_embeddings(path: Path, key: str, start: int, end: int) -> np.ndarray:
    """Load a contiguous row slice from embedding H5 dataset.

    Args:
        path: Path to H5 file.
        key: Dataset key in H5 (default: `mol_reps`).
        start: Inclusive start row index.
        end: Exclusive end row index. `0` means use full length.

    Returns:
        Float32 embedding matrix slice with shape (N, D).
    """
    fallback_keys = []
    if key != "mol_reps":
        fallback_keys.append("mol_reps")
    if key != "pocket_reps":
        fallback_keys.append("pocket_reps")
    candidate_keys = [key] + fallback_keys

    with h5py.File(path, "r") as f:
        arr = None
        used_key = None

        for i, candidate in enumerate(candidate_keys):
            try:
                arr = f[candidate]
                used_key = candidate
                break
            except KeyError:
                if i == 0 and len(candidate_keys) > 1:
                    print(
                        f"Attempt to load with '{candidate}' failed. "
                        f"Trying with '{candidate_keys[i + 1]}'...",
                        flush=True,
                    )
                elif i < len(candidate_keys) - 1:
                    print(
                        f"Attempt to load with '{candidate}' failed. "
                        f"Trying again with '{candidate_keys[i + 1]}'...",
                        flush=True,
                    )

        if arr is None or used_key is None:
            raise KeyError(
                f"None of the candidate keys were found in {path}: {candidate_keys}"
            )

        print(f"Successfully loaded embeddings using key '{used_key}'.", flush=True)
        n = arr.shape[0]
        s = max(0, start)
        e = n if end <= 0 else min(end, n)
        if s >= e:
            raise ValueError(f"Invalid slice start={s}, end={e}, total={n}")
        return arr[s:e].astype(np.float32)


def ids_to_selfies_and_smiles(tokenizer: SelfiesTokenizer, ids: List[int]) -> tuple[str, str | None]:
    """Convert generated token IDs into SELFIES and best-effort SMILES.

    Args:
        tokenizer: Tokenizer with ID-to-token mapping.
        ids: Generated token IDs for one decoded sample.

    Returns:
        Tuple of reconstructed SELFIES and decoded SMILES (or None on failure).
    """
    selfies_str = tokenizer.ids_to_selfies(ids)
    try:
        smiles = sf.decoder(selfies_str)
    except Exception:
        smiles = None
    return selfies_str, smiles


def add_gaussian_noise_unit_sphere(
    embeddings: torch.Tensor,
    strength: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add isotropic Gaussian noise calibrated for unit-norm embeddings.

    Embeddings in this project are L2-normalized to magnitude 1. We scale noise so
    expected perturbation norm is approximately `strength`, then renormalize each
    vector back to unit norm to stay on the same manifold.

    Args:
        embeddings: Input tensor of shape (N, D).
        strength: Noise strength in [0, 1].
        generator: Optional torch RNG for reproducible draws.

    Returns:
        Noised and L2-normalized embeddings of shape (N, D).
    """
    if strength <= 0:
        return embeddings

    dim = int(embeddings.shape[1])
    sigma = strength / math.sqrt(max(dim, 1))
    noise = torch.randn(
        embeddings.shape,
        device=embeddings.device,
        dtype=embeddings.dtype,
        generator=generator,
    )
    perturbed = embeddings + sigma * noise
    return torch.nn.functional.normalize(perturbed, p=2, dim=1)


def main() -> None:
    """Run decoding loop and write predictions as JSONL records.

    Returns:
        None. Results are written to disk.
    """
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.embedding_repeat < 1:
        raise ValueError("--embedding-repeat must be >= 1")
    if not (0.0 <= args.embedding_noise_strength <= 1.0):
        raise ValueError("--embedding-noise-strength must be in [0, 1]")

    if args.num_samples > 1 and not args.stochastic_sampling:
        print(
            "Warning: --num-samples > 1 with greedy decoding (stochastic sampling disabled). "
            "Outputs are likely identical.",
            flush=True,
        )

    device = resolve_device(args.device)
    noise_generator: torch.Generator | None = None
    if args.noise_seed >= 0:
        noise_generator = torch.Generator(device=device.type)
        noise_generator.manual_seed(args.noise_seed)

    # Restore model and tokenizer state from training checkpoint.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    tokenizer = SelfiesTokenizer(ckpt["tokenizer_tokens"])

    config = ckpt["config"]
    model = EmbeddingToSelfiesDecoder(**config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    # Select the embedding rows requested by --start/--end.
    embs = load_embeddings(Path(args.embedding_h5), args.embedding_key, args.start, args.end)
    dataset = TensorDataset(torch.from_numpy(embs))
    base_batch_size = max(1, args.batch_size // args.embedding_repeat)
    loader = DataLoader(dataset, batch_size=base_batch_size, shuffle=False)

    total_original = int(embs.shape[0])
    total_effective = total_original * args.embedding_repeat
    total_decode_calls = total_effective * args.num_samples

    print(
        "Inference configuration: "
        f"original_rows={total_original}, "
        f"embedding_repeat={args.embedding_repeat}, "
        f"effective_rows={total_effective}, "
        f"num_samples={args.num_samples}, "
        f"total_decode_sequences={total_decode_calls}, "
        f"target_batch_size={args.batch_size}, "
        f"base_batch_size={base_batch_size}, "
        f"noise_strength={args.embedding_noise_strength}, "
        f"noise_seed={args.noise_seed}, "
        f"device={device}"
    )

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    global_idx = args.start
    processed_original = 0
    processed_decode_sequences = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for batch_num, (batch_embs,) in enumerate(loader, start=1):
            batch_embs = batch_embs.to(device)
            batch_size = int(batch_embs.shape[0])
            batch_start_idx = global_idx

            print(
                f"[batch {batch_num:04d}] rows={batch_size}, "
                f"expanded_rows={batch_size * args.embedding_repeat}, "
                f"index_range=[{batch_start_idx}, {batch_start_idx + batch_size})"
            )

            # Expand each row into repeated copies to support embedding-level batching/diversity.
            expanded_embs = batch_embs.repeat_interleave(args.embedding_repeat, dim=0)
            expanded_embs = add_gaussian_noise_unit_sphere(
                expanded_embs,
                strength=args.embedding_noise_strength,
                generator=noise_generator,
            )

            # Collect all sampled outputs for each embedding row in this batch.
            grouped_samples = [[] for _ in range(batch_size)]

            for sample_id in range(args.num_samples):
                # Autoregressively decode token IDs for each expanded embedding.
                generated = model.generate(
                    mol_emb=expanded_embs,
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id,
                    pad_id=tokenizer.pad_id,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    do_sample=args.stochastic_sampling,
                )
                generated = generated.detach().cpu().numpy()
                processed_decode_sequences += int(generated.shape[0])

                if args.num_samples > 1:
                    print(
                        f"  sample {sample_id + 1}/{args.num_samples}: "
                        f"decoded_sequences={int(generated.shape[0])}, "
                        f"cumulative_decoded={processed_decode_sequences}/{total_decode_calls}"
                    )

                for row_idx, row in enumerate(generated):
                    orig_row_idx = row_idx // args.embedding_repeat
                    repeat_id = row_idx % args.embedding_repeat
                    token_ids = [int(x) for x in row.tolist()]
                    selfies_str, smiles = ids_to_selfies_and_smiles(tokenizer, token_ids)
                    grouped_samples[orig_row_idx].append(
                        {
                            "sample_id": sample_id,
                            "repeat_id": repeat_id,
                            "token_ids": token_ids,
                            "selfies": selfies_str,
                            "smiles": smiles,
                        }
                    )

            for row_idx, samples in enumerate(grouped_samples):
                # Emit one JSON object per input embedding row with all samples.
                record = {
                    "index": batch_start_idx + row_idx,
                    "samples": samples,
                }
                fout.write(json.dumps(record) + "\n")

            global_idx += batch_size
            processed_original += batch_size
            print(
                f"[batch {batch_num:04d}] complete: "
                f"processed_original={processed_original}/{total_original}, "
                f"written_lines={processed_original}, "
                f"cumulative_decoded={processed_decode_sequences}/{total_decode_calls}",
                flush=True,
            )

    print(f"Saved inference outputs to: {out_path}")


if __name__ == "__main__":
    main()
