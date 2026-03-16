#!/usr/bin/env python3 -u
"""Encode pocket LMDB into a multi-fold DrugCLIP embedding H5.

This script encodes pocket records using multiple fold checkpoints (default: 6),
collects one 128-d embedding per fold, and concatenates them to produce
N x (num_folds * 128) embeddings.

For a single-pocket LMDB and 6 folds, output is 1 x 768.

Example usage:
python encode_pocket_multifold.py ./data \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --max-pocket-atoms 256 \
  --path /Users/beaulonnquist/Library/CloudStorage/Box-Box/shoichetlab/drugclip_model_weights/6_folds/fold_0.pt \
  --pocket-path /Users/beaulonnquist/projects/shoichet_lab/drugclip_mol_gen/DrugCLIP/molgen_data/pocket_embs/mu_prot.lmdb \
  --fold-dir /Users/beaulonnquist/Library/CloudStorage/Box-Box/shoichetlab/drugclip_model_weights/6_folds \
  --num-folds 6 \
  --cpu
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch

import unicore
from unicore import checkpoint_utils, distributed_utils, options, tasks

# Ensure user task/model registrations are imported.
import unimol  # noqa: F401


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("drugclip.encode_pocket_multifold")


def encode_pockets_for_current_model(
    task,
    model,
    pocket_path: str,
    use_cuda: bool,
    batch_size: int,
    num_workers: int,
    progress_every: int,
) -> Tuple[np.ndarray, List[str]]:
    """Encode all pockets from LMDB using the currently loaded model weights."""
    pocket_dataset = task.load_pockets_dataset(pocket_path)
    pocket_data = torch.utils.data.DataLoader(
        pocket_dataset,
        batch_size=batch_size,
        collate_fn=pocket_dataset.collater,
        num_workers=num_workers,
    )

    pocket_reps = []
    pocket_names = []
    total_batches = len(pocket_data)

    for batch_idx, sample in enumerate(pocket_data, start=1):
        if use_cuda:
            sample = unicore.utils.move_to_cuda(sample)

        dist = sample["net_input"]["pocket_src_distance"]
        edge_type = sample["net_input"]["pocket_src_edge_type"]
        src_tokens = sample["net_input"]["pocket_src_tokens"]

        pocket_padding_mask = src_tokens.eq(model.pocket_model.padding_idx)
        pocket_x = model.pocket_model.embed_tokens(src_tokens)

        n_node = dist.size(-1)
        gbf_feature = model.pocket_model.gbf(dist, edge_type)
        gbf_result = model.pocket_model.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous().view(
            -1, n_node, n_node
        )

        pocket_outputs = model.pocket_model.encoder(
            pocket_x,
            padding_mask=pocket_padding_mask,
            attn_mask=graph_attn_bias,
        )
        pocket_encoder_rep = pocket_outputs[0][:, 0, :]
        pocket_emb = model.pocket_project(pocket_encoder_rep)
        pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
        pocket_emb = pocket_emb.detach().cpu().numpy().astype(np.float32)

        pocket_reps.append(pocket_emb)
        pocket_names.extend(sample["pocket_name"])

        if batch_idx % progress_every == 0 or batch_idx == total_batches:
            encoded_so_far = sum(x.shape[0] for x in pocket_reps)
            logger.info(
                "Pocket encoding progress: batch %d/%d, encoded=%d",
                batch_idx,
                total_batches,
                encoded_so_far,
            )

    pocket_reps = np.concatenate(pocket_reps, axis=0)
    return pocket_reps, pocket_names


def main(args) -> None:
    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    task = tasks.setup_task(args)
    model = task.build_model(args)

    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    fold_embs = []
    ref_names: List[str] | None = None

    for fold_idx in range(args.num_folds):
        ckpt_path = os.path.join(args.fold_dir, f"fold_{fold_idx}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        logger.info("Loading fold checkpoint: %s", ckpt_path)
        state = checkpoint_utils.load_checkpoint_to_cpu(ckpt_path)
        model.load_state_dict(state["model"], strict=False)
        model.eval()

        reps, names = encode_pockets_for_current_model(
            task=task,
            model=model,
            pocket_path=args.pocket_path,
            use_cuda=use_cuda,
            batch_size=args.pocket_batch_size,
            num_workers=args.num_workers,
            progress_every=args.progress_every,
        )

        if ref_names is None:
            ref_names = names
        elif names != ref_names:
            raise RuntimeError(
                "Pocket ordering mismatch across folds; cannot safely concatenate."
            )

        fold_embs.append(reps)
        logger.info("Fold %d embedding shape: %s", fold_idx, reps.shape)

    # (F, N, 128) -> (N, F, 128) -> (N, F*128)
    stacked = np.stack(fold_embs, axis=0)
    pocket_reps = stacked.transpose(1, 0, 2).reshape(stacked.shape[1], -1)

    expected_dim = args.num_folds * 128
    if pocket_reps.shape[1] != expected_dim:
        raise RuntimeError(
            f"Unexpected embedding dim {pocket_reps.shape[1]} (expected {expected_dim})."
        )

    output_h5 = args.output_h5
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    with h5py.File(output_h5, "w") as h5f:
        h5f.create_dataset("pocket_reps", data=pocket_reps.astype(np.float32))
        if ref_names is not None:
            str_dtype = h5py.string_dtype(encoding="utf-8")
            h5f.create_dataset("pocket_names", data=np.array(ref_names, dtype=object), dtype=str_dtype)

    logger.info("Saved pocket embeddings to: %s", output_h5)
    logger.info("Final embedding shape: %s", pocket_reps.shape)


def cli_main() -> None:
    # Local script already imports unimol; avoid parser-time --user-dir collisions.
    cleaned_argv = [sys.argv[0]]
    skip_next = False
    for idx, token in enumerate(sys.argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue

        if token == "--user-dir":
            user_dir_val = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
            if os.path.basename(os.path.normpath(user_dir_val)) == "unimol":
                logger.warning(
                    "Ignoring --user-dir %s to avoid non-unique unimol module import.",
                    user_dir_val,
                )
                skip_next = True
                continue

        if token.startswith("--user-dir="):
            user_dir_val = token.split("=", 1)[1]
            if os.path.basename(os.path.normpath(user_dir_val)) == "unimol":
                logger.warning(
                    "Ignoring --user-dir %s to avoid non-unique unimol module import.",
                    user_dir_val,
                )
                continue

        cleaned_argv.append(token)

    sys.argv = cleaned_argv

    parser = options.get_validation_parser()
    parser.add_argument(
        "--pocket-path",
        type=str,
        required=True,
        help="Path to pocket LMDB file",
    )
    parser.add_argument(
        "--output-h5",
        type=str,
        default=None,
        help="Output H5 file path. Default: same prefix as --pocket-path with .h5",
    )
    parser.add_argument(
        "--fold-dir",
        type=str,
        default="./data/model_weights/6_folds",
        help="Directory containing fold_i.pt checkpoints",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=6,
        help="Number of folds/checkpoints to use",
    )
    parser.add_argument(
        "--pocket-batch-size",
        type=int,
        default=16,
        help="Pocket dataloader batch size",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Log progress every N batches while encoding pockets",
    )

    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")

    if args.output_h5 is None:
        pocket_path = Path(args.pocket_path)
        args.output_h5 = str(pocket_path.with_suffix(".h5"))

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
