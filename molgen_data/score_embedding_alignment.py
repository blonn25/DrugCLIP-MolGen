#!/usr/bin/env python3
"""Score alignment between pocket and molecule embeddings from H5 files.

This mirrors the retrieval logic in DrugCLIP's retrieval_multi_folds:
1) Load multi-fold molecule/pocket embeddings from H5.
2) Compute pocket-vs-molecule dot products for each fold.
3) Average scores across folds.
4) Apply per-pocket median/MAD adjustment.
5) Reduce to one score per molecule by max over pockets.
6) Save scores (ranked) to a user-provided output file.

Example:
python score_embedding_alignment.py \
    --mol-h5 mol_embs/random_mols/random_mols_embeddings.h5 \
	--pocket-h5 pocket_embs/mu_prot.h5 \
	--output mol_embs/random_mols/rand_scores/alignment_scores.txt \
	--num-folds 6 \
	--no-median-adjustment
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Sequence

import h5py
import numpy as np


def _pick_first_existing_key(h5f: h5py.File, candidates: Sequence[str]) -> str:
	for key in candidates:
		if key in h5f:
			return key
	raise KeyError(f"None of the candidate datasets were found: {candidates}")


def _read_optional_names(h5f: h5py.File, candidates: Sequence[str]) -> list[str] | None:
	for key in candidates:
		if key in h5f:
			raw = h5f[key][:]
			out = []
			for x in raw:
				if isinstance(x, bytes):
					out.append(x.decode("utf-8"))
				else:
					out.append(str(x))
			return out
	return None


def _to_fold_tensor(x: np.ndarray, num_folds: int, name: str) -> np.ndarray:
	"""Convert embeddings to shape (N, F, D)."""
	x = np.asarray(x, dtype=np.float32)

	if x.ndim == 3:
		if x.shape[1] != num_folds:
			raise ValueError(
				f"{name} has shape {x.shape}; expected fold dimension {num_folds} at axis=1"
			)
		return x

	if x.ndim == 2:
		n, dim = x.shape
		if dim % num_folds != 0:
			raise ValueError(
				f"{name} has shape {x.shape}; embedding dim {dim} is not divisible by num_folds={num_folds}"
			)
		per_fold_dim = dim // num_folds
		return x.reshape(n, num_folds, per_fold_dim)

	raise ValueError(f"{name} must be rank-2 or rank-3, got shape {x.shape}")


def score_from_h5_multi_folds(
	mol_h5_path: str,
	pocket_h5_path: str,
	output_path: str,
	num_folds: int = 6,
	mol_dataset_key: str | None = None,
	pocket_dataset_key: str | None = None,
	apply_median_adjustment: bool = True,
) -> None:
	"""Compute per-molecule retrieval scores from H5 embeddings and save to disk."""
	with h5py.File(mol_h5_path, "r") as mol_h5:
		if mol_dataset_key is None:
			mol_dataset_key = _pick_first_existing_key(
				mol_h5, ("mol_reps", "molecule_reps", "mols", "embeddings")
			)
		mol_reps_raw = mol_h5[mol_dataset_key][:]
		mol_names = _read_optional_names(
			mol_h5, ("mol_names", "molecule_names", "smi_name", "smi", "names")
		)

	with h5py.File(pocket_h5_path, "r") as pocket_h5:
		if pocket_dataset_key is None:
			pocket_dataset_key = _pick_first_existing_key(
				pocket_h5, ("pocket_reps", "pockets", "embeddings")
			)
		pocket_reps_raw = pocket_h5[pocket_dataset_key][:]

	mol_reps = _to_fold_tensor(mol_reps_raw, num_folds=num_folds, name="mol_reps")
	pocket_reps = _to_fold_tensor(pocket_reps_raw, num_folds=num_folds, name="pocket_reps")

	if mol_reps.shape[2] != pocket_reps.shape[2]:
		raise ValueError(
			f"Per-fold embedding dims do not match: mol={mol_reps.shape[2]}, pocket={pocket_reps.shape[2]}"
		)

	# Fold-wise scores: for each fold f, compute (P, D) @ (M, D)^T => (P, M)
	fold_scores = []
	for fold in range(num_folds):
		p = pocket_reps[:, fold, :].astype(np.float32, copy=False)
		m = mol_reps[:, fold, :].astype(np.float32, copy=False)
		fold_scores.append(p @ m.T)

	scores = np.mean(np.stack(fold_scores, axis=0), axis=0)

	if apply_median_adjustment:
		medians = np.median(scores, axis=1, keepdims=True)
		mads = np.median(np.abs(scores - medians), axis=1, keepdims=True)
		scores = 0.6745 * (scores - medians) / (mads + 1e-6)

	# Match retrieval_multi_folds behavior: reduce pockets -> one score per molecule.
	mol_scores = np.max(scores, axis=0)

	if mol_names is None:
		num_mols = mol_scores.shape[0]
		width = len(str(max(num_mols - 1, 0)))
		mol_names = [f"mol_{i:0{width}d}" for i in range(num_mols)]
	elif len(mol_names) != mol_scores.shape[0]:
		raise ValueError(
			f"Number of molecule names ({len(mol_names)}) does not match mol scores ({mol_scores.shape[0]})"
		)

	ranked = list(zip(mol_scores.tolist(), mol_names))
	ranked.sort(key=lambda x: x[0], reverse=True)

	out_dir = os.path.dirname(output_path)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)

	with open(output_path, "w", encoding="utf-8") as f:
		for score, name in ranked:
			f.write(f"{name},{score}\n")


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Score mol/pocket alignment from H5 embeddings using multi-fold averaging."
	)
	parser.add_argument("--mol-h5", required=True, help="Path to molecule embedding H5")
	parser.add_argument("--pocket-h5", required=True, help="Path to pocket embedding H5")
	parser.add_argument("--output", required=True, help="Output CSV path: name,score")
	parser.add_argument(
		"--num-folds",
		type=int,
		default=6,
		help="Number of folds in embeddings (default: 6)",
	)
	parser.add_argument(
		"--mol-dataset-key",
		default=None,
		help="Dataset key for molecule embeddings in --mol-h5",
	)
	parser.add_argument(
		"--pocket-dataset-key",
		default=None,
		help="Dataset key for pocket embeddings in --pocket-h5",
	)
	parser.add_argument(
		"--no-median-adjustment",
		action="store_true",
		help="Disable median/MAD adjustment (enabled by default)",
	)
	return parser


def main() -> None:
	parser = _build_parser()
	args = parser.parse_args()

	score_from_h5_multi_folds(
		mol_h5_path=args.mol_h5,
		pocket_h5_path=args.pocket_h5,
		output_path=args.output,
		num_folds=args.num_folds,
		mol_dataset_key=args.mol_dataset_key,
		pocket_dataset_key=args.pocket_dataset_key,
		apply_median_adjustment=not args.no_median_adjustment,
	)


if __name__ == "__main__":
	main()
