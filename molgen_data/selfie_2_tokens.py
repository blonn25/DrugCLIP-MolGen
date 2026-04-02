#!/usr/bin/env python3
"""Convert padded SELFIES strings into ML-ready numeric tensors.

This script is designed for DrugCLIP preprocessing workflows and writes outputs
to H5 (matching existing project conventions).

It always writes:
- token_ids: int matrix of shape (N, L)
- attention_mask: uint8 matrix of shape (N, L), 1 for non-[nop], else 0
- vocab: token list used for integer mapping

It can optionally write:
- one_hot: uint8 tensor of shape (N, L, V)

Why token_ids are the default training target:
- For transformer decoders, token IDs + embedding layer + cross-entropy are
  usually more memory-efficient and standard than dense one-hot targets.

Example usages:
python selfie_2_tokens.py \
  --input-padded-selfies /Users/beaulonnquist/projects/shoichet_lab/drugclip_mol_gen/DrugCLIP/molgen_data/mol_embs/first_100k.selfies.padded \
  --output-h5 /Users/beaulonnquist/projects/shoichet_lab/drugclip_mol_gen/DrugCLIP/molgen_data/mol_embs/first_100k.selfies.tokens.h5 \

python selfie_2_tokens.py \
  --input-padded-selfies mol_embs/1M_chemstep_seedset/seed_smis_1M_canonical_valid.selfies.padded \
  --output-h5 mol_embs/1M_chemstep_seedset/seed_smis_1M_canonical_valid.tokens.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import selfies as sf


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Convert padded SELFIES file into token IDs and optional one-hot H5 datasets."
	)
	parser.add_argument(
		"--input-padded-selfies",
		required=True,
		help="Path to padded SELFIES file (one SELFIES per line).",
	)
	parser.add_argument(
		"--output-h5",
		default=None,
		help="Output H5 path. Default: <input>.onehot.h5",
	)
	parser.add_argument(
		"--alphabet-file",
		default=None,
		help="Optional alphabet file (one token per line). If omitted, infer from input.",
	)
	parser.add_argument(
		"--include-onehot",
		action="store_true",
		help="Also write dense one_hot dataset with shape (N, L, V).",
	)
	parser.add_argument(
		"--compression",
		default="gzip",
		choices=["gzip", "lzf", "none"],
		help="H5 compression for large arrays (default: gzip).",
	)
	parser.add_argument(
		"--compression-level",
		type=int,
		default=4,
		help="Gzip compression level (0-9). Ignored for non-gzip compression.",
	)
	parser.add_argument(
		"--verify-rows",
		type=int,
		default=0,
		help="Optional: ensure number of SELFIES rows matches this value.",
	)
	return parser


def parse_selfies_line(line: str) -> str:
	"""Return the SELFIES string from one input line.

	If multiple space-separated entries exist, the first token is used.
	"""
	stripped = line.strip()
	if not stripped:
		raise ValueError("Encountered empty line in input.")
	return stripped.split()[0]


def split_tokens(selfies_str: str) -> List[str]:
	tokens = list(sf.split_selfies(selfies_str))
	if not tokens:
		raise ValueError(f"Failed to parse SELFIES: {selfies_str[:80]}")
	return tokens


def infer_vocab_and_shape(input_path: Path) -> Tuple[List[str], int, int]:
	token_set = set()
	expected_len = None
	n_rows = 0

	with input_path.open("r", encoding="utf-8") as fin:
		for line_no, line in enumerate(fin, start=1):
			selfies_str = parse_selfies_line(line)
			tokens = split_tokens(selfies_str)

			if expected_len is None:
				expected_len = len(tokens)
			elif len(tokens) != expected_len:
				raise ValueError(
					f"Inconsistent token length at line {line_no}: "
					f"got {len(tokens)}, expected {expected_len}. "
					"Input should be padded to a fixed length."
				)

			token_set.update(tokens)
			n_rows += 1

	if n_rows == 0 or expected_len is None:
		raise ValueError("Input file has no valid SELFIES lines.")

	if "[nop]" not in token_set:
		raise ValueError("[nop] token not found. Input should contain padded SELFIES.")

	vocab = sorted(token_set)
	return vocab, n_rows, expected_len


def load_vocab_from_file(alphabet_path: Path) -> List[str]:
	vocab = []
	with alphabet_path.open("r", encoding="utf-8") as fin:
		for line in fin:
			tok = line.strip()
			if tok:
				vocab.append(tok)
	if not vocab:
		raise ValueError(f"Alphabet file is empty: {alphabet_path}")
	if "[nop]" not in vocab:
		raise ValueError("Alphabet file must include [nop].")
	# Preserve file order while dropping duplicates.
	seen = set()
	deduped = []
	for tok in vocab:
		if tok not in seen:
			seen.add(tok)
			deduped.append(tok)
	return deduped


def get_h5_compression(args: argparse.Namespace):
	if args.compression == "none":
		return None, None
	if args.compression == "gzip":
		return "gzip", args.compression_level
	return args.compression, None


def main() -> None:
	args = build_parser().parse_args()

	input_path = Path(args.input_padded_selfies)
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	output_h5 = Path(args.output_h5) if args.output_h5 else Path(str(input_path) + ".onehot.h5")
	output_h5.parent.mkdir(parents=True, exist_ok=True)

	print(f"Scanning input: {input_path}", flush=True)
	inferred_vocab, n_rows, seq_len = infer_vocab_and_shape(input_path)

	if args.alphabet_file:
		vocab = load_vocab_from_file(Path(args.alphabet_file))
		missing = sorted(set(inferred_vocab) - set(vocab))
		if missing:
			raise ValueError(
				"Alphabet file does not cover all tokens in input. "
				f"Missing tokens: {missing[:10]}"
			)
	else:
		vocab = inferred_vocab

	if args.verify_rows and n_rows != args.verify_rows:
		raise ValueError(
			f"Row count mismatch: input has {n_rows}, expected {args.verify_rows}."
		)

	vocab_size = len(vocab)
	token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(vocab)}
	nop_id = token_to_id["[nop]"]

	if vocab_size <= np.iinfo(np.uint8).max:
		token_dtype = np.uint8
	elif vocab_size <= np.iinfo(np.uint16).max:
		token_dtype = np.uint16
	else:
		token_dtype = np.uint32

	compression, compression_opts = get_h5_compression(args)

	print(
		f"Writing H5: rows={n_rows}, seq_len={seq_len}, vocab={vocab_size}, "
		f"include_onehot={args.include_onehot}",
		flush=True,
	)

	chunk_rows = min(1024, n_rows)
	chunk_tokens = min(seq_len, 128)
	chunk_oh_vocab = min(vocab_size, 64)

	with h5py.File(output_h5, "w") as h5f:
		token_ds = h5f.create_dataset(
			"token_ids",
			shape=(n_rows, seq_len),
			dtype=token_dtype,
			chunks=(chunk_rows, chunk_tokens),
			compression=compression,
			compression_opts=compression_opts,
		)
		mask_ds = h5f.create_dataset(
			"attention_mask",
			shape=(n_rows, seq_len),
			dtype=np.uint8,
			chunks=(chunk_rows, chunk_tokens),
			compression=compression,
			compression_opts=compression_opts,
		)

		onehot_ds = None
		if args.include_onehot:
			onehot_ds = h5f.create_dataset(
				"one_hot",
				shape=(n_rows, seq_len, vocab_size),
				dtype=np.uint8,
				chunks=(max(1, min(64, n_rows)), chunk_tokens, chunk_oh_vocab),
				compression=compression,
				compression_opts=compression_opts,
			)

		str_dtype = h5py.string_dtype(encoding="utf-8")
		h5f.create_dataset("vocab", data=np.array(vocab, dtype=object), dtype=str_dtype)

		# Store mapping as JSON metadata for convenient downstream loading.
		h5f.attrs["token_to_id_json"] = json.dumps(token_to_id)
		h5f.attrs["pad_token"] = "[nop]"
		h5f.attrs["pad_token_id"] = int(nop_id)
		h5f.attrs["num_rows"] = int(n_rows)
		h5f.attrs["seq_len"] = int(seq_len)
		h5f.attrs["vocab_size"] = int(vocab_size)

		with input_path.open("r", encoding="utf-8") as fin:
			for row_idx, line in enumerate(fin):
				selfies_str = parse_selfies_line(line)
				tokens = split_tokens(selfies_str)
				if len(tokens) != seq_len:
					raise ValueError(
						f"Length changed at row {row_idx}: got {len(tokens)}, expected {seq_len}."
					)

				ids = np.fromiter((token_to_id[tok] for tok in tokens), dtype=token_dtype)
				token_ds[row_idx, :] = ids
				mask_ds[row_idx, :] = (ids != nop_id).astype(np.uint8)

				if onehot_ds is not None:
					oh = np.zeros((seq_len, vocab_size), dtype=np.uint8)
					oh[np.arange(seq_len), ids.astype(np.int64)] = 1
					onehot_ds[row_idx, :, :] = oh

				if (row_idx + 1) % 5000 == 0 or (row_idx + 1) == n_rows:
					print(f"Processed {row_idx + 1}/{n_rows}", flush=True)

	print(f"Saved: {output_h5}", flush=True)


if __name__ == "__main__":
	main()
