#!/usr/bin/env python3
"""Convert SMILES files to SELFIES.

This script supports:
- one-to-one conversion (one SELFIES per input SMILES)
- optional one-to-many mapping by generating randomized SMILES traversals
  and encoding each to SELFIES (requires RDKit)

Input format:
- One record per line
- First token is treated as SMILES
- Optional second token is treated as an ID/name

Examples:
    python smi_2_selfie.py --input molecules.smi --output-selfies molecules.selfies

    python smi_2_selfie.py --input molecules.smi --output-selfies molecules.selfies \
        --variants 8 --mapping-jsonl molecules.selfies_map.jsonl

    In --variants > 1 mode, each output line contains all generated SELFIES variants
    for that molecule, separated by spaces.

    After conversion, the script derives an alphabet from the SELFIES output,
    adds [nop], and writes a padded SELFIES file where every SELFIES string is
    padded to the longest SELFIES length.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import selfies as sf


def parse_line(line: str) -> Optional[Tuple[str, Optional[str]]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    parts = stripped.split()
    smiles = parts[0]
    mol_id = parts[1] if len(parts) > 1 else None
    return smiles, mol_id


def encode_smiles(smiles: str) -> Optional[str]:
    try:
        return sf.encoder(smiles)
    except sf.EncoderError:
        return None


def randomized_smiles_variants(smiles: str, variants: int, seed: int) -> List[str]:
    # Lazy import so one-to-one conversion works without RDKit.
    try:
        from rdkit import Chem
        from rdkit import rdBase
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for --variants > 1. Install rdkit and try again."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    results = {Chem.MolToSmiles(mol, canonical=True)}
    # Oversample attempts to improve chance of collecting unique traversals.
    max_attempts = max(variants * 10, 20)

    for i in range(max_attempts):
        # Newer RDKit builds support randomSeed in MolToSmiles; older builds do not.
        try:
            random_smiles = Chem.MolToSmiles(
                mol,
                canonical=False,
                doRandom=True,
                randomSeed=seed + i,
            )
        except Exception:
            rdBase.SeedRandomNumberGenerator(seed + i)
            random_smiles = Chem.MolToSmiles(
                mol,
                canonical=False,
                doRandom=True,
            )
        results.add(random_smiles)
        if len(results) >= variants:
            break

    return list(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SMILES file to SELFIES file")
    parser.add_argument("--input", required=True, help="Input .smi file")
    parser.add_argument(
        "--output-selfies",
        required=True,
        help=(
            "Output file with one line per molecule. "
            "If --variants > 1, variants are space-separated on that line."
        ),
    )
    parser.add_argument(
        "--mapping-jsonl",
        default=None,
        help=(
            "Optional JSONL output describing SMILES->SELFIES mapping. "
            "Useful when --variants > 1."
        ),
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Number of SELFIES variants per input molecule (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for randomized SMILES generation",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Keep failed conversions as empty lines in output-selfies",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N processed molecules (default: 1000)",
    )
    parser.add_argument(
        "--output-alphabet",
        default=None,
        help=(
            "Output file for alphabet tokens (one token per line). "
            "Default: <output-selfies>.alphabet.txt"
        ),
    )
    parser.add_argument(
        "--output-padded-selfies",
        default=None,
        help=(
            "Output file for padded SELFIES strings. "
            "Default: <output-selfies>.padded"
        ),
    )
    return parser


def pad_selfies_with_nop(selfies_str: str, pad_to_len: int) -> str:
    pad_count = pad_to_len - sf.len_selfies(selfies_str)
    if pad_count <= 0:
        return selfies_str
    return selfies_str + ("[nop]" * pad_count)


def build_alphabet_and_padded_selfies(
    selfies_path: Path,
    alphabet_path: Path,
    padded_path: Path,
) -> Tuple[int, int, int]:
    groups: List[List[str]] = []
    all_selfies: List[str] = []

    with selfies_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            group = stripped.split()
            groups.append(group)
            all_selfies.extend(group)

    if not all_selfies:
        raise RuntimeError(
            "No SELFIES were found in output file; cannot build alphabet/padded file."
        )

    alphabet = sf.get_alphabet_from_selfies(all_selfies)
    alphabet.add("[nop]")
    alphabet_sorted = sorted(alphabet)

    max_len = max(sf.len_selfies(s) for s in all_selfies)

    with alphabet_path.open("w", encoding="utf-8") as fout:
        for token in alphabet_sorted:
            fout.write(token + "\n")

    with padded_path.open("w", encoding="utf-8") as fout:
        for group in groups:
            padded_group = [pad_selfies_with_nop(s, max_len) for s in group]
            fout.write(" ".join(padded_group) + "\n")

    return len(alphabet_sorted), max_len, len(all_selfies)


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output_selfies)
    alphabet_path = (
        Path(args.output_alphabet)
        if args.output_alphabet
        else Path(str(output_path) + ".alphabet.txt")
    )
    padded_path = (
        Path(args.output_padded_selfies)
        if args.output_padded_selfies
        else Path(str(output_path) + ".padded")
    )
    mapping_path = Path(args.mapping_jsonl) if args.mapping_jsonl else None

    if args.variants < 1:
        raise ValueError("--variants must be >= 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")

    total = 0
    converted = 0
    failed = 0
    start_time = time.time()

    print(
        f"Starting conversion: input={input_path}, variants={args.variants}, "
        f"progress_every={args.progress_every}",
        flush=True,
    )

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        fmap = mapping_path.open("w", encoding="utf-8") if mapping_path else None
        try:
            for line_no, line in enumerate(fin, start=1):
                parsed = parse_line(line)
                if parsed is None:
                    continue

                total += 1
                smiles, mol_id = parsed

                if args.variants == 1:
                    variants_smiles = [smiles]
                else:
                    variants_smiles = randomized_smiles_variants(
                        smiles=smiles,
                        variants=args.variants,
                        seed=args.seed + line_no * 1000,
                    )

                selfies_variants = []
                for smi_variant in variants_smiles:
                    encoded = encode_smiles(smi_variant)
                    if encoded is not None:
                        selfies_variants.append(encoded)

                # Deduplicate while preserving order.
                selfies_variants = list(dict.fromkeys(selfies_variants))

                if not selfies_variants:
                    failed += 1
                    if args.keep_failed:
                        fout.write("\n")
                    if fmap is not None:
                        record = {
                            "line": line_no,
                            "id": mol_id,
                            "input_smiles": smiles,
                            "selfies": [],
                            "status": "failed",
                        }
                        fmap.write(json.dumps(record) + "\n")
                    continue

                converted += 1
                if args.variants > 1:
                    fout.write(" ".join(selfies_variants) + "\n")
                else:
                    fout.write(selfies_variants[0] + "\n")

                if fmap is not None:
                    record = {
                        "line": line_no,
                        "id": mol_id,
                        "input_smiles": smiles,
                        "selfies": selfies_variants,
                        "status": "ok",
                    }
                    fmap.write(json.dumps(record) + "\n")

                if total % args.progress_every == 0:
                    elapsed = time.time() - start_time
                    rate = total / elapsed if elapsed > 0 else 0.0
                    print(
                        "Progress: "
                        f"processed={total}, converted={converted}, failed={failed}, "
                        f"rate={rate:.1f} mol/s",
                        flush=True,
                    )
        finally:
            if fmap is not None:
                fmap.close()

    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0.0
    print(f"Processed: {total}")
    print(f"Converted: {converted}")
    print(f"Failed: {failed}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Rate: {rate:.1f} mol/s")
    print(f"SELFIES written to: {output_path}")
    if mapping_path:
        print(f"Mapping JSONL written to: {mapping_path}")

    print("Building alphabet and padded SELFIES...", flush=True)
    alpha_size, max_selfies_len, total_selfies_strings = build_alphabet_and_padded_selfies(
        selfies_path=output_path,
        alphabet_path=alphabet_path,
        padded_path=padded_path,
    )
    print(f"Alphabet size (with [nop]): {alpha_size}")
    print(f"Longest SELFIES length: {max_selfies_len}")
    print(f"Total SELFIES strings used for alphabet: {total_selfies_strings}")
    print(f"Alphabet written to: {alphabet_path}")
    print(f"Padded SELFIES written to: {padded_path}")


if __name__ == "__main__":
    main()