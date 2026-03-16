#!/usr/bin/env python3
"""Build per-record .smi files from a decoder JSONL output.

Input JSONL format (one object per line):
- index: integer or string identifier for the record
- samples: list of sample objects containing at least:
  - sample_id
  - smiles (may be null)

For each JSON object, this script writes one .smi file with lines:
    <smiles> mol_<repeat_id>r_<sample_id>s

By default output files are written next to the input JSONL, with names:
  <input_stem>_<index>.smi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_int(value: Any) -> int | None:
    """Return integer value when possible, otherwise None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_mol_label(
    repeat_id: Any,
    sample_id: Any,
    repeat_pad_width: int,
    sample_pad_width: int,
) -> str:
    """Format output label as mol_<repeat_id>r_<sample_id>s with zero padding when numeric."""
    rid_int = _safe_int(repeat_id)
    sid_int = _safe_int(sample_id)

    if rid_int is not None:
        rid_text = f"{rid_int:0{repeat_pad_width}d}"
    else:
        rid_text = str(repeat_id)

    if sid_int is not None:
        sid_text = f"{sid_int:0{sample_pad_width}d}"
    else:
        sid_text = str(sample_id)

    return f"mol_{rid_text}r_{sid_text}s"


def _derive_output_path(input_jsonl: Path, index_value: Any, out_dir: Path | None) -> Path:
    """Build default output path <stem>_<index>.smi."""
    directory = out_dir if out_dir is not None else input_jsonl.parent
    return directory / f"{input_jsonl.stem}_{index_value}.smi"


def build_smi_files(input_jsonl: Path, out_dir: Path | None = None, skip_null_smiles: bool = True) -> int:
    """Read input JSONL and write one .smi file per JSON object.

    Returns the number of written files.
    """
    written = 0

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}: {exc}") from exc

            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_num} JSON must be an object.")

            if "index" not in obj:
                raise ValueError(f"Line {line_num} missing required key 'index'.")
            if "samples" not in obj:
                raise ValueError(f"Line {line_num} missing required key 'samples'.")

            index_value = obj["index"]
            samples = obj["samples"]

            if not isinstance(samples, list):
                raise ValueError(f"Line {line_num} field 'samples' must be a list.")

            numeric_repeat_ids = [
                _safe_int(s.get("repeat_id"))
                for s in samples
                if isinstance(s, dict) and "repeat_id" in s
            ]
            numeric_repeat_ids = [rid for rid in numeric_repeat_ids if rid is not None]
            repeat_pad_width = max(1, len(str(max(numeric_repeat_ids)))) if numeric_repeat_ids else 1

            numeric_sample_ids = [
                _safe_int(s.get("sample_id"))
                for s in samples
                if isinstance(s, dict) and "sample_id" in s
            ]
            numeric_sample_ids = [sid for sid in numeric_sample_ids if sid is not None]
            sample_pad_width = max(1, len(str(max(numeric_sample_ids)))) if numeric_sample_ids else 1

            out_path = _derive_output_path(input_jsonl, index_value, out_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with out_path.open("w", encoding="utf-8") as out_f:
                for sample in samples:
                    if not isinstance(sample, dict):
                        continue

                    if "sample_id" not in sample:
                        continue
                    if "repeat_id" not in sample:
                        continue

                    smiles = sample.get("smiles")
                    if smiles is None and skip_null_smiles:
                        continue
                    if smiles is None:
                        smiles = ""

                    mol_label = _format_mol_label(
                        repeat_id=sample["repeat_id"],
                        sample_id=sample["sample_id"],
                        repeat_pad_width=repeat_pad_width,
                        sample_pad_width=sample_pad_width,
                    )
                    out_f.write(f"{smiles} {mol_label}\n")

            written += 1

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one .smi file per JSON object in a decoder JSONL file. "
            "Each line is: '<smiles> mol_<repeat_id>r_<sample_id>s'."
        )
    )
    parser.add_argument(
        "input_jsonl",
        type=Path,
        help="Path to input JSONL (e.g., preds_mu_prot.jsonl).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: same folder as input JSONL.",
    )
    parser.add_argument(
        "--keep-null-smiles",
        action="store_true",
        help="Keep entries with null smiles as empty smiles lines instead of skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_jsonl = args.input_jsonl

    if input_jsonl.suffix.lower() != ".jsonl":
        raise ValueError(f"Expected a .jsonl input, got: {input_jsonl}")
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    count = build_smi_files(
        input_jsonl=input_jsonl,
        out_dir=args.out_dir,
        skip_null_smiles=not args.keep_null_smiles,
    )
    print(f"Wrote {count} .smi file(s).")


if __name__ == "__main__":
    main()
