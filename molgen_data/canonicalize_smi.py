'''
Canonicalize a SMILES string using RDKit.
Usage:
    python canonicalize_smi.py <input_smi> <output_smi>
Example:
    python canonicalize_smi.py input.smi output.smi
    python molgen_data/canonicalize_smi.py --input-smi molgen_data/mol_embs/1M_chemstep_seedset/seed_smis_1M.smi
'''

import argparse
from rdkit import Chem

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Canonicalize a SMILES file using RDKit.")
    parser.add_argument(
        "--input-smi",
        type=str,
        required=True,
        help="Path to input SMILES file",
    )
    parser.add_argument(
        "--output-smi",
        type=str,
        default=None,
        help="Path to output SMILES file",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100000,
        help="Log progress every N processed molecules (default: 100000)",
    )
    args = parser.parse_args()

    if args.output_smi is None:
        args.output_smi = args.input_smi.rsplit(".", 1)[0] + "_canonical.smi"

    total_lines = 0
    written_count = 0
    skipped_count = 0

    with open(args.input_smi, "r") as f_in:
        with open(args.output_smi, "w") as f_out:
            for line_num, line in enumerate(f_in, start=1):
                total_lines += 1
                tokens = line.strip().split()

                if not tokens:
                    skipped_count += 1
                    print(f"Skipping line {line_num}: empty or whitespace-only line")
                    continue

                smi_str = tokens[0]
                identifiers = " ".join(tokens[1:])  # Any additional identifiers (e.g., molecule name)

                try:
                    mol = Chem.MolFromSmiles(smi_str)
                    if mol is None:
                        raise ValueError("RDKit could not parse SMILES")
                    canonical_smi = Chem.MolToSmiles(mol)
                except Exception as exc:
                    skipped_count += 1
                    print(f"Skipping line {line_num}: {smi_str} ({exc})")
                    continue

                if identifiers:
                    f_out.write(f"{canonical_smi} {identifiers}\n")
                else:
                    f_out.write(f"{canonical_smi}\n")
                written_count += 1

                if args.log_every > 0 and total_lines % args.log_every == 0:
                    print(
                        f"Processed {total_lines:,} lines | "
                        f"written: {written_count:,} | skipped: {skipped_count:,}"
                    )

    print(
        f"Done. Output: {args.output_smi} | "
        f"total lines: {total_lines:,} | written: {written_count:,} | skipped: {skipped_count:,}"
    )