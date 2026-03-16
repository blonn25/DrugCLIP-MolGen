# MolGen Decoder

Transformer decoder pipeline for mapping DrugCLIP molecule embeddings to SELFIES token IDs.

## Setup

### Local Setup

Run the following in a directory where you would like to build the DrugCLIP-MolGen codebase:
```bash
# clone the DrugCLIP-MolGen repo
git clone https://github.com/blonn25/DrugCLIP-MolGen.git
conda create -n drugclip python=3.9 -y
conda activate drugclip
python -m pip install -r DrugCLIP-MolGen/docker/requirements.txt

# set up unicore (remove --enable-cuda-ext for CPU-only or MPS gpu install)
git clone https://github.com/dptech-corp/Uni-Core.git
cd Uni-Core
python setup.py install	--enable-cuda-ext

# move back into the molgen directory and copy model weights to the directory
cd ../DrugCLIP-MolGen
cp -r /wynton/group/bks/work/blonn25/software/drugclip_molgen/DrugCLIP-MolGen/drugclip_model_weights .

# copy exisitng model weights and training output to continue training or to use for inference
mkdir molgen_data/training_output
cp -r /wynton/group/bks/work/blonn25/software/drugclip_molgen/DrugCLIP-MolGen/molgen_data/training_output/decoder_macbook_air_v2_randmols_seed42 molgen_data/training_output/
```

### Setup on Wynton

Run the following in a directory where you would like to build the DrugCLIP-MolGen codebase (Uni-Core installed via wheels instead of cloning it):
```bash
# clone the DrugCLIP-MolGen repo
git clone https://github.com/blonn25/DrugCLIP-MolGen.git
conda create -n drugclip python=3.9 -y
conda activate drugclip
python -m pip install -r DrugCLIP-MolGen/docker/requirements_wynton.txt

# move back into the molgen directory and copy model weights to the directory
cd ../DrugCLIP-MolGen
cp -r /wynton/group/bks/work/blonn25/software/drugclip_molgen/DrugCLIP-MolGen/drugclip_model_weights .

# copy exisitng model weights and training output to continue training or to use for inference
mkdir molgen_data/training_output
cp -r /wynton/group/bks/work/blonn25/software/drugclip_molgen/DrugCLIP-MolGen/molgen_data/training_output/decoder_macbook_air_v2_randmols_seed42 molgen_data/training_output/
```

## Files

### Model, Training, and Inference Files (see example usage below)

- `selfies_decoder.py`: model + tokenizer + autoregressive generation.
- `train_decoder.py`: training loop from `mol_reps` H5 -> `token_ids` H5.
- `infer_decoder.py`: checkpoint inference; writes token IDs and optional SELFIES/SMILES strings.

### Data Preparation Files (see example usage commented at the top of each script)

- `encode_mols.sh` / `encode_pocket.sh`: DrugCLIP scripts to generate molecule embeddings (h5 format) and pockets (lmdb format which can be processed with `molgen_data/encode_pocket_multifold.py`) to create h5 files for inference.
- `molgen_data/smi_2_selfie.py`: converts SMILES strings to SELFIES strings (and a padded version) in preparation for tokenization.
- `molgen_data/selfie_2_tokens.py`: converts SELFIES strings to token ID sequences and saves in H5 format for training.
- `molgen_data/encode_pocket_multifold.py`: encodes pocket pocket and saves an h5 file which can be used for inference with the trained decoder.

### Inference Post-Processing Files (see example usage commented at the top of each script)

- `molgen_data/build_smi_from_jsonl.py`: builds SMILES from JSONL inference output.

## Data Assumptions
- Embeddings H5 contains dataset: `mol_reps` with shape `(N, D)`.
- Token H5 contains datasets: `token_ids` `(N, L)` and `vocab` `(V,)`.
- Rows are aligned between embedding and token files.

## Training Example Usage

### Large-scale
```bash
cd /Users/beaulonnquist/projects/shoichet_lab/drugclip_mol_gen/DrugCLIP/MolGen_Decoder
conda activate drugclip 
python train_decoder.py \
  --embedding-h5 ../molgen_data/mol_embs/first_100k_embeddings.h5 \
  --token-h5 ../molgen_data/mol_embs/first_100k.selfies.tokens.h5 \
  --output-dir ../molgen_data/training_output/decoder_v1 \
  --epochs 30 \
  --batch-size 256 \
  --lr 3e-4 \
  --d-model 512 \
  --nhead 8 \
  --num-layers 6 \
  --dim-feedforward 2048 \
  --device auto
```

### Small-scale (training on Mac for quick iteration)
```bash
cd /Users/beaulonnquist/projects/shoichet_lab/drugclip_mol_gen/DrugCLIP/MolGen_Decoder
conda activate drugclip 
python train_decoder.py \
  --embedding-h5 ../molgen_data/mol_embs/random_mols/random_mols_embeddings.h5 \
  --token-h5 ../molgen_data/mol_embs/random_mols/random_mols.tokens.h5 \
  --output-dir ../molgen_data/training_output/decoder_macbook_air_v2_randmols_seed42 \
  --epochs 30 \
  --batch-size 64 \
  --lr 2e-4 \
  --weight-decay 1e-2 \
  --val-split 0.10 \
  --d-model 384 \
  --nhead 6 \
  --num-layers 4 \
  --dim-feedforward 1536 \
  --dropout 0.1 \
  --num-workers 0 \
  --device mps \
  --seed 42 \
  --log-every 250
```

Outputs:
- `decoder_best.pt`
- `decoder_last.pt`
- `train_history.json`

### Resuming Training From Checkpoint
Use `--resume-from` with a previous `decoder_last.pt` to continue optimization.
Set `--epochs` to the final epoch you want to reach (not additional epochs).

```bash
cd /Users/beaulonnquist/projects/shoichet_lab/drugclip_mol_gen/DrugCLIP/MolGen_Decoder
conda activate drugclip
python train_decoder.py \
  --embedding-h5 ../molgen_data/mol_embs/random_mols/random_mols_embeddings.h5 \
  --token-h5 ../molgen_data/mol_embs/random_mols/random_mols.tokens.h5 \
  --output-dir ../molgen_data/training_output/decoder_macbook_air_v2_randmols_seed42 \
  --epochs 60 \
  --batch-size 64 \
  --lr 2e-4 \
  --weight-decay 1e-2 \
  --val-split 0.10 \
  --d-model 384 \
  --nhead 6 \
  --num-layers 4 \
  --dim-feedforward 1536 \
  --dropout 0.1 \
  --num-workers 0 \
  --device mps \
  --seed 42 \
  --log-every 250 \
  --resume-from ../molgen_data/training_output/decoder_macbook_air_v2_randmols_seed42/decoder_last.pt
```

## Inference Example Usage

### Infer on first 1000 training samples
```bash
cd /Users/beaulonnquist/projects/shoichet_lab/drugclip_mol_gen/DrugCLIP/MolGen_Decoder
conda activate drugclip 
python infer_decoder.py \
  --checkpoint ../molgen_data/training_output/decoder_macbook_air_v2_randmols_seed42/decoder_best.pt \
  --embedding-h5 ../molgen_data/mol_embs/random_mols/random_mols_embeddings.h5 \
  --output-jsonl ../molgen_data/training_output/decoder_macbook_air_v2_randmols_seed42/preds_0_1000.jsonl \
  --start 0 \
  --end 500 \
  --batch-size 256 \
  --num-samples 1 \
  --max-new-tokens 80 \
  --device auto
```

### Infer on the mu opiod receptor
```bash
# output length up to 80 tokens, create 100 noisy copies per input embedding,
# then generate 10 decoding sample for each copy (1000 total per input embedding),
# default temperature for sampling (no scaling), sample from all logits (no top-k filtering),
# and use stochastic sampling instead of greedy decoding.
cd /Users/beaulonnquist/projects/shoichet_lab/drugclip_mol_gen/DrugCLIP/MolGen_Decoder
conda activate drugclip 
python infer_decoder.py \
  --checkpoint ../molgen_data/training_output/decoder_macbook_air_v2_randmols_seed42/decoder_best.pt \
  --embedding-h5 ../molgen_data/pocket_embs/mu_prot.h5 \
  --output-jsonl ../molgen_data/inference_output/random_mols/preds_mu_prot_100_repeats_10_samples.jsonl \
  --max-new-tokens 80 \
  --embedding-repeat 100 \
  --embedding-noise-strength 0.15 \
  --noise-seed 42 \
  --num-samples 10 \
  --temperature 1.0 \
  --top-k 0 \
  --stochastic-sampling
```

Notes for repeat/noise sampling:
- `--batch-size` is a target decode batch size.
- Internally, DataLoader batch size is set to `max(1, batch_size // embedding_repeat)` before repeat expansion.
- If `embedding_repeat > batch_size`, DataLoader falls back to 1 original row per step, so effective decode batch becomes `embedding_repeat`.
- Total decoded samples per original input embedding are:
  `embedding_repeat * num_samples`.
- Noise is applied independently to every effective decode embedding (including repeats).
- Noise strength uses a unit-sphere-aware Gaussian perturbation and then re-normalizes each embedding to L2 norm 1.

Each JSONL line contains:
- `index`
- `samples`: list of decoded samples for that embedding, where each sample has:
- `sample_id`
- `repeat_id`
- `token_ids`
- `selfies`
- `smiles` (or `null` if SELFIES decode fails)

Example JSONL record:
```json
{
  "index": 0,
  "samples": [
    {
      "sample_id": 0,
      "repeat_id": 0,
      "token_ids": [33, 13, 20, 33],
      "selfies": "[O][=C][Branch1][O]",
      "smiles": "O=CO"
    },
    {
      "sample_id": 1,
      "repeat_id": 0,
      "token_ids": [26, 26, 31, 20],
      "selfies": "[C][C][N][Branch1]",
      "smiles": "CCN"
    }
  ]
}
```

## Notes
- Training does not perform token-id -> string conversion.
- Conversion to SELFIES/SMILES is inference-only by design.
- `--max-samples` in training is useful for quick smoke tests.
