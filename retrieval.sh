
#!/usr/bin/env bash

# run with: bash retrieval.sh GPU-8c25cdee-043b-c774-b6d7-cecf60554421

# If use_cache=True, pre-encoded molecules are used for screening.
# This is default for wet-lab experiment targets.
# Otherwise, set MOL_PATH to the lmdb path for your screening library.


echo "First argument: $1"   # set to a cuda device if desired

MOL_PATH="/wynton/group/bks/work/blonn25/software/drugclip_molgen/DrugCLIP-MolGen/molgen_data/inference_output/random_mols/opt_scores/preds_mu_prot_100_repeats_10_samples-opt_0.lmdb" # path to the molecule file
POCKET_PATH="/wynton/group/bks/work/blonn25/software/drugclip_molgen/DrugCLIP-MolGen/molgen_data/pocket_embs/mu_prot.lmdb"
FOLD_VERSION=6_folds
# use_cache=True
use_cache=False
save_path="/wynton/group/bks/work/blonn25/software/drugclip_molgen/DrugCLIP-MolGen/molgen_data/inference_output/random_mols/opt_scores/NET.txt"
frac_to_save=1.0     # save all scores

# PyTorch 2.6+ defaults torch.load(weights_only=True). unicore calls torch.load
# internally without overriding this, so force legacy behavior for trusted checkpoints.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1




CUDA_VISIBLE_DEVICES="1" python ./unimol/retrieval.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --cpu \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --pocket-path $POCKET_PATH \
       --fold-version $FOLD_VERSION \
       --use-cache $use_cache \
       --save-path $save_path \
       --frac-to-save $frac_to_save \
       --turn-off-scaling   # turns of the z-score scaling for fair comparisons