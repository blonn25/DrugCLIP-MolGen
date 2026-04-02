#!/bin/bash
#$ -q gpu.q
#$ -l h_rt=96:00:00
#$ -l compute_cap=70
#$ -l gpu_mem=12G
#$ -cwd
#$ -V
#$ -j y
#$ -o /wynton/group/bks/work/blonn25/software/drugclip_molgen/DrugCLIP-MolGen/logs/$JOB_NAME.$JOB_ID.$TASK_ID.log

# Array job: each task processes one LMDB file from the file list
# Handles growing LMDBs by processing in fixed-size chunks
# Usage: qsub encode_mols_chunked.sh [OUTPUT_BASE_DIR] [CHUNK_SIZE]

module load Sali cuda
conda activate drugclip

echo "=== Task $SGE_TASK_ID ==="
echo "Job $JOB_ID on host: $(hostname)"
echo "SGE_GPU=$SGE_GPU"
echo "LMDB_PATH=$LMDB_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CHUNK_SIZE=$CHUNK_SIZE"

export CUDA_VISIBLE_DEVICES="$SGE_GPU"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

device=cuda
results_path="./test"  # replace to your results path
batch_size=256
mol_path=/wynton/group/bks/work/blonn25/software/drugclip_molgen/DrugCLIP-MolGen/molgen_data/mol_embs/1M_chemstep_seedset/seed_smis_1M_canonical.lmdb # path to the molecule file
save_path=/wynton/group/bks/work/blonn25/software/drugclip_molgen/DrugCLIP-MolGen/molgen_data/mol_embs/1M_chemstep_seedset # path to the save dir

python ./unimol/encode_mols.py --user-dir ./unimol ./dict --valid-subset test \
       --results-path $results_path \
       --num-workers 0 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 256 \
       --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-path $mol_path \
       --save-dir $save_path \
       --write-h5
       # --cpu \
       # --start 0 --end 30000 