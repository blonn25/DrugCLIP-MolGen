#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=96:00:00
#$ -l mem_free=1G
#$ -pe smp 8
#$ -t 1-1
#$ -o /wynton/group/bks/work/blonn25/software/drugclip_molgen/DrugCLIP-MolGen/logs/$JOB_NAME.$JOB_ID.$TASK_ID.log

# # Create logs directory if it doesn't exist
# mkdir -p logs

# Get the file for this task
SMI_FILE=/wynton/group/bks/work/blonn25/software/drugclip_molgen/DrugCLIP-MolGen/molgen_data/mol_embs/13M_chemstep_seedset/seed_smis_13M.smi
BASENAME=$(basename "$SMI_FILE" .smi)
OUTPUT_DIR="/wynton/group/bks/work/blonn25/software/drugclip_molgen/DrugCLIP-MolGen/molgen_data/mol_embs/13M_chemstep_seedset"
mkdir -p "$OUTPUT_DIR"

echo "Processing: $SMI_FILE"
echo "Task ID: $SGE_TASK_ID"

# source /wynton/group/bks/work/bwhall61/drugclip/drugclip_env/bin/activate
conda activate drugclip
python ../smi_2_lmdb_restart.py "$SMI_FILE" "${OUTPUT_DIR}/${BASENAME}.lmdb" --n_cpu 8
