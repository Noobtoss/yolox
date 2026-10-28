#!/bin/bash
#SBATCH --job-name=yolox # Kurzname des Jobs
#SBATCH --array=1-3%3
#SBATCH --output=logs/%A_%a.out
#SBATCH --partition=p4,p6             # p4
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                  # Anzahl Knoten
#SBATCH --ntasks=1                 # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=1          # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem-per-cpu=64G          # RAM pro CPU Kern #20G #32G #64G

# ----- DIRS --------------------------------------------------------
ROOT=/nfs/scratch/staff/schmittth/code_hub/yolox
export TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/yolox_${SLURM_JOB_ID}_XXXXXX")

# ----- ENVIRONMENT SETUP -------------------------------------------
module purge
module load python/anaconda3

eval "$(conda shell.bash hook)"
conda activate conda-yolox

export PYTHONPATH="$ROOT/custom/src:$PYTHONPATH"

# ----- GET ARGS ----------------------------------------------------
EXP_MAP_FILE="$ROOT/custom/slurm/experiments_map.txt"
EXP_ARGS_FILE="$ROOT/custom/slurm/experiments.json"

EXP_NAME=$(grep -v '^[[:space:]]*#' "$EXP_MAP_FILE" | awk -v id="$SLURM_ARRAY_TASK_ID" '$1 == id {print $2; exit}')
if [[ -z "$EXP_NAME" ]]; then
    echo "ERROR: no exp_name for task id $SLURM_ARRAY_TASK_ID in $EXP_MAP_FILE" >&2
    exit 1
fi

ARGS=$(python -c '
import json, sys
exp = json.load(open(sys.argv[1]))[sys.argv[2]]
print(" ".join(f"{k} {v}" for k, v in exp.items()))
' "$EXP_ARGS_FILE" "$EXP_NAME") || { echo "ERROR: exp_name '$EXP_NAME' not found in $EXP_ARGS_FILE" >&2; exit 1; }
ARGS="exp_name $EXP_NAME $ARGS"

declare -A KV
read -r -a ARR <<< "$ARGS"
for ((i=0; i<${#ARR[@]}; i+=2)); do
    key="${ARR[$i]}"
    value="${ARR[$i+1]}"
    KV["$key"]="$value"
done

# ----- SET ARGS ----------------------------------------------------
[[ "$ARGS" != *"seed"* ]] && ARGS="$ARGS seed ${SLURM_ARRAY_JOB_ID}"

RUN_NAME="${KV[exp_name]:-unnamed_run}"
RUN_NAME="${RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUT_DIR="${ROOT}/runs/${RUN_NAME}"
EXP="${KV[exp]:-custom/src/Images04.py}"
CKPT="${KV[ckpt]:-checkpoints/yolox_x.pth}"

ARGS=$(echo "$ARGS" | sed -E "s/(exp_name[[:space:]]+)[^[:space:]]+/\1${RUN_NAME}/")

# ----- WANDB -------------------------------------------------------
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_CACHE_DIR=$TMPDIR
export WANDB_DATA_DIR=$TMPDIR
export WANDB_DIR=$TMPDIR
export WANDB_CONFIG_DIR=$TMPDIR

# ----- TRAINING ----------------------------------------------------
python tools/train.py \
    --exp_file $ROOT/$EXP \
    --devices 1 \
    --batch-size 8 \
    --fp16 \
    --occup \
    --ckpt $ROOT/$CKPT \
    --cache \
    --logger wandb \
        wandb-project runs-yolox \
        wandb-entity team-noobtoss \
        wandb-name $RUN_NAME \
        wandb-log_checkpoints False \
    output_dir "${OUT_DIR%/$RUN_NAME}" \
    $ARGS

# ----- CLEANUP -----------------------------------------------------
[[ -f "$OUT_DIR/last_epoch_ckpt.pth" ]] && mv -f "$OUT_DIR/last_epoch_ckpt.pth" "$OUT_DIR/${RUN_NAME}.pth"
KEEP_FILES=("train_log.txt" "${RUN_NAME}.pth")

for i in 1 2 3 4 5; do
    wandb sync --sync-all && break
    echo "sync attempt $i failed, retrying..."
    sleep 20
done

rm -rf "$TMPDIR"
find "$OUT_DIR" -type f $(printf ' ! -name %s' "${KEEP_FILES[@]}") -delete
