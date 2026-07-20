#!/bin/bash
#SBATCH --job-name=yolox_train_arr # Kurzname des Jobs
#SBATCH --array=1,8-19%4
#SBATCH --output=logs/R-%A-%a.out
#SBATCH --partition=p2,p6             # p4
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                  # Anzahl Knoten
#SBATCH --ntasks=1                 # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=1          # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem-per-cpu=64G          # RAM pro CPU Kern #20G #32G #64G

# ----- ROOT_DIR ----------------------------------------------------
ROOT_DIR=/nfs/scratch/staff/schmittth/code_nexus/yolox

# ----- GET ARGS ----------------------------------------------------
PARAMS_FILE="$ROOT_DIR/custom/slurm/slurm_params.txt"
PARAMS=$(grep -v '^[[:space:]]*#' "$PARAMS_FILE" | sed -n "$((SLURM_ARRAY_TASK_ID))p")

# Add SLURM_ARRAY_JOB_ID and SLURM_ARRAY_TASK_ID to exp_name
PARAMS=$(echo "$PARAMS" | sed -E "s/(exp_name[[:space:]]+[^[:space:]]+)/\1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/")
declare -A KV
read -r -a ARR <<< "$PARAMS"
for ((i=0; i<${#ARR[@]}; i+=2)); do
    key="${ARR[$i]}"
    value="${ARR[$i+1]}"
    KV["$key"]="$value"
done
[[ "$PARAMS" != *"seed"* ]] && PARAMS="$PARAMS seed ${SLURM_ARRAY_JOB_ID}"

echo $KV

OUT_DIR="${ROOT_DIR}/runs"
EXP_NAME="${KV[exp_name]:-unnamed_experiment}"
EXP="${KV[exp]:-custom/src/Images04.py}"
CKPT="${KV[ckpt]:-checkpoints/yolox_x.pth}"

# ----- ENVIRONMENT SETUP -------------------------------------------
module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate conda-yolox

export PYTHONPATH="$ROOT_DIR/custom/src:$PYTHONPATH"
export TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/yolox_${SLURM_JOB_ID}_XXXXXX")

# ----- WANDB -------------------------------------------------------
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_CACHE_DIR=$TMPDIR
export WANDB_DATA_DIR=$TMPDIR
export WANDB_DIR=$TMPDIR
export WANDB_CONFIG_DIR=$TMPDIR

# ----- TRAINING ----------------------------------------------------
python tools/train.py \
    --exp_file $ROOT_DIR/$EXP \
    --devices 1 \
    --batch-size 8 \
    --fp16 \
    --occup \
    --ckpt $ROOT_DIR/$CKPT \
    --cache \
    --logger wandb \
        wandb-project runs-yolox \
        wandb-entity team-noobtoss \
        wandb-name $EXP_NAME \
        wandb-log_checkpoints False \
    output_dir $OUT_DIR \
    $PARAMS

# ----- CLEANUP -----------------------------------------------------
wandb sync --sync-all || true
rm -rf "$TMPDIR"
KEEP_FILES=("train_log.txt" "last_epoch_ckpt.pth")
eval find "$OUT_DIR/$EXP_NAME" -type f $(printf ' ! -name "%s"' "${KEEP_FILES[@]}") -delete
