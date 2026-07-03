#!/bin/bash
#SBATCH --job-name=yolox_train_arr # Kurzname des Jobs
#SBATCH --array=44,45%8  # Previous runs: 17-40%8, 9-16%8
#SBATCH --output=logs/R-%A-%a.out
#SBATCH --gres=gpu:a40:1     # Request 1x A40 GPUs
#SBATCH --partition=a40      # Submit to the a40 node partition
#SBATCH --ntasks=1           # 1 process total (not MPI)
#SBATCH --ntasks-per-node=1  # That 1 process runs on 1 node
#SBATCH --cpus-per-task=4    # 4 CPU cores for that process (data loading etc)
#SBATCH --time=02:40:32      # Walltime limit: kill job after 3hr 52min 32sec
#SBATCH --mail-type=ALL      # Email on job start, end, fail
#SBATCH --mail-user=thomas.schmitt@th-nuernberg.de

# ----- BASE_DIR ----------------------------------------------------
BASE_DIR="$WORK/code_nexus/yolox"
JOB_DIR=$TMPDIR

# ----- GET ARGS ----------------------------------------------------
PARAMS_FILE="$BASE_DIR/custom/slurm/alex/slurm_params.txt"
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

OUTPUT_DIR="${BASE_DIR}/runs"
EXP_NAME="${KV[exp_name]:-unnamed_experiment}"
EXP="${KV[exp]:-custom/exps/Images04.py}"
CKPT="${KV[ckpt]:-checkpoints/yolox_x.pth}"
DATA_DIR="${KV[data_dir]:-datasets_tar/Images05MetaFood2026.tar}"

# ----- ENVIRONMENT SETUP -------------------------------------------
unset SLURM_EXPORT_ENV

module purge
module load python/3.12-base
module load cuda/12.8.1

conda activate conda-yolox

export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# --- PROXY  --------------------------------------------------------
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# ----- WANDB -------------------------------------------------------
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_DIR=$TMPDIR
export WANDB_CACHE_DIR=$TMPDIR
export WANDB_CONFIG_DIR=$TMPDIR

# ----- DATA STAGING ------------------------------------------------
tar xf $BASE_DIR/$DATA_DIR --strip-components=1 -C $JOB_DIR \
  --warning=no-unknown-keyword \
  --exclude='._*' \
  --exclude='.DS_Store' \
  --exclude='__MACOSX'

echo ErrorMessage unpacking: $?  # $? = exit code (0 = success, anything else = error)

DATA_DIR=$JOB_DIR
PARAMS=$(echo "$PARAMS" | sed "s|data_dir [^ ]*|data_dir $DATA_DIR|")
echo $DATA_DIR
echo $JOB_DIR

# ----- TRAINING ----------------------------------------------------
python tools/train.py \
    --exp_file $BASE_DIR/$EXP \
    --devices 1 \
    --batch-size 8 \
    --fp16 \
    --occup \
    --ckpt $BASE_DIR/$CKPT \
    --cache \
    --logger wandb \
        wandb-project runs-alex \
        wandb-entity team-noobtoss \
        wandb-name $EXP_NAME \
        wandb-log_checkpoints False \
    output_dir $OUTPUT_DIR \
    data_dir $DATA_DIR \
    $PARAMS

# ----- CLEANUP -----------------------------------------------------
KEEP_FILES=("train_log.txt" "last_epoch_ckpt.pth")
eval find "$OUTPUT_DIR/$EXP_NAME" -type f $(printf ' ! -name "%s"' "${KEEP_FILES[@]}") -delete
