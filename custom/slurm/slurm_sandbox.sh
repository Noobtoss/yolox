#!/bin/bash
#SBATCH --job-name=yolox_train_arr # Kurzname des Jobs
#SBATCH --array=1
#SBATCH --output=logs/R-%A-%a.out
#SBATCH --partition=p2,p3,p4,p5,p6 # p4
#SBATCH --qos=preemptible          # gpuultimate
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

OUT_DIR="${ROOT_DIR}/tmp"
EXP_NAME="${KV[exp_name]:-unnamed_experiment}"
CFG="${KV[cfg]:-custom/exps/Images04.py}"
CKPT="${KV[ckpt]:-checkpoints/yolox_x.pth}"

# ----- ENVIRONMENT SETUP -------------------------------------------
module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate conda-yolox

export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

export TMPDIR=/nfs/scratch/staff/schmittth/tmp

# ----- WANDB -------------------------------------------------------
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_DIR=/nfs/scratch/staff/schmittth/tmp
export WANDB_CACHE_DIR=/nfs/scratch/staff/schmittth/tmp
export WANDB_CONFIG_DIR=/nfs/scratch/staff/schmittth/tmp

# ----- TRAINING ----------------------------------------------------
python tools/train.py \
    --exp_file $ROOT_DIR/$CFG \
    --devices 1 \
    --batch-size 8 \
    --fp16 \
    --occup \
    --ckpt $ROOT_DIR/$CKPT \
    --cache \
    --logger wandb \
        wandb-project tmp \
        wandb-entity team-noobtoss \
        wandb-name $EXP_NAME \
    output_dir $OUT_DIR \
    $PARAMS

# ----- CLEANUP -----------------------------------------------------
KEEP_FILES=("train_log.txt" "last_epoch_ckpt.pth")
eval find "$OUT_DIR/$EXP_NAME" -type f $(printf ' ! -name "%s"' "${KEEP_FILES[@]}") -delete
