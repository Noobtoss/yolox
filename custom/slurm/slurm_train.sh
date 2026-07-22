#!/bin/bash
#SBATCH --job-name=yolox_train        # Kurzname des Jobs
#SBATCH --output=logs/R-%j.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=1        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem-per-cpu=64G        # RAM pro CPU Kern #20G #32G #64G

# ----- ROOT_DIR ----------------------------------------------------
ROOT_DIR=/nfs/scratch/staff/schmittth/code_nexus/yolox

# ----- GET ARGS ----------------------------------------------------
EXP=${1:-custom/src/Images04.py}
CKPT=${2:-checkpoints/yolox_x.pth}

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
        wandb-project runs \
        wandb-entity team-noobtoss \
        wandb-name "$(basename "$CKPT" .pth)_$(basename "$EXP" .py)_$(date +"%Y-%m-%d_%H-%M")" \
        wandb-log_checkpoints False
