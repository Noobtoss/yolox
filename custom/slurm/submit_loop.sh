#!/bin/bash
# submit_loop.sh
#
# Usage:
#   ./submit_loop.sh [slurm_script] [array] [qos] [loop_num]
#
# Make executable:
#   chmod +x submit_loop.sh
#
# Run:
#   ./submit_loop.sh
#
# Run in background:
#   nohup ./submit_loop.sh > submit_loop.out 2>&1 &
#
# Monitor:
#   tail -f submit_loop.out
#
# Stop:
#   pkill -f submit_loop.sh

# ----- GET ARGS ----------------------------------------------------
SLURM_SCRIPT=${1:-custom/slurm/slurm_train.sh}
ARRAY=${2:-1-5%3}
QOS=${3:-gpuultimate}
LOOP_NUM=${4:-12}

echo "Script: $SLURM_SCRIPT | Array: $ARRAY | QOS: $QOS | Loops: $LOOP_NUM"

# ----- GET SLURM LIMITS --------------------------------------------
SUBMIT_LIMIT=$(sacctmgr show qos format=name,maxsubmitpu -p -n | awk -F'|' -v qos="$QOS" '$1 == qos {print $2}')
RUNNING_LIMIT=$(sacctmgr show qos format=name,maxjobspu -p -n | awk -F'|' -v qos="$QOS" '$1 == qos {print $2}')

# ----- GET SUBMIT SIZE ---------------------------------------------
SUBMIT_SIZE=0
IFS=',' read -ra PARTS <<< "${ARRAY%%%*}"

for part in "${PARTS[@]}"; do
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        SUBMIT_SIZE=$((SUBMIT_SIZE + BASH_REMATCH[2] - BASH_REMATCH[1] + 1))
    else
        SUBMIT_SIZE=$((SUBMIT_SIZE + 1))
    fi
done

if (( SUBMIT_SIZE > SUBMIT_LIMIT )); then
    echo "ERROR: SUBMIT_SIZE=$SUBMIT_SIZE exceeds SUBMIT_LIMIT=$SUBMIT_LIMIT"
    exit 1
fi

# ----- SUBMIT_LOOPS ------------------------------------------------
for ((i=1; i<=LOOP_NUM; i++)); do
    sleep 30

    while :; do
        CURRENT=$(squeue -u "$USER" -h -r | wc -l)
        FREE=$((SUBMIT_LIMIT - CURRENT))

        (( FREE < 0 )) && FREE=0

        if (( FREE >= SUBMIT_SIZE )); then
            break
        fi
        sleep 600
    done

    out=$(sbatch --qos="$QOS" --array="$ARRAY" "$SLURM_SCRIPT")
    echo "[$(date)] $out" >> submitted_jobs.log
done
