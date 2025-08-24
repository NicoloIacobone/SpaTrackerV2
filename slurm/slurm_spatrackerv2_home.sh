#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=spatrackerv2_home
#
# Specify output file.
#SBATCH --output=spatrackerv2_%j.log
#
# Specify error file.
#SBATCH --error=spatrackerv2_%j.err
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify time limit.
#SBATCH --time=00:05:00
#
# Specify number of tasks.
#SBATCH --ntasks=1
#
# Specify number of CPU cores per task.
#SBATCH --cpus-per-task=8
#
# Specify number of required GPUs.
#SBATCH --gpus=rtx_4090:2

echo "=== Job starting on $(hostname) at $(date) ==="
# DATE_VAR=$(date +%Y%m%d%H%M%S)

# Specify directories.
# export REPO="/cluster/work/igp_psr/niacobone/SpaTrackerV2"

# Load modules.
module load stack/2024-06 python/3.11
echo "Loaded modules: $(module list 2>&1)"

# Activate virtual environment for SpatialTrackerV2.
source /cluster/home/niacobone/projects/SpaTrackerV2/myenv/bin/activate
echo "Activated Python venv: $(which python)"


# Execute
cd /cluster/home/niacobone/projects/SpaTrackerV2
echo "Starting SpaTrackerV2 inference..."
python inference.py

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"
