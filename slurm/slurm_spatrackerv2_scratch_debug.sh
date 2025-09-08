#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=spatrackerv2_scratch
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
#SBATCH --time=00:20:00
#
# Specify number of tasks.
#SBATCH --ntasks=1
#
# Specify number of CPU cores per task.
#SBATCH --cpus-per-task=1
#
# Specify memory limit per CPU core.
#SBATCH --mem-per-cpu=8192
#
# Specify number of required GPUs.
#SBATCH --gpus=rtx_4090:1
# #SBATCH --gpus=a100:1

echo "=== Job starting on $(hostname) at $(date) ==="
# DATE_VAR=$(date +%Y%m%d%H%M%S)

# Load modules.
module load stack/2024-06 python/3.11 eth_proxy
echo "Loaded modules: $(module list 2>&1)"

# Activate virtual environment for SpatialTrackerV2.
source /cluster/scratch/niacobone/SpaTrackerV2/myenv/bin/activate
echo "Activated Python venv: $(which python)"

# Execute

include_list=("test_0", "test_1", "test_2", "video_01_static_short", "video_02_static_medium", "video_03_static_long", "video_04_static_long")  # Inserisci qui i nomi dei video (senza estensione)

cd /cluster/scratch/niacobone/SpaTrackerV2
echo "Starting SpaTrackerV2 inference..."
echo "----------------------------------"

for video_name in "${include_list[@]}"; do
    echo "Processing video: $video_name"
    python inference.py --video_name="$video_name"
    echo "----------------------------------"
done

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"