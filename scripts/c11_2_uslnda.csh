#!/bin/tcsh
#SBATCH --job-name=localnews-topic
#SBATCH --output=logs/classification_%A_%a.out
#SBATCH --error=logs/classification_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-15
# Submit from repository root with an activated Python environment.
if ( $#argv != 4 ) then
    echo "Usage: sbatch scripts/c11_2_uslnda.csh INPUT_ROOT OUTPUT_ROOT START_DATE END_DATE"
    exit 2
endif
cd "$SLURM_SUBMIT_DIR"
python -u src/classification/classify_uslnda_news_parallel.py \
    --adapter-dir models/adapter \
    --in-root "$argv[1]" --out-root "$argv[2]" \
    --start-date "$argv[3]" --end-date "$argv[4]" \
    --num-shards "$SLURM_ARRAY_TASK_COUNT" --shard-index "$SLURM_ARRAY_TASK_ID" \
    --batch-size 8 --max-seq-length 2048 --truncation-strategy head_tail
exit $status
