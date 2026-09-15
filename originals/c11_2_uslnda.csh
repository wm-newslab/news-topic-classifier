#!/bin/tcsh
#SBATCH -J c11_2_uslnda
#SBATCH --output=logs/c11_2_uslnda_%A_%a.out
#SBATCH --error=logs/c11_2_uslnda_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-15

# Parallel design:
#   - This Slurm array has 16 workers (task indexes 0-15).
#   - Each worker receives one GPU and loads one copy of the base model + LoRA adapter.
#   - The sorted parquet-file list is divided deterministically across workers.
#   - Within each GPU worker, articles are classified in batches.
#
# Usage:
#   sbatch classify_uslnda_news_parallel.csh 2026-01-01 2026-12-31
#
# To change the number of parallel GPU workers, change BOTH:
#   #SBATCH --array=0-15
#   set NUM_SHARDS = 16

set DEFAULT_START_DATE = "2025-11-16"
set DEFAULT_END_DATE = "2025-11-30"

if ( $#argv >= 1 ) then
    set START_DATE = "$argv[1]"
else
    set START_DATE = "$DEFAULT_START_DATE"
endif

if ( $#argv >= 2 ) then
    set END_DATE = "$argv[2]"
else
    set END_DATE = "$DEFAULT_END_DATE"
endif

if ( $#argv > 2 ) then
    echo "ERROR: Too many arguments"
    echo "Usage: sbatch $0 [START_DATE] [END_DATE]"
    exit 1
endif

# -----------------------------------------------------------------------------
# EDIT THESE ABSOLUTE PATHS
# -----------------------------------------------------------------------------
set CLASSIFIER_DIR = "./"
set TRAINING_PROJECT_DIR = "/sciclone/data10/gchewababarand/community-news-alignment/0-comm-in/LN"
set IN_ROOT = "/sciclone/data10/gchewababarand/0-data/USLNDA/data/USLNDA/USLNDA"
set OUT_ROOT = "/sciclone/data10/gchewababarand/0-data/USLNDA/data/USLNDA/TOPIC_CLASSIFICATION"

set PIPELINE_DIR = "$TRAINING_PROJECT_DIR/lora_pipeline_run"
set EXPERIMENT_ROOT = "$TRAINING_PROJECT_DIR/lora_local_news_experiments"
set PYTHON_SCRIPT = "$CLASSIFIER_DIR/classify_uslnda_news_parallel.py"
set BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Must match the size of #SBATCH --array above.
set NUM_SHARDS = 16
set SHARD_INDEX = "$SLURM_ARRAY_TASK_ID"

# Number of articles generated together on each GPU. Increase gradually if GPU
# memory permits (for example 8 -> 12 -> 16). Reduce it if CUDA OOM occurs.
set BATCH_SIZE = 8
set MAX_SEQ_LENGTH = 2048

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
cd "$CLASSIFIER_DIR"
mkdir -p logs outputs

setenv HF_HUB_DISABLE_XET 1
setenv HF_HOME "/sciclone/data10/gchewababarand/huggingface_cache"
setenv HF_HUB_CACHE "$HF_HOME/hub"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

module load python
source /sciclone/data10/gchewababarand/Nationalization/llnr/bin/activate.csh

if ( ! -f "$PYTHON_SCRIPT" ) then
    echo "ERROR: Missing Python script: $PYTHON_SCRIPT"
    exit 2
endif
if ( ! -d "$IN_ROOT" ) then
    echo "ERROR: Input root does not exist: $IN_ROOT"
    exit 3
endif
if ( ! -f "$PIPELINE_DIR/cv_winner.json" ) then
    echo "ERROR: Missing winning-model file: $PIPELINE_DIR/cv_winner.json"
    exit 4
endif
if ( ! -d "$EXPERIMENT_ROOT" ) then
    echo "ERROR: Experiment root does not exist: $EXPERIMENT_ROOT"
    exit 5
endif

set START_TAG = `echo "$START_DATE" | tr -d '-'`
set END_TAG = `echo "$END_DATE" | tr -d '-'`
set RUN_LOG = "$CLASSIFIER_DIR/outputs/c11_2_uslnda_${START_TAG}_${END_TAG}_${SLURM_ARRAY_JOB_ID}_${SHARD_INDEX}.log"

echo "============================================================"
echo "Job started: `date`"
echo "Host: `hostname`"
echo "Array job: $SLURM_ARRAY_JOB_ID"
echo "Shard: $SHARD_INDEX / $NUM_SHARDS"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Input root: $IN_ROOT"
echo "Output root: $OUT_ROOT"
echo "Date range: $START_DATE through $END_DATE"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Detailed log: $RUN_LOG"
echo "============================================================"

python -u "$PYTHON_SCRIPT" \
    --in-root "$IN_ROOT" \
    --out-root "$OUT_ROOT" \
    --pipeline-dir "$PIPELINE_DIR" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --training-project-dir "$TRAINING_PROJECT_DIR" \
    --base-model "$BASE_MODEL" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --batch-size "$BATCH_SIZE" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --truncation-strategy head_tail \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    >& "$RUN_LOG"

set STATUS = $status
echo "Shard $SHARD_INDEX exit status: $STATUS"
echo "Job finished: `date`"
exit $STATUS
