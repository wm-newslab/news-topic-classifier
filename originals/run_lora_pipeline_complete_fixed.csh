#!/bin/tcsh
# ============================================================================
# Automated Slurm workflow for LoRA local-news classification:
#
# random search -> rank screening runs -> CV for top configurations
# -> rank CV runs -> final training -> one-time final-test evaluation
# -> consolidated report
#
# Required files in the same working directory:
#   lora_pipeline.py                 (pipeline manager)
#   lora_local_news_tuning.py        (training/evaluation script)
#   data/labels_with_title.csv
#   data/gangani_local_news_labeling_round_2_v2.csv
#
# Start from the login node:
#   setenv HF_TOKEN your_huggingface_token
#   tcsh run_lora_pipeline_complete_fixed.csh submit
#
# Optional controls:
#   setenv N_RANDOM_TRIALS 12
#   setenv TOP_K_FOR_CV 2
#   setenv PIPELINE_SEED 20260728
#   setenv PIPELINE_DIR lora_pipeline_run
#
# Optional smoke test:
#   setenv MAX_ROWS 120
#   setenv N_RANDOM_TRIALS 3
#   setenv TOP_K_FOR_CV 1
#   setenv PIPELINE_DIR lora_pipeline_smoke_test
#   tcsh run_lora_pipeline_complete_fixed.csh submit
#
# The initial command is run with tcsh, not sbatch. This script submits all
# search, ranking, CV, final-training, testing, and reporting jobs itself.
# ============================================================================

# Default Slurm values. These apply if a worker stage is submitted manually.
#SBATCH --job-name=ln-lora-pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/lora_pipeline_%A_%a.out
#SBATCH --error=logs/lora_pipeline_%A_%a.err

# ----------------------------------------------------------------------------
# Resolve action safely. Block-form conditionals avoid tcsh argv expansion
# errors such as: argv: Subscript out of range.
# ----------------------------------------------------------------------------
set action = "submit"
if ( $#argv >= 1 ) then
    set action = "$argv[1]"
endif

# ----------------------------------------------------------------------------
# Resolve the project directory correctly.
#
# IMPORTANT: inside a Slurm job, $0 points to a temporary copy under
# /var/spool/slurmd/job..., not to the original project script. Therefore the
# login-node submit action captures the real project directory and script path
# in environment variables. Every dependent job inherits them via --export=ALL.
# ----------------------------------------------------------------------------
if ( "$action" == "submit" ) then
    set work_dir = "`pwd`"

    if ( -x /usr/bin/readlink ) then
        set resolved_work_dir = "`/usr/bin/readlink -f "$work_dir"`"
        if ( "$resolved_work_dir" != "" ) then
            set work_dir = "$resolved_work_dir"
        endif
    endif

    set script_path = "$0"
    if ( "$script_path" !~ /* ) then
        set script_path = "$work_dir/$script_path"
    endif

    if ( -x /usr/bin/readlink ) then
        set resolved_script_path = "`/usr/bin/readlink -f "$script_path"`"
        if ( "$resolved_script_path" != "" ) then
            set script_path = "$resolved_script_path"
        endif
    endif

    setenv PIPELINE_WORK_DIR "$work_dir"
    setenv PIPELINE_SCRIPT_PATH "$script_path"
else
    if ( ! $?PIPELINE_WORK_DIR ) then
        echo "ERROR: PIPELINE_WORK_DIR was not inherited by this Slurm job."
        echo "Start the workflow from the project directory with:"
        echo "  tcsh run_lora_pipeline_complete_fixed.csh submit"
        exit 2
    endif

    if ( ! $?PIPELINE_SCRIPT_PATH ) then
        echo "ERROR: PIPELINE_SCRIPT_PATH was not inherited by this Slurm job."
        exit 2
    endif

    set work_dir = "$PIPELINE_WORK_DIR"
    set script_path = "$PIPELINE_SCRIPT_PATH"
endif

cd "$work_dir"
if ( $status != 0 ) then
    echo "ERROR: Could not change to project directory: $work_dir"
    exit 2
endif

# ----------------------------------------------------------------------------
# User-adjustable pipeline controls.
# ----------------------------------------------------------------------------
if ( ! $?PIPELINE_DIR ) then
    setenv PIPELINE_DIR "lora_pipeline_run"
endif

if ( ! $?N_RANDOM_TRIALS ) then
    setenv N_RANDOM_TRIALS 12
endif

if ( ! $?TOP_K_FOR_CV ) then
    setenv TOP_K_FOR_CV 2
endif

if ( ! $?PIPELINE_SEED ) then
    setenv PIPELINE_SEED 20260728
endif

# Script names may be overridden before submission.
if ( ! $?PIPELINE_MANAGER_SCRIPT ) then
    setenv PIPELINE_MANAGER_SCRIPT "automated_lora_pipeline.py"
endif

if ( ! $?TRAINER_SCRIPT ) then
    setenv TRAINER_SCRIPT "lora_local_news_tuning.py"
endif

set manager = "$work_dir/$PIPELINE_MANAGER_SCRIPT"
set trainer = "$work_dir/$TRAINER_SCRIPT"

# Backward-compatible fallback for the previously generated manager filename.
if ( ! -e "$manager" && -e "$work_dir/automated_lora_pipeline.py" ) then
    set manager = "$work_dir/automated_lora_pipeline.py"
    setenv PIPELINE_MANAGER_SCRIPT "automated_lora_pipeline.py"
endif

# ----------------------------------------------------------------------------
# Validate required files.
# ----------------------------------------------------------------------------
if ( ! -e "$manager" ) then
    echo "ERROR: Pipeline manager was not found:"
    echo "  $manager"
    echo "Expected automated_lora_pipeline.py, or set PIPELINE_MANAGER_SCRIPT explicitly."
    exit 2
endif

if ( ! -e "$trainer" ) then
    echo "ERROR: Training script was not found:"
    echo "  $trainer"
    exit 2
endif

if ( ! -e "$work_dir/data/labels_with_title.csv" ) then
    echo "ERROR: Missing data/labels_with_title.csv"
    exit 2
endif

if ( ! -e "$work_dir/data/gangani_local_news_labeling_round_2_v2.csv" ) then
    echo "ERROR: Missing data/gangani_local_news_labeling_round_2_v2.csv"
    exit 2
endif

mkdir -p "$PIPELINE_DIR" logs outputs lora_local_news_experiments

# ============================================================================
# SUBMISSION STAGE
# Runs on the login node and submits the full dependency chain.
# ============================================================================
if ( "$action" == "submit" ) then

    if ( ! $?HF_TOKEN ) then
        echo "ERROR: HF_TOKEN is not set."
        echo "Run first:"
        echo "  setenv HF_TOKEN your_huggingface_token"
        exit 3
    endif

    # Validate numerical controls before constructing array ranges.
    if ( $N_RANDOM_TRIALS < 1 ) then
        echo "ERROR: N_RANDOM_TRIALS must be at least 1."
        exit 3
    endif

    if ( $TOP_K_FOR_CV < 1 ) then
        echo "ERROR: TOP_K_FOR_CV must be at least 1."
        exit 3
    endif

    if ( $TOP_K_FOR_CV > $N_RANDOM_TRIALS ) then
        echo "ERROR: TOP_K_FOR_CV cannot exceed N_RANDOM_TRIALS."
        exit 3
    endif

    echo "Initializing automated LoRA pipeline..."
    echo "Working directory: $work_dir"
    echo "Manager:           $manager"
    echo "Trainer:           $trainer"
    echo "Pipeline directory:$PIPELINE_DIR"
    echo "Random trials:     $N_RANDOM_TRIALS"
    echo "Top configurations:$TOP_K_FOR_CV"
    echo "Pipeline seed:     $PIPELINE_SEED"

    python "$manager" init \
        --pipeline-dir "$PIPELINE_DIR" \
        --n-trials "$N_RANDOM_TRIALS" \
        --top-k "$TOP_K_FOR_CV" \
        --seed "$PIPELINE_SEED"

    set init_status = $status
    if ( $init_status != 0 ) then
        echo "ERROR: Pipeline initialization failed with status $init_status."
        exit $init_status
    endif

    @ search_last = $N_RANDOM_TRIALS - 1
    @ cv_last = $TOP_K_FOR_CV - 1

    # ------------------------------------------------------------------------
    # 1. Random-search screening array.
    # ------------------------------------------------------------------------
    set search_job = `sbatch --parsable --export=ALL \
        --job-name=ln-rsearch \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --gres=gpu:1 \
        --mem=128G \
        --time=72:00:00 \
        --array=0-$search_last \
        --output=logs/random_search_%A_%a.out \
        --error=logs/random_search_%A_%a.err \
        "$script_path" search`

    set submit_status = $status
    if ( $submit_status != 0 || "$search_job" == "" ) then
        echo "ERROR: Failed to submit random-search array."
        exit 4
    endif

    # ------------------------------------------------------------------------
    # 2. Rank completed screening runs.
    # afterany allows the ranker to ignore occasional failed trials and use
    # successful screen_summary.json files. It must fail if none succeeded.
    # ------------------------------------------------------------------------
    set rank_search_job = `sbatch --parsable --export=ALL \
        --job-name=ln-rank-search \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=2 \
        --mem=8G \
        --time=00:20:00 \
        --dependency=afterany:$search_job \
        --output=logs/rank_search_%j.out \
        --error=logs/rank_search_%j.err \
        "$script_path" rank_search`

    set submit_status = $status
    if ( $submit_status != 0 || "$rank_search_job" == "" ) then
        echo "ERROR: Failed to submit screening-ranking job."
        exit 4
    endif

    # ------------------------------------------------------------------------
    # 3. Cross-validation array for automatically selected finalists.
    # ------------------------------------------------------------------------
    set cv_job = `sbatch --parsable --export=ALL \
        --job-name=ln-cv \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --gres=gpu:1 \
        --mem=128G \
        --time=120:00:00 \
        --array=0-$cv_last \
        --dependency=afterok:$rank_search_job \
        --output=logs/cv_%A_%a.out \
        --error=logs/cv_%A_%a.err \
        "$script_path" cv`

    set submit_status = $status
    if ( $submit_status != 0 || "$cv_job" == "" ) then
        echo "ERROR: Failed to submit CV array."
        exit 4
    endif

    # ------------------------------------------------------------------------
    # 4. Rank completed CV finalists.
    # ------------------------------------------------------------------------
    set rank_cv_job = `sbatch --parsable --export=ALL \
        --job-name=ln-rank-cv \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=2 \
        --mem=8G \
        --time=00:20:00 \
        --dependency=afterany:$cv_job \
        --output=logs/rank_cv_%j.out \
        --error=logs/rank_cv_%j.err \
        "$script_path" rank_cv`

    set submit_status = $status
    if ( $submit_status != 0 || "$rank_cv_job" == "" ) then
        echo "ERROR: Failed to submit CV-ranking job."
        exit 4
    endif

    # ------------------------------------------------------------------------
    # 5. Train the winning configuration on the full development set.
    # ------------------------------------------------------------------------
    set final_job = `sbatch --parsable --export=ALL \
        --job-name=ln-final-train \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --gres=gpu:1 \
        --mem=128G \
        --time=120:00:00 \
        --dependency=afterok:$rank_cv_job \
        --output=logs/final_train_%j.out \
        --error=logs/final_train_%j.err \
        "$script_path" final`

    set submit_status = $status
    if ( $submit_status != 0 || "$final_job" == "" ) then
        echo "ERROR: Failed to submit final-training job."
        exit 4
    endif

    # ------------------------------------------------------------------------
    # 6. Evaluate exactly once on the untouched final test set.
    # ------------------------------------------------------------------------
    set test_job = `sbatch --parsable --export=ALL \
        --job-name=ln-final-test \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --gres=gpu:1 \
        --mem=128G \
        --time=24:00:00 \
        --dependency=afterok:$final_job \
        --output=logs/final_test_%j.out \
        --error=logs/final_test_%j.err \
        "$script_path" test`

    set submit_status = $status
    if ( $submit_status != 0 || "$test_job" == "" ) then
        echo "ERROR: Failed to submit final-test job."
        exit 4
    endif

    # ------------------------------------------------------------------------
    # 7. Produce one consolidated pipeline report.
    # afterany ensures a diagnostic report can still be attempted if testing
    # fails; the manager should mark missing outputs clearly.
    # ------------------------------------------------------------------------
    set report_job = `sbatch --parsable --export=ALL \
        --job-name=ln-pipeline-report \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=2 \
        --mem=8G \
        --time=00:20:00 \
        --dependency=afterany:$test_job \
        --output=logs/pipeline_report_%j.out \
        --error=logs/pipeline_report_%j.err \
        "$script_path" report`

    set submit_status = $status
    if ( $submit_status != 0 || "$report_job" == "" ) then
        echo "ERROR: Failed to submit final-report job."
        exit 4
    endif

    echo ""
    echo "Automated pipeline submitted successfully."
    echo "  Search array:        $search_job"
    echo "  Rank search:         $rank_search_job"
    echo "  CV array:            $cv_job"
    echo "  Rank CV:             $rank_cv_job"
    echo "  Final training:      $final_job"
    echo "  Final testing:       $test_job"
    echo "  Consolidated report: $report_job"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u $USER"
    echo ""
    echo "Final report path:"
    echo "  $work_dir/$PIPELINE_DIR/AUTOMATED_PIPELINE_FINAL_SUMMARY.json"
    exit 0
endif

# ============================================================================
# COMMON ENVIRONMENT SETUP
# Used by GPU workers and lightweight CPU ranking/report jobs.
# ============================================================================
if ( "$action" == "search" || "$action" == "cv" || "$action" == "final" || "$action" == "test" || "$action" == "rank_search" || "$action" == "rank_cv" || "$action" == "report" ) then

    echo "Job started: `date`"
    echo "Host: `hostname`"
    echo "Directory: `pwd`"
    echo "Action: $action"
    if ( $?SLURM_JOB_ID ) then
        echo "Slurm job ID: $SLURM_JOB_ID"
    endif
    if ( $?SLURM_ARRAY_TASK_ID ) then
        echo "Slurm array task: $SLURM_ARRAY_TASK_ID"
    endif

    # Activate the same environment used by the earlier working C shell script.
    source /sciclone/data10/gchewababarand/Nationalization/llnr/bin/activate.csh
    if ( $status != 0 ) then
        echo "ERROR: Failed to activate the llnr environment."
        exit 5
    endif

    if ( ! $?HF_TOKEN ) then
        echo "ERROR: HF_TOKEN was not inherited by this Slurm job."
        echo "Submit the pipeline after running:"
        echo "  setenv HF_TOKEN your_huggingface_token"
        exit 5
    endif

    # Hugging Face cache configuration, matching the earlier working script.
    setenv HF_HOME "/sciclone/scr10/gchewababarand/tmp/hf_cache"
    setenv TRANSFORMERS_CACHE "$HF_HOME"
    setenv HF_HUB_CACHE "$HF_HOME/hub"
    mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

    # Runtime settings.
    setenv PYTORCH_CUDA_ALLOC_CONF "expandable_segments:True"
    setenv TOKENIZERS_PARALLELISM false

    if ( $?SLURM_CPUS_PER_TASK ) then
        setenv OMP_NUM_THREADS "$SLURM_CPUS_PER_TASK"
    else
        setenv OMP_NUM_THREADS 2
    endif
endif

# ============================================================================
# GPU WORKER STAGES
# ============================================================================
if ( "$action" == "search" || "$action" == "cv" || "$action" == "final" || "$action" == "test" ) then

    # Shared dataset/model settings.
    setenv CSV_PATH "$work_dir/data/labels_with_title.csv"
    setenv ROUND2_CSV_PATH "$work_dir/data/gangani_local_news_labeling_round_2_v2.csv"
    setenv OUTPUT_ROOT "$work_dir/lora_local_news_experiments"
    setenv BASE_MODEL "meta-llama/Llama-3.1-8B-Instruct"
    setenv SEED 42
    setenv FINAL_TEST_SIZE 0.20
    setenv VALIDATION_SIZE_WITHIN_DEVELOPMENT 0.10
    setenv N_SPLITS 5

    # Shared training defaults.
    setenv NUM_EPOCHS 6
    setenv EARLY_STOPPING_PATIENCE 2
    setenv EARLY_STOPPING_THRESHOLD 0.001
    setenv BATCH_SIZE_TRAIN 1
    setenv BATCH_SIZE_EVAL 2
    setenv GRAD_ACCUM_STEPS 16
    setenv MAX_SEQ_LENGTH 2048
    setenv WARMUP_RATIO 0.05
    setenv WEIGHT_DECAY 0.0
    setenv LOGGING_STEPS 10
    setenv TRUNCATION_STRATEGY head_tail
    setenv OVERSAMPLE_TRAINING false
    setenv OVERSAMPLE_CAP_MULTIPLIER 1.0

    # MAX_ROWS is intentionally not overwritten. If the user exported it before
    # submission, Slurm inherits it and the Python script performs a smoke test.

    # ------------------------------------------------------------------------
    # Emit the selected hyperparameters for the current phase.
    # The manager prints valid tcsh commands such as setenv NAME value.
    # ------------------------------------------------------------------------
    if ( "$action" == "search" ) then
        if ( ! $?SLURM_ARRAY_TASK_ID ) then
            echo "ERROR: search action requires SLURM_ARRAY_TASK_ID."
            exit 6
        endif

        set emitted = "`python "$manager" emit --pipeline-dir "$PIPELINE_DIR" --phase search --index "$SLURM_ARRAY_TASK_ID"`"
        set emit_status = $status
        if ( $emit_status != 0 || "$emitted" == "" ) then
            echo "ERROR: Failed to load random-search configuration $SLURM_ARRAY_TASK_ID."
            exit 6
        endif
        eval "$emitted"
        setenv MODE screen

    else if ( "$action" == "cv" ) then
        if ( ! $?SLURM_ARRAY_TASK_ID ) then
            echo "ERROR: cv action requires SLURM_ARRAY_TASK_ID."
            exit 6
        endif

        set emitted = "`python "$manager" emit --pipeline-dir "$PIPELINE_DIR" --phase cv --index "$SLURM_ARRAY_TASK_ID"`"
        set emit_status = $status
        if ( $emit_status != 0 || "$emitted" == "" ) then
            echo "ERROR: Failed to load CV finalist $SLURM_ARRAY_TASK_ID."
            exit 6
        endif
        eval "$emitted"
        setenv MODE cv

    else if ( "$action" == "final" ) then
        set emitted = "`python "$manager" emit --pipeline-dir "$PIPELINE_DIR" --phase final --index 0`"
        set emit_status = $status
        if ( $emit_status != 0 || "$emitted" == "" ) then
            echo "ERROR: Failed to load the winning CV configuration."
            exit 6
        endif
        eval "$emitted"
        setenv MODE final

    else if ( "$action" == "test" ) then
        set emitted = "`python "$manager" emit --pipeline-dir "$PIPELINE_DIR" --phase test --index 0`"
        set emit_status = $status
        if ( $emit_status != 0 || "$emitted" == "" ) then
            echo "ERROR: Failed to load the winning test configuration."
            exit 6
        endif
        eval "$emitted"

        set adapter_assignment = "`python "$manager" resolve-adapter --pipeline-dir "$PIPELINE_DIR" --output-root "$OUTPUT_ROOT" --base-model "$BASE_MODEL"`"
        set adapter_status = $status
        if ( $adapter_status != 0 || "$adapter_assignment" == "" ) then
            echo "ERROR: Could not resolve the final adapter directory."
            exit 6
        endif
        eval "$adapter_assignment"
        setenv MODE test
    endif

    # Validate manager-emitted values before starting expensive GPU work.
    if ( ! $?EXPERIMENT_ID ) then
        echo "ERROR: EXPERIMENT_ID was not emitted by the pipeline manager."
        exit 7
    endif
    if ( ! $?LEARNING_RATE || ! $?LORA_R || ! $?LORA_ALPHA || ! $?LORA_DROPOUT ) then
        echo "ERROR: Incomplete hyperparameter configuration from pipeline manager."
        exit 7
    endif
    if ( ! $?TARGET_MODULES || ! $?PROMPT_VARIANT || ! $?MAX_INPUT_CHARS ) then
        echo "ERROR: Incomplete prompt/input configuration from pipeline manager."
        exit 7
    endif
    if ( "$action" == "test" && ! $?FINAL_ADAPTER_DIR ) then
        echo "ERROR: FINAL_ADAPTER_DIR was not resolved for test mode."
        exit 7
    endif

    echo ""
    echo "Python environment:"
    python --version
    python -c "import torch, transformers, peft; print('torch', torch.__version__); print('transformers', transformers.__version__); print('peft', peft.__version__); print('CUDA', torch.cuda.is_available()); print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
    if ( $status != 0 ) then
        echo "ERROR: Python environment validation failed."
        exit 7
    endif

    nvidia-smi

    echo ""
    echo "Mode:              $MODE"
    echo "Experiment:        $EXPERIMENT_ID"
    echo "Learning rate:     $LEARNING_RATE"
    echo "LoRA rank:         $LORA_R"
    echo "LoRA alpha:        $LORA_ALPHA"
    echo "LoRA dropout:      $LORA_DROPOUT"
    echo "Target modules:    $TARGET_MODULES"
    echo "Prompt variant:    $PROMPT_VARIANT"
    echo "Max input chars:   $MAX_INPUT_CHARS"
    echo "Max sequence len:  $MAX_SEQ_LENGTH"
    if ( $?MAX_ROWS ) then
        echo "Smoke-test rows:   $MAX_ROWS"
    endif
    if ( "$action" == "test" ) then
        echo "Final adapter:     $FINAL_ADAPTER_DIR"
    endif

    set detailed_log = "$work_dir/outputs/${EXPERIMENT_ID}_${MODE}_${SLURM_JOB_ID}.txt"

    python "$trainer" >&! "$detailed_log"
    set python_status = $status

    echo "Python exit status: $python_status"
    echo "Detailed log:       $detailed_log"
    echo "Job finished:       `date`"
    exit $python_status
endif

# ============================================================================
# LIGHTWEIGHT CPU STAGES
# ============================================================================
if ( "$action" == "rank_search" ) then
    python "$manager" rank-search \
        --pipeline-dir "$PIPELINE_DIR" \
        --output-root "$work_dir/lora_local_news_experiments" \
        --base-model "meta-llama/Llama-3.1-8B-Instruct" \
        --top-k "$TOP_K_FOR_CV"

    set command_status = $status
    echo "Screening ranking finished with status $command_status at `date`."
    exit $command_status
endif

if ( "$action" == "rank_cv" ) then
    python "$manager" rank-cv \
        --pipeline-dir "$PIPELINE_DIR" \
        --output-root "$work_dir/lora_local_news_experiments" \
        --base-model "meta-llama/Llama-3.1-8B-Instruct"

    set command_status = $status
    echo "CV ranking finished with status $command_status at `date`."
    exit $command_status
endif

if ( "$action" == "report" ) then
    python "$manager" finalize \
        --pipeline-dir "$PIPELINE_DIR" \
        --output-root "$work_dir/lora_local_news_experiments" \
        --base-model "meta-llama/Llama-3.1-8B-Instruct"

    set command_status = $status
    echo "Pipeline report finished with status $command_status at `date`."
    exit $command_status
endif

# ----------------------------------------------------------------------------
# Unknown action.
# ----------------------------------------------------------------------------
echo "ERROR: Unknown action '$action'."
echo "Valid actions are:"
echo "  submit"
echo "  search"
echo "  rank_search"
echo "  cv"
echo "  rank_cv"
echo "  final"
echo "  test"
echo "  report"
exit 9
