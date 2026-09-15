#!/usr/bin/env python3
"""Manage an automated random-search -> CV -> final-train -> final-test LoRA pipeline.

This script is a lightweight controller around ``lora_local_news_tuning.py``.
It does not train models itself.  The companion tcsh script launches Slurm jobs,
while this controller creates reproducible configurations, emits environment
variables for workers, ranks completed runs, and records the final result.

The fixed final test set remains untouched during random search and CV because
all training/evaluation is still performed by the existing training script's
``screen``, ``cv``, ``final``, and ``test`` modes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT_ROOT = "lora_local_news_experiments"


@dataclass(frozen=True)
class TrialConfig:
    trial_index: int
    experiment_id: str
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str
    prompt_variant: str
    max_input_chars: int

    def env(self) -> Dict[str, str]:
        return {
            "EXPERIMENT_ID": self.experiment_id,
            "LEARNING_RATE": format(self.learning_rate, ".8g"),
            "LORA_R": str(self.lora_r),
            "LORA_ALPHA": str(self.lora_alpha),
            "LORA_DROPOUT": format(self.lora_dropout, ".4g"),
            "TARGET_MODULES": self.target_modules,
            "PROMPT_VARIANT": self.prompt_variant,
            "MAX_INPUT_CHARS": str(self.max_input_chars),
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def model_folder(base_model: str) -> str:
    return base_model.replace("/", "__")


def sanitize_float(value: float) -> str:
    text = format(value, ".0e").replace("e-0", "e-").replace("e+0", "e+")
    return text.replace(".", "p")


def make_experiment_id(index: int, lr: float, rank: int, alpha: int,
                       dropout: float, modules: str, prompt: str,
                       max_chars: int) -> str:
    d = int(round(dropout * 100))
    module_code = "attn" if modules == "attention" else "all"
    prompt_code = {"compact": "cmp", "targeted": "tgt", "full": "full", "examples": "ex"}[prompt]
    return (
        f"rs{index:03d}_lr{sanitize_float(lr)}_r{rank}_a{alpha}_"
        f"d{d:02d}_{module_code}_{prompt_code}_c{max_chars}"
    )


def generate_unique_trials(n_trials: int, seed: int) -> List[TrialConfig]:
    # MAX_INPUT_CHARS deliberately stops at 2500.  A prior 4000-character run
    # exceeded MAX_SEQ_LENGTH=2048 and truncated the gold answer.
    learning_rates = [2e-5, 5e-5, 1e-4, 2e-4]
    ranks = [8, 16, 32, 64]
    alpha_multipliers = [1, 2]
    dropouts = [0.0, 0.05, 0.10]
    target_modules = ["attention", "all"]
    prompt_variants = ["compact", "targeted", "full", "examples"]
    max_chars_options = [1500, 2000, 2500]

    candidates = []
    for lr in learning_rates:
        for rank in ranks:
            for multiplier in alpha_multipliers:
                alpha = rank * multiplier
                for dropout in dropouts:
                    for modules in target_modules:
                        for prompt in prompt_variants:
                            for max_chars in max_chars_options:
                                candidates.append(
                                    (lr, rank, alpha, dropout, modules, prompt, max_chars)
                                )

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials > len(candidates):
        raise ValueError(f"n_trials={n_trials} exceeds {len(candidates)} unique candidates")

    rng = random.Random(seed)
    selected = rng.sample(candidates, n_trials)
    trials: List[TrialConfig] = []
    for index, values in enumerate(selected):
        lr, rank, alpha, dropout, modules, prompt, max_chars = values
        trials.append(
            TrialConfig(
                trial_index=index,
                experiment_id=make_experiment_id(
                    index, lr, rank, alpha, dropout, modules, prompt, max_chars
                ),
                learning_rate=lr,
                lora_r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=modules,
                prompt_variant=prompt,
                max_input_chars=max_chars,
            )
        )
    return trials


def load_trials(pipeline_dir: Path) -> List[TrialConfig]:
    payload = read_json(pipeline_dir / "search_configs.json")
    return [TrialConfig(**item) for item in payload["trials"]]


def load_selected_search(pipeline_dir: Path) -> List[TrialConfig]:
    payload = read_json(pipeline_dir / "selected_for_cv.json")
    return [TrialConfig(**item["config"]) for item in payload["selected"]]


def tcsh_quote(value: str) -> str:
    # Values used here are controlled and do not contain newlines.  Double-quote
    # and escape the few characters that tcsh can interpret.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
    return f'"{escaped}"'


def emit_env(config: TrialConfig) -> None:
    for key, value in config.env().items():
        print(f"setenv {key} {tcsh_quote(value)};")


def metric_float(payload: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def finite_or(value: float, fallback: float) -> float:
    return value if math.isfinite(value) else fallback


def search_summary_path(output_root: Path, base_model: str, experiment_id: str) -> Path:
    return output_root / model_folder(base_model) / experiment_id / "screen" / "screen_summary.json"


def cv_summary_path(output_root: Path, base_model: str, experiment_id: str) -> Path:
    return output_root / model_folder(base_model) / experiment_id / "cv" / "cv_summary.json"


def final_summary_path(output_root: Path, base_model: str, experiment_id: str) -> Path:
    return output_root / model_folder(base_model) / experiment_id / "final" / "final_training_summary.json"


def test_summary_path(output_root: Path, base_model: str, experiment_id: str) -> Path:
    return output_root / model_folder(base_model) / experiment_id / "test" / "final_test_summary.json"


def cmd_init(args: argparse.Namespace) -> None:
    pipeline_dir = Path(args.pipeline_dir)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    config_path = pipeline_dir / "search_configs.json"
    if config_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{config_path} already exists. Use --overwrite to create a new search."
        )

    trials = generate_unique_trials(args.n_trials, args.seed)
    payload = {
        "created_utc": utc_now(),
        "search_method": "reproducible_random_search_without_replacement",
        "seed": args.seed,
        "n_trials": len(trials),
        "top_k_for_cv": args.top_k,
        "selection_primary_metric": "validation_macro_f1",
        "selection_secondary_metric": "validation_accuracy",
        "notes": [
            "The final test set is not used during search or CV.",
            "MAX_INPUT_CHARS is restricted to at most 2500 with MAX_SEQ_LENGTH=2048.",
        ],
        "trials": [asdict(trial) for trial in trials],
    }
    write_json(config_path, payload)

    with (pipeline_dir / "search_configs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(trials[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(trial) for trial in trials)

    print(config_path)


def cmd_emit(args: argparse.Namespace) -> None:
    pipeline_dir = Path(args.pipeline_dir)
    if args.phase == "search":
        configs = load_trials(pipeline_dir)
    elif args.phase == "cv":
        configs = load_selected_search(pipeline_dir)
    elif args.phase in {"final", "test"}:
        winner = read_json(pipeline_dir / "cv_winner.json")
        configs = [TrialConfig(**winner["config"])]
    else:
        raise ValueError(f"Unsupported phase: {args.phase}")

    if args.index < 0 or args.index >= len(configs):
        raise IndexError(f"Index {args.index} is outside 0..{len(configs)-1}")
    emit_env(configs[args.index])


def cmd_rank_search(args: argparse.Namespace) -> None:
    pipeline_dir = Path(args.pipeline_dir)
    trials = load_trials(pipeline_dir)
    output_root = Path(args.output_root)
    rows: List[Dict[str, Any]] = []

    for config in trials:
        path = search_summary_path(output_root, args.base_model, config.experiment_id)
        row: Dict[str, Any] = {"config": asdict(config), "summary_path": str(path)}
        if not path.exists():
            row.update({"status": "missing_or_failed"})
        else:
            metrics = read_json(path)
            row.update(
                {
                    "status": "complete",
                    "macro_f1": metric_float(metrics, "macro_f1"),
                    "accuracy": metric_float(metrics, "accuracy"),
                    "weighted_f1": metric_float(metrics, "weighted_f1"),
                    "n_unparseable": int(metrics.get("n_unparseable", 0)),
                    "selected_epoch": metric_float(metrics, "selected_epoch"),
                }
            )
        rows.append(row)

    completed = [row for row in rows if row["status"] == "complete"]
    completed.sort(
        key=lambda row: (
            finite_or(row["macro_f1"], -1.0),
            finite_or(row["accuracy"], -1.0),
            -row["n_unparseable"],
        ),
        reverse=True,
    )
    if not completed:
        write_json(pipeline_dir / "search_ranking.json", {"created_utc": utc_now(), "runs": rows})
        raise RuntimeError("No completed screening runs were found; CV cannot start.")

    top_k = min(args.top_k, len(completed))
    selected = completed[:top_k]
    for rank, row in enumerate(completed, 1):
        row["rank"] = rank

    write_json(
        pipeline_dir / "search_ranking.json",
        {
            "created_utc": utc_now(),
            "completed_runs": len(completed),
            "failed_or_missing_runs": len(rows) - len(completed),
            "ranking_rule": "macro_f1 desc, accuracy desc, n_unparseable asc",
            "runs": completed + [r for r in rows if r["status"] != "complete"],
        },
    )
    write_json(
        pipeline_dir / "selected_for_cv.json",
        {
            "created_utc": utc_now(),
            "top_k": top_k,
            "selected": selected,
        },
    )

    with (pipeline_dir / "search_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "rank", "experiment_id", "macro_f1", "accuracy", "weighted_f1",
            "n_unparseable", "selected_epoch", "learning_rate", "lora_r",
            "lora_alpha", "lora_dropout", "target_modules", "prompt_variant",
            "max_input_chars", "summary_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in completed:
            config = row["config"]
            writer.writerow(
                {
                    "rank": row["rank"],
                    "experiment_id": config["experiment_id"],
                    "macro_f1": row["macro_f1"],
                    "accuracy": row["accuracy"],
                    "weighted_f1": row["weighted_f1"],
                    "n_unparseable": row["n_unparseable"],
                    "selected_epoch": row["selected_epoch"],
                    "learning_rate": config["learning_rate"],
                    "lora_r": config["lora_r"],
                    "lora_alpha": config["lora_alpha"],
                    "lora_dropout": config["lora_dropout"],
                    "target_modules": config["target_modules"],
                    "prompt_variant": config["prompt_variant"],
                    "max_input_chars": config["max_input_chars"],
                    "summary_path": row["summary_path"],
                }
            )
    print(json.dumps({"selected_experiments": [r["config"]["experiment_id"] for r in selected]}, indent=2))


def cmd_rank_cv(args: argparse.Namespace) -> None:
    pipeline_dir = Path(args.pipeline_dir)
    configs = load_selected_search(pipeline_dir)
    output_root = Path(args.output_root)
    rows: List[Dict[str, Any]] = []

    for config in configs:
        path = cv_summary_path(output_root, args.base_model, config.experiment_id)
        row: Dict[str, Any] = {"config": asdict(config), "summary_path": str(path)}
        if not path.exists():
            row["status"] = "missing_or_failed"
        else:
            metrics = read_json(path)
            row.update(
                {
                    "status": "complete",
                    "pooled_macro_f1": metric_float(metrics, "pooled_macro_f1"),
                    "pooled_accuracy": metric_float(metrics, "pooled_accuracy"),
                    "mean_fold_macro_f1": metric_float(metrics, "mean_fold_macro_f1"),
                    "std_fold_macro_f1": metric_float(metrics, "std_fold_macro_f1"),
                    "mean_selected_epoch": metric_float(metrics, "mean_selected_epoch"),
                }
            )
        rows.append(row)

    completed = [row for row in rows if row["status"] == "complete"]
    completed.sort(
        key=lambda row: (
            finite_or(row["pooled_macro_f1"], -1.0),
            finite_or(row["mean_fold_macro_f1"], -1.0),
            -finite_or(row["std_fold_macro_f1"], 999.0),
            finite_or(row["pooled_accuracy"], -1.0),
        ),
        reverse=True,
    )
    if not completed:
        write_json(pipeline_dir / "cv_ranking.json", {"created_utc": utc_now(), "runs": rows})
        raise RuntimeError("No completed CV runs were found; final training cannot start.")

    for rank, row in enumerate(completed, 1):
        row["rank"] = rank
    winner = completed[0]

    write_json(
        pipeline_dir / "cv_ranking.json",
        {
            "created_utc": utc_now(),
            "ranking_rule": (
                "pooled_macro_f1 desc, mean_fold_macro_f1 desc, "
                "std_fold_macro_f1 asc, pooled_accuracy desc"
            ),
            "runs": completed + [r for r in rows if r["status"] != "complete"],
        },
    )
    write_json(
        pipeline_dir / "cv_winner.json",
        {
            "created_utc": utc_now(),
            **winner,
        },
    )
    print(json.dumps(winner, indent=2))


def cmd_resolve_adapter(args: argparse.Namespace) -> None:
    pipeline_dir = Path(args.pipeline_dir)
    winner = read_json(pipeline_dir / "cv_winner.json")
    config = TrialConfig(**winner["config"])
    path = final_summary_path(Path(args.output_root), args.base_model, config.experiment_id)
    if not path.exists():
        raise FileNotFoundError(f"Final-training summary is missing: {path}")
    summary = read_json(path)
    adapter = Path(summary["final_adapter_dir"])
    if not adapter.exists():
        raise FileNotFoundError(f"Final adapter directory is missing: {adapter}")
    print(f"setenv FINAL_ADAPTER_DIR {tcsh_quote(str(adapter.resolve()))};")


def cmd_finalize(args: argparse.Namespace) -> None:
    pipeline_dir = Path(args.pipeline_dir)
    winner = read_json(pipeline_dir / "cv_winner.json")
    config = TrialConfig(**winner["config"])
    output_root = Path(args.output_root)
    final_path = final_summary_path(output_root, args.base_model, config.experiment_id)
    test_path = test_summary_path(output_root, args.base_model, config.experiment_id)

    final_summary = read_json(final_path) if final_path.exists() else None
    test_summary = read_json(test_path) if test_path.exists() else None
    payload = {
        "created_utc": utc_now(),
        "winning_configuration": asdict(config),
        "cv_selection": winner,
        "final_training_summary_path": str(final_path),
        "final_training_summary": final_summary,
        "final_test_summary_path": str(test_path),
        "final_test_summary": test_summary,
        "test_set_policy": (
            "The fixed final test set was used only after search, CV selection, and final training. "
            "Do not use these test results for additional tuning."
        ),
    }
    write_json(pipeline_dir / "AUTOMATED_PIPELINE_FINAL_SUMMARY.json", payload)
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create reproducible random-search configurations")
    init.add_argument("--pipeline-dir", required=True)
    init.add_argument("--n-trials", type=int, default=12)
    init.add_argument("--top-k", type=int, default=2)
    init.add_argument("--seed", type=int, default=20260728)
    init.add_argument("--overwrite", action="store_true")
    init.set_defaults(func=cmd_init)

    emit = sub.add_parser("emit", help="Emit tcsh setenv commands for one worker")
    emit.add_argument("--pipeline-dir", required=True)
    emit.add_argument("--phase", choices=["search", "cv", "final", "test"], required=True)
    emit.add_argument("--index", type=int, default=0)
    emit.set_defaults(func=cmd_emit)

    rank_search = sub.add_parser("rank-search", help="Rank screening runs and select CV finalists")
    rank_search.add_argument("--pipeline-dir", required=True)
    rank_search.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    rank_search.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    rank_search.add_argument("--top-k", type=int, default=2)
    rank_search.set_defaults(func=cmd_rank_search)

    rank_cv = sub.add_parser("rank-cv", help="Rank CV finalists and select the winner")
    rank_cv.add_argument("--pipeline-dir", required=True)
    rank_cv.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    rank_cv.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    rank_cv.set_defaults(func=cmd_rank_cv)

    adapter = sub.add_parser("resolve-adapter", help="Emit FINAL_ADAPTER_DIR after final training")
    adapter.add_argument("--pipeline-dir", required=True)
    adapter.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    adapter.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    adapter.set_defaults(func=cmd_resolve_adapter)

    finalize = sub.add_parser("finalize", help="Create one consolidated final summary")
    finalize.add_argument("--pipeline-dir", required=True)
    finalize.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    finalize.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    finalize.set_defaults(func=cmd_finalize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
