#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_INPUT_CHARS = 2500
DEFAULT_MAX_SEQ_LENGTH = 2048

TAXONOMY: Dict[str, str] = {
    "Politics & Government": (
        "News and discussion about governance, elections, public policy, political "
        "institutions, public administration, government services, civic institutions, "
        "public programs, legislation, political actors, civil rights, diplomacy, and "
        "government decision-making at local, regional, national, and international "
        "levels. Includes public services, nonprofit and civic institutions, and "
        "government-supported community infrastructure. IMPORTANT: a story is NOT "
        "automatically Politics & Government just because it mentions a government "
        "office, agency, official, or title. Classify by what the story is actually ABOUT."
    ),
    "Crime & Public Safety": (
        "News and discussion about crime, policing, courts, justice, public safety, "
        "emergencies, disasters, conflict, violence, terrorism, accidents, and emergency "
        "response. Includes criminal activities, legal proceedings, civil unrest, threats "
        "to public order or safety, official safety warnings, recalls, arrests, "
        "investigations, scams, cybercrime, or hacking."
    ),
    "Economy & Business": (
        "News and discussion about economic activity, employment, labour, business, "
        "industry, trade, finance, entrepreneurship, markets, utilities, economic "
        "development, companies, jobs, inflation, wages, banking, and broader economic "
        "conditions."
    ),
    "Education": (
        "News and discussion related to formal and informal education, including schools, "
        "universities, curricula, teaching, learning, educational policy, teachers, "
        "students, training, educational opportunities, school board elections or "
        "meetings, and school athletics or extracurricular programs."
    ),
    "Health": (
        "News and discussion concerning physical and mental health, healthcare systems, "
        "medicine, disease, public health, hospitals, medical treatment, vaccines, "
        "healthcare policy, wellness, and health services."
    ),
    "Environment & Weather": (
        "News and discussion about the natural environment, climate, sustainability, "
        "weather, environmental hazards, conservation, pollution, natural resources, "
        "ecosystems, weather forecasts or warnings, environmental planning, and "
        "climate-related impacts."
    ),
    "Transportation": (
        "News and discussion about transportation systems, mobility, roads, traffic, "
        "public transit, transportation infrastructure, commuting, transportation policy, "
        "and travel accessibility."
    ),
    "Housing": (
        "News and discussion concerning housing, real estate, rent, home ownership, urban "
        "development, zoning, affordability, homelessness, housing markets, land use, and "
        "residential planning issues."
    ),
    "Lifestyle, Arts & Entertainment": (
        "News and discussion about arts, culture, media, entertainment, leisure, travel, "
        "food, recreation, hobbies, fashion, social and cultural events, literature, music, "
        "film, television, gaming, cultural heritage, social media culture, religion, "
        "family life, community events, everyday lifestyle activities, human-interest "
        "stories, and quality-of-life stories. Obituaries or death notices belong in Other "
        "unless the main content is specifically about a notable artistic, cultural, or "
        "entertainment career."
    ),
    "Sports": (
        "News and discussion about sports, athletes, teams, competitions, sporting events, "
        "sports organizations, coaching, venues, achievements, and sports-related controversies."
    ),
    "Science & Technology": (
        "News and discussion about science, research, engineering, innovation, technology, "
        "computing, artificial intelligence, scientific institutions, scientific "
        "discoveries, or technological development."
    ),
    "Other": (
        "Content that does not clearly belong to any predefined category, contains "
        "insufficient information for reliable classification, covers miscellaneous topics "
        "outside the taxonomy, is a paywall/broken-page notice with no real article content, "
        "or is written in a language other than English."
    ),
}

LABELS = list(TAXONOMY.keys())
LABEL_SET_LOWER = {label.lower(): label for label in LABELS}

BOUNDARY_RULES = """
Important category boundaries:
- Economy & Business: choose this when the main subject is a company, employer,
  industry, jobs, layoffs, wages, labor, business opening/closing/ownership,
  commercial activity, prices, inflation, finance, banking, investment, trade,
  utilities as businesses, agriculture as an industry, or economic development.
- Lifestyle, Arts & Entertainment: choose this for the consumer experience,
  dining/reviews, recipes, leisure, arts, entertainment, cultural events, travel,
  hobbies, or a human-interest profile. A restaurant opening, acquisition,
  ownership change, expansion, or commercial performance is Economy & Business;
  a restaurant review, menu feature, or dining guide is Lifestyle.
- Science & Technology: choose this when the main subject is research,
  engineering, a scientific discovery, or how a technology works. A technology
  company's layoffs, earnings, market, investment, conference participation, or
  business expansion is Economy & Business.
- Environment & Weather: choose this for weather, climate, conservation,
  pollution, ecosystems, environmental hazards, or natural-resource impacts.
  Farming, oil, gas, energy, or utilities discussed mainly as industries,
  employers, prices, production, or markets are Economy & Business.
- Housing: choose this when the main subject is residential housing, rent,
  homeownership, homelessness, residential affordability, residential zoning,
  or the residential real-estate market. Commercial real estate, office leasing,
  hotels, and business development are Economy & Business unless residential
  housing is central.
- Politics & Government: choose this when the main subject is governance,
  elections, legislation, public policy, or a government decision. Government
  involvement alone does not override the substantive topic. For example, a
  report mainly about jobs, prices, businesses, or economic effects remains
  Economy & Business.
- Other: use only when no substantive category fits, the usable article content
  is missing, the page is non-English, or the text is only boilerplate/paywall
  material. Do not use Other merely because the article is a short business
  notice, incorporation list, bankruptcy list, market update, or press release.
- Use the article's main subject and purpose. Do not classify from one keyword,
  section heading, website name, or weather/navigation boilerplate.
""".strip()

BOUNDARY_EXAMPLES = """
Boundary examples:
- Restaurant opening or ownership change -> Economy & Business
- Restaurant review or dining guide -> Lifestyle, Arts & Entertainment
- Technology company layoffs or earnings -> Economy & Business
- Scientific discovery or engineering research -> Science & Technology
- School board curriculum decision -> Education
- General mayoral election -> Politics & Government
- Hospital expansion or healthcare services -> Health
- Residential development or rent affordability -> Housing
- Highway closure or transit project -> Transportation
""".strip()


def log(message: str) -> None:
    print(message, flush=True)


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD."
        ) from exc


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(
            str(item).strip() for item in value
            if item is not None and str(item).strip()
        )
    if isinstance(value, dict):
        return "\n".join(
            str(item).strip() for item in value.values()
            if item is not None and str(item).strip()
        )
    return str(value).strip()


def clean_boilerplate(text: str) -> str:
    lines: List[str] = []
    drop_exact = {
        "subscribe", "sign in", "log in", "advertisement", "privacy policy",
        "cookie policy", "all rights reserved", "newsletter signup", "share this article",
    }
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
        if normalized in drop_exact:
            continue
        if len(line) <= 80 and re.match(
            r"^(subscribe|sign in|log in|advertisement|privacy policy|cookie policy|related stories)$",
            line,
            flags=re.IGNORECASE,
        ):
            continue
        lines.append(line)
    return "\n".join(lines)


def truncate_text(text: str, max_input_chars: int, strategy: str) -> str:
    text = clean_boilerplate(normalize_text(text))
    if max_input_chars <= 0 or len(text) <= max_input_chars:
        return text
    if strategy == "head":
        return text[:max_input_chars] + " [...]"
    if strategy == "head_tail":
        head = int(max_input_chars * 0.80)
        tail = max_input_chars - head
        return text[:head] + "\n[...]\n" + text[-tail:]
    raise ValueError("truncation strategy must be 'head' or 'head_tail'")


def taxonomy_block(prompt_variant: str) -> str:
    compact = prompt_variant != "full"
    if compact:
        return "\n".join(f"- {label}: {definition}" for label, definition in TAXONOMY.items())
    return "\n".join(
        f"{index}. {label}: {definition}"
        for index, (label, definition) in enumerate(TAXONOMY.items(), 1)
    )


def system_prompt(prompt_variant: str) -> str:
    if prompt_variant not in {"full", "compact", "targeted", "examples"}:
        raise ValueError(
            f"prompt variant must be full, compact, targeted, or examples; got {prompt_variant!r}"
        )
    prompt = (
        "You are a precise local-news topic classification system.\n\n"
        "Choose exactly ONE category from the fixed taxonomy below.\n\n"
        f"Categories:\n{taxonomy_block(prompt_variant)}\n"
    )
    if prompt_variant in {"full", "targeted", "examples"}:
        prompt += f"\n\n{BOUNDARY_RULES}"
    if prompt_variant == "examples":
        prompt += f"\n\n{BOUNDARY_EXAMPLES}"
    prompt += (
        "\n\nOutput rules:\n"
        "- Return only one category name copied verbatim from the taxonomy.\n"
        "- Do not provide confidence, reasoning, notes, numbering, or extra text.\n"
        "- Never invent a category or use a synonym.\n"
        "- Exact output example: Economy & Business"
    )
    return prompt


def build_user_prompt(title: str, content: str, max_input_chars: int, strategy: str) -> str:
    return (
        "Article title:\n"
        f"{normalize_text(title)}\n\n"
        "Use the title as a strong signal, but classify according to the article's main subject.\n\n"
        "Article body:\n"
        '"""\n'
        f"{truncate_text(content, max_input_chars, strategy)}\n"
        '"""\n\n'
        "Category:"
    )


def match_label(text: str) -> str:
    if not text:
        return "UNPARSEABLE"
    first_line = text.strip().splitlines()[0].strip()
    cleaned = first_line.strip().strip("\"'.,` ")
    cleaned = re.sub(r"^(category\s*:\s*|\d+\.\s*)", "", cleaned, flags=re.I).strip()
    exact = LABEL_SET_LOWER.get(cleaned.lower())
    if exact:
        return exact
    for label in sorted(LABELS, key=len, reverse=True):
        if re.match(rf"^{re.escape(label)}(?:\s*$|\s*[|:.,-])", cleaned, flags=re.I):
            return label
    return "UNPARSEABLE"


def get_hf_token() -> Optional[str]:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    return None


def quantization_config() -> BitsAndBytesConfig:
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def extract_date_from_path(file_path: Path, in_root: Path) -> Optional[date]:
    try:
        relative = file_path.relative_to(in_root)
        if len(relative.parts) < 5:
            return None
        year, month, day = relative.parts[-3], relative.parts[-2], file_path.stem
        return datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").date()
    except (ValueError, OSError):
        return None


def iter_parquet_files(
    in_root: Path,
    start_date: Optional[date],
    end_date: Optional[date],
    shard_index: int = 0,
    num_shards: int = 1,
) -> List[Path]:
    selected: List[Path] = []
    skipped = 0
    for path in in_root.rglob("*.parquet"):
        file_date = extract_date_from_path(path, in_root)
        if file_date is None:
            skipped += 1
            log(f"[skip] Unrecognized date path: {path}")
            continue
        if start_date is not None and file_date < start_date:
            continue
        if end_date is not None and file_date > end_date:
            continue
        selected.append(path)
    selected.sort()
    all_selected_count = len(selected)
    if num_shards > 1:
        selected = [
            path for position, path in enumerate(selected)
            if position % num_shards == shard_index
        ]
    if skipped:
        log(f"Skipped {skipped:,} parquet files with unrecognized date paths")
    log(
        f"Shard {shard_index + 1}/{num_shards}: selected "
        f"{len(selected):,} of {all_selected_count:,} matching parquet files"
    )
    return selected


def make_fallback_unique_id(row: pd.Series) -> str:
    basis = (
        normalize_text(row.get("article_url", "")) + "\n" +
        normalize_text(row.get("title", "")) + "\n" +
        normalize_text(row.get("content", ""))[:2000]
    )
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def standardize_dataframe(df: pd.DataFrame, file_path: Path, in_root: Path) -> pd.DataFrame:
    out = df.copy()
    for column in ("unique_id", "article_url", "title", "content"):
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].map(normalize_text)

    missing_uid = out["unique_id"].fillna("").astype(str).str.strip() == ""
    if missing_uid.any():
        out.loc[missing_uid, "unique_id"] = out.loc[missing_uid].apply(
            make_fallback_unique_id, axis=1
        )

    relative = file_path.relative_to(in_root)
    year, month, day = relative.parts[-3], relative.parts[-2], file_path.stem
    for column, value in (("year", year), ("month", month), ("day", day)):
        if column not in out.columns:
            out[column] = value
        else:
            out[column] = out[column].fillna("").astype(str).str.strip()
            out.loc[out[column] == "", column] = value
    return out


def resolve_adapter(
    adapter_dir: Optional[Path],
    pipeline_dir: Optional[Path],
    experiment_root: Path,
    base_model: str,
    training_project_dir: Optional[Path],
) -> Tuple[Path, str, Dict[str, Any]]:
    if adapter_dir is not None:
        resolved = adapter_dir.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Adapter directory does not exist: {resolved}")

        packaged_config = resolved.parent / "configuration.json"
        if packaged_config.is_file():
            metadata = json.loads(packaged_config.read_text(encoding="utf-8"))
            return resolved, metadata["experiment_id"], metadata
        experiment_id = (
            resolved.parents[2].name if len(resolved.parents) >= 3 else resolved.name
        )
        metadata: Dict[str, Any] = {}
        final_dir = resolved.parents[2] / "final"
        configuration_path = final_dir / "configuration.json"
        if configuration_path.exists():
            metadata = json.loads(configuration_path.read_text(encoding="utf-8"))
        return resolved, experiment_id, metadata

    if pipeline_dir is None:
        raise ValueError("Provide either --adapter-dir or --pipeline-dir")

    pipeline_dir = pipeline_dir.expanduser().resolve()
    experiment_root = experiment_root.expanduser().resolve()
    winner_path = pipeline_dir / "cv_winner.json"
    if not winner_path.exists():
        raise FileNotFoundError(f"Missing CV winner file: {winner_path}")

    winner = json.loads(winner_path.read_text(encoding="utf-8"))
    winner_config = dict(winner.get("config", {}))
    experiment_id = winner_config["experiment_id"]
    model_folder = base_model.replace("/", "__")
    final_dir = experiment_root / model_folder / experiment_id / "final"
    final_summary = final_dir / "final_training_summary.json"
    if not final_summary.exists():
        raise FileNotFoundError(f"Missing final-training summary: {final_summary}")

    summary = json.loads(final_summary.read_text(encoding="utf-8"))
    resolved = Path(summary["final_adapter_dir"]).expanduser()
    if not resolved.is_absolute():
        if training_project_dir is not None:
            resolved = (training_project_dir.expanduser().resolve() / resolved).resolve()
        else:
            # Fall back to the conventional location beneath the experiment's final directory.
            resolved = (final_dir / "final_training_all_development" / "adapter").resolve()

    if not resolved.is_dir():
        raise FileNotFoundError(f"Resolved adapter directory does not exist: {resolved}")

    metadata = winner_config
    configuration_path = final_dir / "configuration.json"
    if configuration_path.exists():
        metadata = {
            **metadata,
            **json.loads(configuration_path.read_text(encoding="utf-8")),
        }
    return resolved, experiment_id, metadata

def build_output_path(
    input_file: Path,
    in_root: Path,
    out_root: Path,
    experiment_id: str,
) -> Path:
    return out_root / f"model={experiment_id}" / input_file.relative_to(in_root)


def load_model(base_model: str, adapter_dir: Path):
    token = get_hf_token()
    if token is None:
        log("WARNING: no Hugging Face token found; gated base-model loading may fail")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config(),
        device_map="auto",
        token=token,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer, base


@torch.inference_mode()
def classify_dataframe(
    df: pd.DataFrame,
    model,
    tokenizer,
    batch_size: int,
    max_input_chars: int,
    max_seq_length: int,
    prompt_variant: str,
    truncation_strategy: str,
) -> pd.DataFrame:
    output = df.copy()
    predictions = ["UNPARSEABLE"] * len(output)
    raw_outputs = [""] * len(output)
    statuses = ["no_text"] * len(output)

    usable_positions = [
        index for index, (title, content) in enumerate(zip(output["title"], output["content"]))
        if normalize_text(title) or normalize_text(content)
    ]

    sys_prompt = system_prompt(prompt_variant)
    for start in range(0, len(usable_positions), batch_size):
        positions = usable_positions[start:start + batch_size]
        prompts: List[str] = []
        for position in positions:
            messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        output.iloc[position]["title"],
                        output.iloc[position]["content"],
                        max_input_chars,
                        truncation_strategy,
                    ),
                },
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(model.device)
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = generated_ids[:, encoded["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for position, raw in zip(positions, decoded):
            raw = raw.strip()
            label = match_label(raw)
            predictions[position] = label
            raw_outputs[position] = raw
            statuses[position] = "classified" if label != "UNPARSEABLE" else "unparseable"

        log(f"  classified {min(start + batch_size, len(usable_positions)):,}/{len(usable_positions):,}")

    output["predicted_label"] = predictions
    output["raw_model_output"] = raw_outputs
    output["classification_status"] = statuses
    return output


def process_file(
    file_path: Path,
    in_root: Path,
    out_root: Path,
    experiment_id: str,
    model,
    tokenizer,
    batch_size: int,
    max_input_chars: int,
    max_seq_length: int,
    prompt_variant: str,
    truncation_strategy: str,
    overwrite_existing: bool,
) -> Tuple[int, int, str]:
    output_file = build_output_path(file_path, in_root, out_root, experiment_id)
    if output_file.exists() and not overwrite_existing:
        log(f"[skip] Output exists: {output_file}")
        return 0, 0, "existing"

    try:
        dataframe = pd.read_parquet(file_path)
    except Exception as exc:
        log(f"[error] Read failed for {file_path}: {type(exc).__name__}: {exc}")
        return 0, 0, "read_error"

    dataframe = standardize_dataframe(dataframe, file_path, in_root)
    try:
        result = classify_dataframe(
            dataframe,
            model,
            tokenizer,
            batch_size,
            max_input_chars,
            max_seq_length,
            prompt_variant,
            truncation_strategy,
        )
    except Exception as exc:
        log(f"[error] Classification failed for {file_path}: {type(exc).__name__}: {exc}")
        return len(dataframe), 0, "classification_error"

    classified = int((result["classification_status"] == "classified").sum())
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_file.with_name(output_file.name + ".tmp")
        result.to_parquet(temporary, index=False)
        temporary.replace(output_file)
    except Exception as exc:
        log(f"[error] Write failed for {output_file}: {type(exc).__name__}: {exc}")
        return len(dataframe), classified, "write_error"

    log(
        f"[done] {file_path} | rows={len(dataframe):,} | "
        f"classified={classified:,} | out={output_file}"
    )
    return len(dataframe), classified, "done"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify USLNDA parquet articles with the final LoRA topic model"
    )
    parser.add_argument("--in-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--pipeline-dir", type=Path, default=None)
    parser.add_argument(
        "--training-project-dir", type=Path, default=None,
        help="Original project directory used when training; needed when saved adapter paths are relative",
    )
    parser.add_argument(
        "--experiment-root", type=Path,
        default=Path("lora_local_news_experiments"),
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--start-date", type=parse_iso_date, default=None)
    parser.add_argument("--end-date", type=parse_iso_date, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-input-chars", type=int, default=DEFAULT_MAX_INPUT_CHARS)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument(
        "--prompt-variant",
        choices=["full", "compact", "targeted", "examples"],
        default=None,
        help="Optional override. By default this is read from cv_winner.json/configuration.json",
    )
    parser.add_argument(
        "--truncation-strategy",
        choices=["head", "head_tail"],
        default="head_tail",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of parallel workers sharing the parquet-file list",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based index of this worker; files are assigned by sorted-position modulo num-shards",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.max_seq_length <= 0:
        parser.error("--max-seq-length must be greater than zero")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("--start-date cannot be later than --end-date")
    if args.num_shards <= 0:
        parser.error("--num-shards must be greater than zero")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.adapter_dir is None and args.pipeline_dir is None:
        parser.error("provide either --adapter-dir or --pipeline-dir")
    return args


def main() -> None:
    args = parse_args()
    started = time.time()

    if not args.in_root.is_dir():
        raise SystemExit(f"Input root does not exist or is not a directory: {args.in_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    adapter_dir, experiment_id, model_metadata = resolve_adapter(
        args.adapter_dir,
        args.pipeline_dir,
        args.experiment_root,
        args.base_model,
        args.training_project_dir,
    )
    prompt_variant = args.prompt_variant or model_metadata.get("prompt_variant", "full")
    max_input_chars = (
        args.max_input_chars
        if args.max_input_chars != DEFAULT_MAX_INPUT_CHARS
        else int(model_metadata.get("max_input_chars", DEFAULT_MAX_INPUT_CHARS))
    )
    files = iter_parquet_files(
        args.in_root,
        args.start_date,
        args.end_date,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    log("=" * 70)
    log("USLNDA LoRA topic classification starting")
    log(f"Input root: {args.in_root}")
    log(f"Output root: {args.out_root}")
    log(f"Date range: {args.start_date or 'beginning'} through {args.end_date or 'end'}")
    log(f"Base model: {args.base_model}")
    log(f"Adapter: {adapter_dir}")
    log(f"Experiment: {experiment_id}")
    log(f"Prompt variant: {prompt_variant}")
    log(f"Max input chars: {max_input_chars}")
    log(f"Batch size: {args.batch_size}")
    log(f"Parallel shard: {args.shard_index + 1}/{args.num_shards}")
    log(f"Selected parquet files for this shard: {len(files):,}")
    log("=" * 70)

    if not files:
        log("No matching parquet files found")
        return

    model, tokenizer, base = load_model(args.base_model, adapter_dir)
    total_rows = 0
    total_classified = 0
    status_counts: Dict[str, int] = {}

    try:
        for file_path in tqdm(files, desc="Classifying parquet files"):
            rows, classified, status = process_file(
                file_path=file_path,
                in_root=args.in_root,
                out_root=args.out_root,
                experiment_id=experiment_id,
                model=model,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_input_chars=max_input_chars,
                max_seq_length=args.max_seq_length,
                prompt_variant=prompt_variant,
                truncation_strategy=args.truncation_strategy,
                overwrite_existing=args.overwrite_existing,
            )
            total_rows += rows
            total_classified += classified
            status_counts[status] = status_counts.get(status, 0) + 1
    finally:
        del model, base, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log("=" * 70)
    log(f"Shard: {args.shard_index + 1}/{args.num_shards}")
    log(f"Files selected by this shard: {len(files):,}")
    log(f"Rows read: {total_rows:,}")
    log(f"Articles classified: {total_classified:,}")
    log(f"File statuses: {status_counts}")
    log(f"Elapsed seconds: {time.time() - started:,.1f}")
    log("USLNDA LoRA topic classification complete")
    log("=" * 70)


if __name__ == "__main__":
    main()
