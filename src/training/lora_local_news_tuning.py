#!/usr/bin/env python3
"""
QLoRA tuning, cross-validation, final training, and final-test evaluation for
12-class local-news topic classification.

Modes
-----
MODE=screen
    Uses a fixed development/validation/final-test partition. Trains one
    configuration, selects the checkpoint with minimum validation loss, and
    reports generation-based validation metrics. The final test set is not used.

MODE=cv
    Runs stratified K-fold cross-validation on the development portion only.
    Every outer training fold is split again into training and checkpoint-
    validation subsets. The fixed final test set remains untouched.

MODE=final
    Uses an internal validation split to find the best epoch, then trains a
    fresh adapter on the complete development set for that number of epochs.
    The final test set is not evaluated.

MODE=test
    Loads FINAL_ADAPTER_DIR and evaluates it once on the fixed final test set.

The script deliberately does not globally undersample the dataset. Optional
minority-class oversampling is applied only to training data.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# =============================================================================
# Configuration
# =============================================================================


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Config:
    mode: str = os.environ.get("MODE", "screen").strip().lower()
    experiment_id: str = os.environ.get("EXPERIMENT_ID", "baseline")

    csv_path: str = os.environ.get("CSV_PATH", "data/labels_with_title.csv")
    round2_csv_path: str = os.environ.get(
        "ROUND2_CSV_PATH", "data/gangani_local_news_labeling_round_2_v2.csv"
    )
    output_root: str = os.environ.get("OUTPUT_ROOT", "lora_local_news_experiments")
    final_adapter_dir: str = os.environ.get("FINAL_ADAPTER_DIR", "")

    base_model: str = os.environ.get(
        "BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
    )
    seed: int = int(os.environ.get("SEED", "42"))
    final_test_size: float = float(os.environ.get("FINAL_TEST_SIZE", "0.20"))
    validation_size_within_development: float = float(
        os.environ.get("VALIDATION_SIZE_WITHIN_DEVELOPMENT", "0.10")
    )
    n_splits: int = int(os.environ.get("N_SPLITS", "5"))

    max_input_chars: int = int(os.environ.get("MAX_INPUT_CHARS", "2500"))
    max_seq_length: int = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
    truncation_strategy: str = os.environ.get(
        "TRUNCATION_STRATEGY", "head_tail"
    ).strip().lower()
    prompt_variant: str = os.environ.get("PROMPT_VARIANT", "full").strip().lower()

    train_batch_size: int = int(os.environ.get("BATCH_SIZE_TRAIN", "1"))
    eval_batch_size: int = int(os.environ.get("BATCH_SIZE_EVAL", "2"))
    gradient_accumulation_steps: int = int(
        os.environ.get("GRAD_ACCUM_STEPS", "16")
    )
    max_epochs: float = float(os.environ.get("NUM_EPOCHS", "6"))
    learning_rate: float = float(os.environ.get("LEARNING_RATE", "1e-4"))
    warmup_ratio: float = float(os.environ.get("WARMUP_RATIO", "0.05"))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", "0.0"))
    logging_steps: int = int(os.environ.get("LOGGING_STEPS", "10"))

    lora_r: int = int(os.environ.get("LORA_R", "16"))
    lora_alpha: int = int(os.environ.get("LORA_ALPHA", "32"))
    lora_dropout: float = float(os.environ.get("LORA_DROPOUT", "0.05"))
    target_modules: str = os.environ.get("TARGET_MODULES", "all").strip().lower()

    early_stopping_patience: int = int(
        os.environ.get("EARLY_STOPPING_PATIENCE", "2")
    )
    early_stopping_threshold: float = float(
        os.environ.get("EARLY_STOPPING_THRESHOLD", "0.001")
    )

    oversample_training: bool = env_bool("OVERSAMPLE_TRAINING", False)
    oversample_cap_multiplier: float = float(
        os.environ.get("OVERSAMPLE_CAP_MULTIPLIER", "1.0")
    )
    max_rows: Optional[int] = (
        int(os.environ["MAX_ROWS"])
        if os.environ.get("MAX_ROWS", "").strip() not in {"", "None", "none"}
        else None
    )


CFG = Config()
TITLE_COL = "title"
TEXT_COL = "text"
LABEL_COL = "final_label"


# =============================================================================
# Taxonomy and prompts
# =============================================================================

TAXONOMY: Dict[str, str] = {
    "Politics & Government": (
        "News and discussion about governance, elections, public policy, "
        "political institutions, public administration, government services, "
        "civic institutions, public programs, legislation, political actors, "
        "civil rights, diplomacy, and government decision-making at local, "
        "regional, national, and international levels. Includes public "
        "services, nonprofit and civic institutions, and government-supported "
        "community infrastructure. IMPORTANT: a story is NOT automatically "
        "Politics & Government just because it mentions a government office, "
        "agency, official, or title. Classify by what the story is actually ABOUT."
    ),
    "Crime & Public Safety": (
        "News and discussion about crime, policing, courts, justice, public "
        "safety, emergencies, disasters, conflict, violence, terrorism, "
        "accidents, and emergency response. Includes criminal activities, "
        "legal proceedings, civil unrest, threats to public order or safety, "
        "official safety warnings, recalls, arrests, investigations, scams, "
        "cybercrime, or hacking."
    ),
    "Economy & Business": (
        "News and discussion about economic activity, employment, labour, "
        "business, industry, trade, finance, entrepreneurship, markets, "
        "utilities, economic development, companies, jobs, inflation, wages, "
        "banking, and broader economic conditions."
    ),
    "Education": (
        "News and discussion related to formal and informal education, "
        "including schools, universities, curricula, teaching, learning, "
        "educational policy, teachers, students, training, educational "
        "opportunities, school board elections or meetings, and school athletics "
        "or extracurricular programs."
    ),
    "Health": (
        "News and discussion concerning physical and mental health, healthcare "
        "systems, medicine, disease, public health, hospitals, medical treatment, "
        "vaccines, healthcare policy, wellness, and health services."
    ),
    "Environment & Weather": (
        "News and discussion about the natural environment, climate, "
        "sustainability, weather, environmental hazards, conservation, pollution, "
        "natural resources, ecosystems, weather forecasts or warnings, "
        "environmental planning, and climate-related impacts."
    ),
    "Transportation": (
        "News and discussion about transportation systems, mobility, roads, "
        "traffic, public transit, transportation infrastructure, commuting, "
        "transportation policy, and travel accessibility."
    ),
    "Housing": (
        "News and discussion concerning housing, real estate, rent, home "
        "ownership, urban development, zoning, affordability, homelessness, "
        "housing markets, land use, and residential planning issues."
    ),
    "Lifestyle, Arts & Entertainment": (
        "News and discussion about arts, culture, media, entertainment, leisure, "
        "travel, food, recreation, hobbies, fashion, social and cultural events, "
        "literature, music, film, television, gaming, cultural heritage, social "
        "media culture, religion, family life, community events, everyday "
        "lifestyle activities, human-interest stories, and quality-of-life stories. "
        "Obituaries or death notices belong in Other unless the main content is "
        "specifically about a notable artistic, cultural, or entertainment career."
    ),
    "Sports": (
        "News and discussion about sports, athletes, teams, competitions, "
        "sporting events, sports organizations, coaching, venues, achievements, "
        "and sports-related controversies."
    ),
    "Science & Technology": (
        "News and discussion about science, research, engineering, innovation, "
        "technology, computing, artificial intelligence, scientific institutions, "
        "scientific discoveries, or technological development."
    ),
    "Other": (
        "Content that does not clearly belong to any predefined category, "
        "contains insufficient information for reliable classification, covers "
        "miscellaneous topics outside the taxonomy, is a paywall/broken-page notice "
        "with no real article content, or is written in a language other than English."
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


def taxonomy_block(compact: bool) -> str:
    if compact:
        return "\n".join(f"- {label}: {definition}" for label, definition in TAXONOMY.items())
    return "\n".join(
        f"{i}. {label}: {definition}" for i, (label, definition) in enumerate(TAXONOMY.items(), 1)
    )


def system_prompt() -> str:
    variant = CFG.prompt_variant
    if variant not in {"full", "compact", "targeted", "examples"}:
        raise ValueError(
            f"PROMPT_VARIANT must be full, compact, targeted, or examples; got {variant!r}."
        )

    prompt = (
        "You are a precise local-news topic classification system.\n\n"
        "Choose exactly ONE category from the fixed taxonomy below.\n\n"
        f"Categories:\n{taxonomy_block(compact=(variant != 'full'))}\n"
    )
    if variant in {"full", "targeted", "examples"}:
        prompt += f"\n\n{BOUNDARY_RULES}"
    if variant == "examples":
        prompt += f"\n\n{BOUNDARY_EXAMPLES}"

    prompt += (
        "\n\nOutput rules:\n"
        "- Return only one category name copied verbatim from the taxonomy.\n"
        "- Do not provide confidence, reasoning, notes, numbering, or extra text.\n"
        "- Never invent a category or use a synonym.\n"
        "- Exact output example: Economy & Business"
    )
    return prompt


# =============================================================================
# General helpers
# =============================================================================


class CudaCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and state.global_step and state.global_step % 10 == 0:
            torch.cuda.empty_cache()
        return control


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_hf_token() -> Optional[str]:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    return None


def unload(*objects) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def clean_boilerplate(text: str) -> str:
    lines = []
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


def truncate_text(text: str) -> str:
    text = clean_boilerplate(str(text).strip())
    if len(text) <= CFG.max_input_chars:
        return text
    if CFG.truncation_strategy == "head":
        return text[: CFG.max_input_chars] + " [...]"
    if CFG.truncation_strategy == "head_tail":
        head = int(CFG.max_input_chars * 0.80)
        tail = CFG.max_input_chars - head
        return text[:head] + "\n[...]\n" + text[-tail:]
    raise ValueError("TRUNCATION_STRATEGY must be 'head' or 'head_tail'.")


def build_user_prompt(title: str, article_text: str) -> str:
    return (
        "Article title:\n"
        f"{str(title).strip()}\n\n"
        "Use the title as a strong signal, but classify according to the article's main subject.\n\n"
        "Article body:\n"
        '"""\n'
        f"{truncate_text(article_text)}\n"
        '"""\n\n'
        "Category:"
    )


def build_messages(title: str, article_text: str, gold_label: Optional[str]) -> List[dict]:
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": build_user_prompt(title, article_text)},
    ]
    if gold_label is not None:
        messages.append({"role": "assistant", "content": gold_label})
    return messages


def render_messages(tokenizer, messages: List[dict], add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def match_label(text: str) -> str:
    if not text:
        return "UNPARSEABLE"
    first_line = text.strip().splitlines()[0].strip()
    cleaned = first_line.strip().strip("\"'.,` ")
    cleaned = re.sub(
        r"^(category\s*:\s*|\d+\.\s*)", "", cleaned, flags=re.IGNORECASE
    ).strip()
    exact = LABEL_SET_LOWER.get(cleaned.lower())
    if exact:
        return exact
    for label in sorted(LABELS, key=len, reverse=True):
        if re.match(rf"^{re.escape(label)}(?:\s*$|\s*[|:.,-])", cleaned, flags=re.I):
            return label
    return "UNPARSEABLE"


# =============================================================================
# Data loading and fixed splits
# =============================================================================


def make_row_id(source: str, source_index: int, title: str, text: str) -> str:
    payload = f"{source}|{source_index}|{title}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()


def load_data() -> pd.DataFrame:
    df1 = pd.read_csv(CFG.csv_path)
    df2 = pd.read_csv(CFG.round2_csv_path)

    required1 = {TITLE_COL, TEXT_COL, LABEL_COL}
    required2 = {"title", "content", "label"}
    if not required1.issubset(df1.columns):
        raise ValueError(f"{CFG.csv_path} must contain {sorted(required1)}; found {list(df1.columns)}")
    if not required2.issubset(df2.columns):
        raise ValueError(
            f"{CFG.round2_csv_path} must contain {sorted(required2)}; found {list(df2.columns)}"
        )

    df1 = df1[[TITLE_COL, TEXT_COL, LABEL_COL]].copy()
    df1["dataset_source"] = "labels"
    df1["source_index"] = np.arange(len(df1))

    df2 = df2[["title", "content", "label"]].rename(
        columns={"content": TEXT_COL, "label": LABEL_COL}
    )
    df2["dataset_source"] = "round_2"
    df2["source_index"] = np.arange(len(df2))

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str).str.strip()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[(df[TEXT_COL] != "") & df[LABEL_COL].isin(LABELS)].copy()

    conflict_count = df.groupby(TEXT_COL)[LABEL_COL].nunique()
    conflict_texts = set(conflict_count[conflict_count > 1].index)
    if conflict_texts:
        print(f"Dropping {len(conflict_texts)} texts with conflicting labels.")
        df = df[~df[TEXT_COL].isin(conflict_texts)].copy()

    before = len(df)
    df = df.drop_duplicates(subset=[TEXT_COL, LABEL_COL]).copy()
    print(f"Removed {before - len(df)} exact duplicate article-label rows.")

    df["row_id"] = [
        make_row_id(s, int(i), t, x)
        for s, i, t, x in zip(
            df["dataset_source"], df["source_index"], df[TITLE_COL], df[TEXT_COL]
        )
    ]

    if CFG.max_rows is not None:
        min_needed = len(LABELS) * 3
        if CFG.max_rows < min_needed:
            raise ValueError(f"MAX_ROWS must be at least {min_needed} for stratified splitting.")
        per_class = CFG.max_rows // len(LABELS)
        df = (
            df.groupby(LABEL_COL, group_keys=False)
            .sample(n=min(per_class, df[LABEL_COL].value_counts().min()), random_state=CFG.seed)
            .sample(frac=1, random_state=CFG.seed)
        )

    df = df.reset_index(drop=True)
    counts = df[LABEL_COL].value_counts().reindex(LABELS, fill_value=0)
    missing = counts[counts == 0].index.tolist()
    if missing:
        raise ValueError(f"Missing classes after cleaning: {missing}")
    print("\nRows per class after cleaning, without global undersampling:")
    print(counts.to_string())
    print(f"Total rows: {len(df)}")
    return df


def fixed_development_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    development, final_test = train_test_split(
        df,
        test_size=CFG.final_test_size,
        random_state=CFG.seed,
        stratify=df[LABEL_COL],
    )
    return development.reset_index(drop=True), final_test.reset_index(drop=True)


def train_validation_split(development: pd.DataFrame, seed_offset: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        development,
        test_size=CFG.validation_size_within_development,
        random_state=CFG.seed + seed_offset,
        stratify=development[LABEL_COL],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def oversample_training_only(train_df: pd.DataFrame) -> pd.DataFrame:
    if not CFG.oversample_training:
        return train_df.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)

    counts = train_df[LABEL_COL].value_counts()
    max_count = int(counts.max())
    target = max(1, int(max_count * CFG.oversample_cap_multiplier))
    parts = []
    for label in LABELS:
        part = train_df[train_df[LABEL_COL] == label]
        if part.empty:
            raise ValueError(f"Training split has no rows for {label}.")
        parts.append(
            part.sample(
                n=target,
                replace=len(part) < target,
                random_state=CFG.seed,
            )
        )
    result = pd.concat(parts, ignore_index=True)
    return result.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)


def save_split_manifests(
    out_dir: Path,
    development: pd.DataFrame,
    final_test: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
) -> None:
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    development[["row_id", LABEL_COL, "dataset_source"]].to_csv(
        split_dir / "development.csv", index=False
    )
    final_test[["row_id", LABEL_COL, "dataset_source"]].to_csv(
        split_dir / "final_test_UNTOUCHED.csv", index=False
    )
    if train_df is not None:
        train_df[["row_id", LABEL_COL, "dataset_source"]].to_csv(
            split_dir / "train.csv", index=False
        )
    if val_df is not None:
        val_df[["row_id", LABEL_COL, "dataset_source"]].to_csv(
            split_dir / "validation.csv", index=False
        )


# =============================================================================
# Tokenization and model construction
# =============================================================================


def make_training_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    rows = []
    for _, row in df.iterrows():
        prompt_messages = build_messages(row[TITLE_COL], row[TEXT_COL], None)
        full_messages = build_messages(row[TITLE_COL], row[TEXT_COL], row[LABEL_COL])

        prompt_text = render_messages(tokenizer, prompt_messages, True)
        full_text = render_messages(tokenizer, full_messages, False)

        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=CFG.max_seq_length,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        full = tokenizer(
            full_text,
            truncation=True,
            max_length=CFG.max_seq_length,
            padding=False,
            add_special_tokens=False,
        )
        input_ids = full["input_ids"]
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        if not any(x != -100 for x in labels):
            raise ValueError(
                "Gold answer was truncated. Reduce MAX_INPUT_CHARS or increase MAX_SEQ_LENGTH."
            )
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": full["attention_mask"],
                "labels": labels,
            }
        )
    return Dataset.from_list(rows)


def quantization_config() -> BitsAndBytesConfig:
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def target_module_list() -> List[str]:
    if CFG.target_modules == "attention":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if CFG.target_modules == "all":
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    raise ValueError("TARGET_MODULES must be 'attention' or 'all'.")


def load_trainable_model_and_tokenizer():
    token = get_hf_token()
    if token is None:
        print("WARNING: no Hugging Face token was found; gated models may fail.")

    tokenizer = AutoTokenizer.from_pretrained(CFG.base_model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        CFG.base_model,
        quantization_config=quantization_config(),
        device_map="auto",
        token=token,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=CFG.lora_r,
            lora_alpha=CFG.lora_alpha,
            lora_dropout=CFG.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_module_list(),
        ),
    )
    model.print_trainable_parameters()
    return model, tokenizer


def make_training_arguments(output_dir: Path, epochs: float, with_validation: bool) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=CFG.train_batch_size,
        per_device_eval_batch_size=CFG.train_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=CFG.learning_rate,
        warmup_ratio=CFG.warmup_ratio,
        weight_decay=CFG.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=CFG.logging_steps,
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        group_by_length=True,
        save_total_limit=1,
        remove_unused_columns=False,
        seed=CFG.seed,
        data_seed=CFG.seed,
    )

    if with_validation:
        kwargs.update(
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        params = inspect.signature(TrainingArguments.__init__).parameters
        if "eval_strategy" in params:
            kwargs["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in params:
            kwargs["evaluation_strategy"] = "epoch"
        else:
            raise RuntimeError(
                "Installed transformers.TrainingArguments supports neither "
                "eval_strategy nor evaluation_strategy. Upgrade transformers."
            )
    else:
        kwargs.update(save_strategy="no")

    return TrainingArguments(**kwargs)


def train_adapter(
    train_df: pd.DataFrame,
    checkpoint_val_df: Optional[pd.DataFrame],
    output_dir: Path,
    epochs: float,
    early_stopping: bool,
) -> Tuple[Path, float, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_trainable_model_and_tokenizer()
    train_df_used = oversample_training_only(train_df)
    train_ds = make_training_dataset(train_df_used, tokenizer)
    val_ds = (
        make_training_dataset(checkpoint_val_df, tokenizer)
        if checkpoint_val_df is not None
        else None
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    callbacks: List[TrainerCallback] = [CudaCleanupCallback()]
    if early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=CFG.early_stopping_patience,
                early_stopping_threshold=CFG.early_stopping_threshold,
            )
        )

    trainer = Trainer(
        model=model,
        args=make_training_arguments(
            output_dir / "trainer_outputs",
            epochs=epochs,
            with_validation=checkpoint_val_df is not None,
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    train_result = trainer.train()
    best_epoch = float(trainer.state.epoch or epochs)
    if trainer.state.best_model_checkpoint:
        print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Selected epoch: {best_epoch:.4f}")

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    history_df = pd.DataFrame(trainer.state.log_history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(
            {
                "selected_epoch": best_epoch,
                "global_step": int(trainer.state.global_step),
                "best_metric_eval_loss": trainer.state.best_metric,
                "train_metrics": train_result.metrics,
                "n_train_original": int(len(train_df)),
                "n_train_after_training_only_oversampling": int(len(train_df_used)),
                "n_checkpoint_validation": int(len(checkpoint_val_df)) if checkpoint_val_df is not None else 0,
            },
            f,
            indent=2,
            default=str,
        )

    unload(trainer, model, tokenizer, train_ds, val_ds)
    return adapter_dir, best_epoch, history_df


# =============================================================================
# Generation-based evaluation and reporting
# =============================================================================


@torch.no_grad()
def evaluate_adapter(adapter_dir: Path, eval_df: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    token = get_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        CFG.base_model,
        quantization_config=quantization_config(),
        device_map="auto",
        token=token,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    records = []
    started = time.time()

    for start in range(0, len(eval_df), CFG.eval_batch_size):
        batch = eval_df.iloc[start : start + CFG.eval_batch_size]
        prompts = [
            render_messages(
                tokenizer,
                build_messages(row[TITLE_COL], row[TEXT_COL], None),
                True,
            )
            for _, row in batch.iterrows()
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CFG.max_seq_length,
        ).to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for (_, row), raw in zip(batch.iterrows(), decoded):
            pred = match_label(raw)
            records.append(
                {
                    "row_id": row["row_id"],
                    "dataset_source": row["dataset_source"],
                    "title": row[TITLE_COL],
                    "text": row[TEXT_COL],
                    "gold_label": row[LABEL_COL],
                    "raw_model_output": raw.strip(),
                    "predicted_label": pred,
                    "correct": pred == row[LABEL_COL],
                }
            )
        print(f"Evaluated {min(start + CFG.eval_batch_size, len(eval_df))}/{len(eval_df)}")

    pred_df = pd.DataFrame(records)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    report_labels = LABELS + (
        ["UNPARSEABLE"] if "UNPARSEABLE" in set(pred_df["predicted_label"]) else []
    )
    acc = accuracy_score(pred_df["gold_label"], pred_df["predicted_label"])
    macro_f1 = f1_score(
        pred_df["gold_label"], pred_df["predicted_label"],
        labels=LABELS, average="macro", zero_division=0,
    )
    weighted_f1 = f1_score(
        pred_df["gold_label"], pred_df["predicted_label"],
        labels=LABELS, average="weighted", zero_division=0,
    )
    precision, recall, f1_values, support = precision_recall_fscore_support(
        pred_df["gold_label"], pred_df["predicted_label"],
        labels=report_labels, zero_division=0,
    )
    pd.DataFrame(
        {
            "label": report_labels,
            "precision": precision,
            "recall": recall,
            "f1": f1_values,
            "support": support,
        }
    ).to_csv(output_dir / "per_class_metrics.csv", index=False)

    report = classification_report(
        pred_df["gold_label"], pred_df["predicted_label"],
        labels=report_labels, zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(
        pred_df["gold_label"], pred_df["predicted_label"], labels=report_labels
    )
    pd.DataFrame(cm, index=report_labels, columns=report_labels).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "n_eval": int(len(pred_df)),
        "n_correct": int(pred_df["correct"].sum()),
        "n_unparseable": int((pred_df["predicted_label"] == "UNPARSEABLE").sum()),
        "elapsed_seconds": float(time.time() - started),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(report)
    unload(model, base_model, tokenizer)
    return metrics


# =============================================================================
# Experiment modes
# =============================================================================


def experiment_dir() -> Path:
    model_name = CFG.base_model.replace("/", "__")
    return Path(CFG.output_root) / model_name / CFG.experiment_id / CFG.mode


def save_config(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = asdict(CFG)
    config_payload["system_prompt"] = system_prompt()
    config_payload["effective_batch_size"] = (
        CFG.train_batch_size * CFG.gradient_accumulation_steps
    )
    with open(out_dir / "configuration.json", "w") as f:
        json.dump(config_payload, f, indent=2)


def run_screen(df: pd.DataFrame, out_dir: Path) -> None:
    development, final_test = fixed_development_test_split(df)
    train_df, val_df = train_validation_split(development)
    save_split_manifests(out_dir, development, final_test, train_df, val_df)

    adapter_dir, best_epoch, _ = train_adapter(
        train_df=train_df,
        checkpoint_val_df=val_df,
        output_dir=out_dir / "training",
        epochs=CFG.max_epochs,
        early_stopping=True,
    )
    metrics = evaluate_adapter(adapter_dir, val_df, out_dir / "validation_evaluation")
    summary = {
        **metrics,
        "selected_epoch": best_epoch,
        "adapter_dir": str(adapter_dir),
        "final_test_evaluated": False,
    }
    with open(out_dir / "screen_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def run_cv(df: pd.DataFrame, out_dir: Path) -> None:
    development, final_test = fixed_development_test_split(df)
    save_split_manifests(out_dir, development, final_test)
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    fold_rows = []
    all_predictions = []

    for fold_id, (outer_train_idx, outer_eval_idx) in enumerate(
        skf.split(development, development[LABEL_COL]), 1
    ):
        print("\n" + "=" * 80)
        print(f"CV fold {fold_id}/{CFG.n_splits}")
        print("=" * 80)
        outer_train = development.iloc[outer_train_idx].reset_index(drop=True)
        outer_eval = development.iloc[outer_eval_idx].reset_index(drop=True)
        inner_train, checkpoint_val = train_validation_split(outer_train, seed_offset=fold_id)

        fold_dir = out_dir / f"fold_{fold_id}"
        adapter_dir, selected_epoch, _ = train_adapter(
            train_df=inner_train,
            checkpoint_val_df=checkpoint_val,
            output_dir=fold_dir / "training",
            epochs=CFG.max_epochs,
            early_stopping=True,
        )
        metrics = evaluate_adapter(adapter_dir, outer_eval, fold_dir / "outer_fold_evaluation")
        predictions = pd.read_csv(fold_dir / "outer_fold_evaluation" / "predictions.csv")
        predictions["fold"] = fold_id
        all_predictions.append(predictions)
        fold_rows.append({"fold": fold_id, "selected_epoch": selected_epoch, **metrics})

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "fold_summary.csv", index=False)
    pooled = pd.concat(all_predictions, ignore_index=True)
    pooled.to_csv(out_dir / "all_fold_predictions.csv", index=False)

    pooled_macro_f1 = f1_score(
        pooled["gold_label"], pooled["predicted_label"],
        labels=LABELS, average="macro", zero_division=0,
    )
    pooled_accuracy = accuracy_score(pooled["gold_label"], pooled["predicted_label"])
    summary = {
        "mean_fold_accuracy": float(fold_df["accuracy"].mean()),
        "std_fold_accuracy": float(fold_df["accuracy"].std(ddof=1)),
        "mean_fold_macro_f1": float(fold_df["macro_f1"].mean()),
        "std_fold_macro_f1": float(fold_df["macro_f1"].std(ddof=1)),
        "pooled_accuracy": float(pooled_accuracy),
        "pooled_macro_f1": float(pooled_macro_f1),
        "mean_selected_epoch": float(fold_df["selected_epoch"].mean()),
        "final_test_evaluated": False,
    }
    with open(out_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def run_final(df: pd.DataFrame, out_dir: Path) -> None:
    development, final_test = fixed_development_test_split(df)
    train_df, val_df = train_validation_split(development)
    save_split_manifests(out_dir, development, final_test, train_df, val_df)

    # Stage 1: select epoch using an internal validation set.
    _, selected_epoch, _ = train_adapter(
        train_df=train_df,
        checkpoint_val_df=val_df,
        output_dir=out_dir / "epoch_selection",
        epochs=CFG.max_epochs,
        early_stopping=True,
    )

    # Stage 2: train a fresh adapter on all development data for the selected epoch.
    selected_epoch_for_full_training = max(1.0, round(selected_epoch, 2))
    final_adapter_dir, _, _ = train_adapter(
        train_df=development,
        checkpoint_val_df=None,
        output_dir=out_dir / "final_training_all_development",
        epochs=selected_epoch_for_full_training,
        early_stopping=False,
    )
    summary = {
        "selected_epoch_from_internal_validation": selected_epoch,
        "epochs_used_for_full_development_training": selected_epoch_for_full_training,
        "final_adapter_dir": str(final_adapter_dir),
        "n_development": int(len(development)),
        "n_final_test_untouched": int(len(final_test)),
        "final_test_evaluated": False,
    }
    with open(out_dir / "final_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def run_test(df: pd.DataFrame, out_dir: Path) -> None:
    if not CFG.final_adapter_dir:
        raise ValueError("MODE=test requires FINAL_ADAPTER_DIR.")
    adapter_dir = Path(CFG.final_adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Final adapter directory does not exist: {adapter_dir}")
    development, final_test = fixed_development_test_split(df)
    save_split_manifests(out_dir, development, final_test)
    metrics = evaluate_adapter(adapter_dir, final_test, out_dir / "FINAL_TEST_EVALUATION")
    with open(out_dir / "final_test_summary.json", "w") as f:
        json.dump(
            {
                **metrics,
                "adapter_dir": str(adapter_dir),
                "warning": "This fixed final test set should not be used for further tuning.",
            },
            f,
            indent=2,
        )


def main() -> None:
    if CFG.mode not in {"screen", "cv", "final", "test"}:
        raise ValueError("MODE must be screen, cv, final, or test.")
    set_seed(CFG.seed)
    out_dir = experiment_dir()
    save_config(out_dir)
    print("Configuration:")
    print(json.dumps(asdict(CFG), indent=2))
    print(f"Effective batch size: {CFG.train_batch_size * CFG.gradient_accumulation_steps}")
    print(f"Outputs: {out_dir}")

    df = load_data()
    if CFG.mode == "screen":
        run_screen(df, out_dir)
    elif CFG.mode == "cv":
        run_cv(df, out_dir)
    elif CFG.mode == "final":
        run_final(df, out_dir)
    else:
        run_test(df, out_dir)


if __name__ == "__main__":
    main()
