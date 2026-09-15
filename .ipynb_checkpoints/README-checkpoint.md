# News Article Topic Classification

Classify a news article into one of 12 topics using a LoRA adapter trained for **Meta Llama 3.1 8B Instruct**. The classifier accepts a public article URL, title and body text, or a collection of Parquet files.

This model assigns a topic to an article. 

## Topics

- Politics & Government
- Crime & Public Safety
- Economy & Business
- Education
- Health
- Environment & Weather
- Transportation
- Housing
- Lifestyle, Arts & Entertainment
- Sports
- Science & Technology
- Other

## Setup

You need Linux, Python 3.9 or newer, an NVIDIA GPU with a compatible CUDA installation, and a Hugging Face account with access to [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

Clone the repository and enter it:

```bash
git clone https://github.com/GANGANI/localnews-topic-classification.git
cd localnews-topic-classification
```

To obtain a Hugging Face token:

1. [Sign in to Hugging Face](https://huggingface.co/login) and open the [Llama 3.1 8B Instruct model page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Agree to its access terms and request access if your account has not already been approved.
2. Open [Settings → Access Tokens](https://huggingface.co/settings/tokens), select **New token**, and create a token with **read** access. Copy the token when it is shown.

From the repository root, create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then run the following command:

```bash
read -rsp "Hugging Face token: " HF_TOKEN; echo
export HF_TOKEN
```

When `Hugging Face token:` appears, paste the token you copied (it begins with `hf_`) and press **Enter**. The token will not appear on screen while you paste or type it. The second command makes it available to the classifier for the current terminal session. Repeat these two commands whenever you open a new terminal; do not add the token to this repository or commit it to Git.

Confirm that PyTorch can use the GPU:

```bash
python -c 'import torch; print(torch.cuda.is_available())'
```

The first classification downloads the base model. The trained adapter and tokenizer are included in `models/adapter/`.

## Classify an article

Classify a public URL:

```bash
python localnews_classifier.py --url "https://news.example.com/article"
```

Classify title and body text:

```bash
python localnews_classifier.py \
  --title "City opens new bus route" \
  --content "The new route connects downtown with nearby neighborhoods."
```

For a longer article, place the body in a UTF-8 text file:

```bash
python localnews_classifier.py \
  --title "City opens new bus route" \
  --content-file article.txt
```

The command prints JSON containing the predicted topic in `predicted_label`. URL extraction works with public HTML pages; pages that require JavaScript, a login, or a subscription may need to be supplied as text.

### Run the included HPC sample

The included `classify_article.csh` is a sample Slurm job for running classification on an HPC GPU compute node instead of consuming resources on the frontend or login node. Edit the `--title` and `--content` values in the file for your article, and adjust its module, virtual-environment path, or `#SBATCH` resource settings for your cluster.

After exporting `HF_TOKEN` as described above, submit the sample from the repository root:

```bash
mkdir -p logs
sbatch classify_article.csh
```

Slurm prints a job ID when the job is submitted. Check its status and results with:

```bash
squeue -j JOB_ID
cat logs/article_JOB_ID.out
cat logs/article_JOB_ID.err
```

The `.out` file contains the classification JSON. Model-loading progress and errors appear in the `.err` file.

## Use from Python

```python
from localnews_classifier import LocalNewsClassifier

classifier = LocalNewsClassifier()
result = classifier.classify(
    title="City opens new bus route",
    content="The new route connects downtown with nearby neighborhoods.",
)

print(result["predicted_label"])
```

Reuse the same `LocalNewsClassifier` instance when processing several articles so the model is loaded only once.

## Classify Parquet files

Input files must contain `title` and `content` columns and use this directory layout:

```text
INPUT_ROOT/collection/outlet/YYYY/MM/DD.parquet
```

Run the batch classifier:

```bash
bash scripts/classify.sh \
  --in-root /absolute/path/to/INPUT_ROOT \
  --out-root /absolute/path/to/OUTPUT_ROOT \
  --start-date 2025-11-16 \
  --end-date 2025-11-30 \
  --batch-size 8
```

Outputs preserve the input directory structure and add `predicted_label`, `raw_model_output`, and `classification_status`. Reduce `--batch-size` if GPU memory is limited.

To try the included sample:

```bash
python examples/make_example.py
bash scripts/classify.sh \
  --in-root examples/input \
  --out-root outputs/example \
  --batch-size 1
```

## Test the interface

The interface tests mock model inference and do not require a GPU:

```bash
python -m unittest discover -s tests -v
```

## Model results

The saved evaluation reports a held-out test accuracy of **86.62%** and macro F1 of **0.8708** across 314 articles. Detailed metrics and training records are available in `evaluation/` and `models/configuration.json`.
