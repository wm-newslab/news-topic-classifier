"""Single-article API and CLI. Reuse LocalNewsClassifier to load weights once."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys
from urllib.parse import urlparse
from urllib.request import Request, urlopen

REPOSITORY = Path(__file__).resolve().parent


def extract_article(url: str, timeout: float = 30) -> dict:
    """Download public HTML and return its title and main article text.

    Does not execute JavaScript or bypass paywalls. Raises ValueError when no
    usable article is extracted; network/HTTP errors propagate to the caller.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Article URL must be an absolute HTTP or HTTPS URL")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    from trafilatura import extract

    request = Request(url, headers={"User-Agent": "LocalNewsTopicClassifier/1.0", "Accept": "text/html,application/xhtml+xml"})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError(f"Expected an HTML article, received {content_type}")
        html = response.read(10 * 1024 * 1024 + 1)
        if len(html) > 10 * 1024 * 1024:
            raise ValueError("Article response exceeds 10 MiB")
        resolved_url = response.geturl()
    raw = extract(html, url=resolved_url, output_format="json",
                  with_metadata=True, include_comments=False)
    article = json.loads(raw) if raw else {}
    title = (article.get("title") or "").strip()
    content = (article.get("text") or "").strip()
    if not content:
        raise ValueError("No article body extracted; provide --title and --content instead")
    return {"url": url, "resolved_url": resolved_url, "title": title, "content": content}


class LocalNewsClassifier:
    """Load the bundled adapter lazily and reuse it across articles.

    Call classify(title, content) or classify_url(url). Requires the same CUDA
    environment and base-model access as the Parquet classifier.
    """

    def __init__(self, adapter_dir=None):
        self.adapter_dir = Path(adapter_dir or REPOSITORY / "models/adapter").resolve()
        config_path = self.adapter_dir.parent / "configuration.json"
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self._runtime = None

    def _load(self):
        if self._runtime is None:
            from src.classification import classify_uslnda_news_parallel as backend
            with contextlib.redirect_stdout(sys.stderr):
                model, tokenizer, base = backend.load_model(self.config["base_model"], self.adapter_dir)
            self._runtime = (backend, model, tokenizer, base)
        return self._runtime

    def classify(self, title: str, content: str) -> dict:
        """Return title, content, predicted_label, raw output, status and model ID.

        Empty titles are accepted, but content must be a nonempty string.
        Returned content is complete; model input follows the saved truncation.
        """
        if not isinstance(title, str) or not isinstance(content, str):
            raise TypeError("title and content must be strings")
        title, content = title.strip(), content.strip()
        if not content:
            raise ValueError("content must not be empty")
        backend, model, tokenizer, _ = self._load()
        import pandas as pd
        with contextlib.redirect_stdout(sys.stderr):
            frame = backend.classify_dataframe(
                pd.DataFrame([{"title": title, "content": content}]),
                model, tokenizer, batch_size=1,
                max_input_chars=self.config["max_input_chars"],
                max_seq_length=self.config["max_seq_length"],
                prompt_variant=self.config["prompt_variant"],
                truncation_strategy=self.config["truncation_strategy"],
            )
        return {**frame.iloc[0].to_dict(), "model_id": self.config["experiment_id"]}

    def classify_url(self, url: str, timeout: float = 30) -> dict:
        """Extract an article before loading the GPU model, then classify it."""
        article = extract_article(url, timeout=timeout)
        return {**self.classify(article["title"], article["content"]),
                "url": article["url"], "resolved_url": article["resolved_url"]}


def classify_article(title: str, content: str, *, classifier=None) -> dict:
    """Convenience function; pass a classifier instance to reuse loaded weights."""
    if classifier is None:
        classifier = LocalNewsClassifier()
    return classifier.classify(title, content)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Classify a URL or article title and content; emits JSON")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Public article URL")
    source.add_argument("--content", help="Article body text (use shell quotes)")
    source.add_argument("--content-file", type=Path, help="UTF-8 file containing the article body")
    parser.add_argument("--title", help="Article title; required with content inputs")
    parser.add_argument("--adapter-dir", type=Path, default=REPOSITORY / "models/adapter")
    parser.add_argument("--timeout", type=float, default=30, help="URL network timeout in seconds")
    args = parser.parse_args(argv)
    if args.url and args.title is not None:
        parser.error("--title cannot be combined with --url; the title is extracted")
    if not args.url and args.title is None:
        parser.error("--title is required with --content or --content-file")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        classifier = LocalNewsClassifier(args.adapter_dir)
        if args.url:
            result = classifier.classify_url(args.url, timeout=args.timeout)
        else:
            content = args.content_file.read_text(encoding="utf-8") if args.content_file else args.content
            result = classifier.classify(args.title, content)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["classification_status"] == "classified" else 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
