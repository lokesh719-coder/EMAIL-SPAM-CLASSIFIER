"""Dataset loading and shared NLTK text preprocessing."""

from __future__ import annotations

import re
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "SMSSpamCollection"
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def _download_dataset(destination: Path) -> None:
    """Download and extract the official UCI SMS Spam Collection file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / "SMSSpamCollection.zip"
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=30) as response:
            archive_path.write_bytes(response.read())
    except (OSError, urllib.error.URLError) as error:
        raise FileNotFoundError(
            "The SMS Spam Collection dataset is missing and could not be downloaded. "
            f"Place the UCI file at {destination} or check your internet connection."
        ) from error

    try:
        with zipfile.ZipFile(archive_path) as archive:
            member = next(
                (name for name in archive.namelist() if Path(name).name == "SMSSpamCollection"),
                None,
            )
            if member is None:
                raise FileNotFoundError("The downloaded UCI archive has no SMSSpamCollection file.")
            temporary_path = destination.with_suffix(".tmp")
            with archive.open(member) as source, temporary_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary_path.replace(destination)
    except (zipfile.BadZipFile, OSError) as error:
        raise RuntimeError(f"Could not extract the UCI dataset: {error}") from error


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Load, validate, and clean the UCI tab-separated dataset."""
    path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    if not path.is_file():
        if dataset_path is not None:
            raise FileNotFoundError(f"Dataset file not found: {path}")
        _download_dataset(path)

    try:
        frame = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["label", "text"],
            encoding="utf-8",
            on_bad_lines="error",
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise RuntimeError(f"Could not read dataset file {path}: {error}") from error

    frame = frame.dropna(subset=["label", "text"]).copy()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["label"].isin({"ham", "spam"})]
    frame = frame[frame["text"].ne("")]
    frame = frame.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No valid ham/spam messages were found in {path}.")
    if frame["label"].nunique() < 2:
        raise ValueError("The dataset must contain both ham and spam examples.")
    return frame


def get_stopwords() -> set[str]:
    """Return NLTK English stopwords, downloading the corpus if necessary."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        try:
            downloaded = nltk.download("stopwords", quiet=True)
            if not downloaded:
                raise LookupError("NLTK reported that the stopwords corpus download failed.")
            return set(stopwords.words("english"))
        except (LookupError, OSError) as error:
            raise RuntimeError(
                "NLTK English stopwords are unavailable. Check your internet connection "
                "or run nltk.download('stopwords') in Python."
            ) from error


def preprocess_text(text: str, stop_words: Iterable[str] | None = None) -> str:
    """Normalize one message; used unchanged by model training and prediction."""
    if not isinstance(text, str):
        raise ValueError("Message text must be a string.")

    blocked_words = set(stop_words) if stop_words is not None else get_stopwords()
    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = wordpunct_tokenize(normalized)
    return " ".join(token for token in tokens if token.isalpha() and token not in blocked_words)
