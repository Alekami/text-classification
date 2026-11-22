import html
import re
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

TOKENIZE = TfidfVectorizer().build_tokenizer()
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
USER_PATTERN = re.compile(r"(?<![\w@])@\w+")


def normalize_tweet(text: str) -> str:
    text = html.unescape(text)
    text = URL_PATTERN.sub(" urltoken ", text)
    text = USER_PATTERN.sub(" usertoken ", text)
    return text.lower()


def normalized_text_key(text: str) -> str:
    """Group tweets with the same ordered tokens after normalization."""
    return " ".join(TOKENIZE(normalize_tweet(text)))


@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame


def text_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("text")["target"].agg(rows="size", unique_targets="nunique")


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    if frame[["id", "text", "target"]].isna().any().any():
        raise ValueError("Columns id, text, and target must be complete.")
    if set(frame["target"]) != {0, 1}:
        raise ValueError("The training file must contain both classes, 0 and 1.")
    if frame["id"].duplicated().any():
        raise ValueError("Tweet IDs must be unique.")

    statistics = text_statistics(frame)
    conflicts = statistics.index[statistics["unique_targets"] > 1]
    return (
        frame.loc[~frame["text"].isin(conflicts), ["id", "text", "target"]]
        .drop_duplicates(subset="text")
        .reset_index(drop=True)
    )


def make_split(
    frame: pd.DataFrame, val_size: float = 0.2, seed: int = 42, *, group_folds: int | None = None
) -> DataSplit:
    cleaned = prepare_data(frame)
    if group_folds is None:
        train, val = train_test_split(
            cleaned,
            test_size=val_size,
            stratify=cleaned["target"],
            random_state=seed,
        )
    else:
        groups = cleaned["text"].map(normalized_text_key)
        splitter = StratifiedGroupKFold(n_splits=group_folds, shuffle=True, random_state=seed)
        train_indices, val_indices = next(splitter.split(cleaned, cleaned["target"], groups))
        train, val = cleaned.iloc[train_indices], cleaned.iloc[val_indices]
        if not set(groups.iloc[train_indices]).isdisjoint(groups.iloc[val_indices]):
            raise ValueError("The training and validation groups overlap.")
    if not set(train["text"]).isdisjoint(val["text"]):
        raise ValueError("The training and validation texts overlap.")
    return DataSplit(train=train, val=val)
