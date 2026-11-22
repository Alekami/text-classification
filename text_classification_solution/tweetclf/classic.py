from functools import partial
from time import perf_counter

import nltk
import numpy as np
from lightgbm import LGBMClassifier
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import TOKENIZE, normalize_tweet
from .metrics import benchmark_ms, classification_metrics

STEMMER = SnowballStemmer("english")
LEMMATIZER = WordNetLemmatizer()
WORDNET_POS = {"J": "a", "V": "v", "N": "n", "R": "r"}


def stem_tokens(text: str, *, stop_words: frozenset[str] = frozenset()) -> list[str]:
    return [STEMMER.stem(token) for token in TOKENIZE(text) if token not in stop_words]


def lemmatize_tokens(text: str) -> list[str]:
    lemmas = []
    for token, tag in nltk.pos_tag(TOKENIZE(text)):
        pos = WORDNET_POS.get(tag[0])
        lemma = LEMMATIZER.lemmatize(token, pos=pos) if pos else token
        lemmas.append(lemma)
    return lemmas


def make_vectorizer(experiment: dict):
    options = {
        "lowercase": experiment.get("lowercase", True),
        "stop_words": experiment.get("stop_words"),
        "ngram_range": tuple(experiment.get("ngram_range", [1, 1])),
    }
    tokenizer = {"stem": stem_tokens, "lemma": lemmatize_tokens}.get(
        experiment.get("morphology")
    )
    if tokenizer is stem_tokens and options["stop_words"] is not None:
        stop_words = options.pop("stop_words")
        if isinstance(stop_words, str):
            if stop_words != "english":
                raise ValueError(f"Unknown stop-word list: {stop_words}")
            stop_words = ENGLISH_STOP_WORDS
        tokenizer = partial(stem_tokens, stop_words=frozenset(stop_words))
    if tokenizer is not None:
        options.update(tokenizer=tokenizer, token_pattern=None)
    if experiment.get("normalize_tweet", False):
        options["preprocessor"] = normalize_tweet
    if experiment["vectorizer"] == "bow":
        return CountVectorizer(**options, dtype=np.float64)
    return TfidfVectorizer(**options)


def make_classifier(name: str, config: dict):
    settings = config["classic"]["classifiers"][name]
    if name == "logistic_regression":
        return LogisticRegression(**settings)
    if name == "lightgbm":
        return LGBMClassifier(
            **settings,
            random_state=config["seed"],
            n_jobs=config["benchmark"]["cpu_threads"],
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    raise ValueError(f"Unknown classifier: {name}")


def predict_labels(classifier, features):
    if isinstance(classifier, LGBMClassifier):
        probabilities = classifier.booster_.predict(features, num_threads=classifier.n_jobs)
        return classifier.classes_[(probabilities > 0.5).astype(int)]
    return classifier.predict(features)


def run_classic(split, config: dict) -> list[dict]:
    train_text = split.train["text"].tolist()
    val_text = split.val["text"].tolist()
    results = []

    for experiment in config["classic"]["experiments"]:
        vectorizer = make_vectorizer(experiment)
        start = perf_counter()
        train_vectors = vectorizer.fit_transform(train_text)
        vectorization_s = perf_counter() - start
        val_vectors = vectorizer.transform(val_text)

        for name in config["classic"]["classifiers"]:
            classifier = make_classifier(name, config)
            start = perf_counter()
            classifier.fit(train_vectors, split.train["target"])
            fit_s = vectorization_s + perf_counter() - start
            prediction = predict_labels(classifier, val_vectors)

            def predict(texts):
                return predict_labels(classifier, vectorizer.transform(texts))

            row = {
                "stage": "classic",
                "representation": experiment["vectorizer"],
                "preprocessing": experiment["name"],
                "classifier": name,
                "features": train_vectors.shape[1],
                "device": "cpu",
                "fit_seconds": fit_s,
                "inference_ms_per_tweet": benchmark_ms(
                    predict, val_text, config["benchmark"]
                ),
                **classification_metrics(split.val["target"], prediction),
            }
            results.append(row)
            print(f"{experiment['name']} / {name}: F1={row['f1']:.4f}", flush=True)
    return results
