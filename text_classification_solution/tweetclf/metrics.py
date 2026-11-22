from statistics import median
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(target, prediction) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, prediction, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy_score(target, prediction)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def benchmark_ms(predict, texts: list[str], settings: dict) -> float:
    """Time raw-text predictions with a fixed batch size after a warm-up."""
    batch_size = settings["batch_size"]
    sample = texts[:settings["sample_size"]]
    batches = [sample[i:i + batch_size] for i in range(0, len(sample), batch_size)]
    predict(batches[0])

    durations = []
    for _ in range(settings["repeats"]):
        start = perf_counter()
        for batch in batches:
            prediction = np.asarray(predict(batch))
            if prediction.shape != (len(batch),):
                raise ValueError("Expected one prediction per tweet.")
        durations.append(perf_counter() - start)
    return 1000 * median(durations) / len(sample)
