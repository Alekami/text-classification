from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

from .classic import make_classifier, predict_labels
from .metrics import benchmark_ms, classification_metrics


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use CPU or install a CUDA build of PyTorch.")
    return torch.device(name)


def model_parameters(settings: dict, cache_dir: Path) -> dict:
    return {"cache_dir": cache_dir, "revision": settings["revision"]}


@torch.inference_mode()
def infer_texts(model, tokenizer, texts, settings, device, *, embeddings=False):
    outputs = []
    batch_size = settings["batch_size"]
    for offset in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[offset:offset + batch_size],
            padding=True,
            truncation=True,
            max_length=settings["max_length"],
            return_tensors="pt",
        ).to(device)
        output = model(**tokens)
        if embeddings:
            result = F.normalize(output.last_hidden_state[:, 0], p=2, dim=1)
        else:
            result = output.logits.argmax(dim=-1)
        outputs.append(result.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def run_embeddings(split, config: dict, cache_dir: Path, head: str, device) -> list[dict]:
    settings = config["embeddings"]
    model_id = settings["model_id"]
    parameters = model_parameters(settings, cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **parameters)
    model = AutoModel.from_pretrained(model_id, **parameters).to(device)
    model.eval()
    classifier = make_classifier(head, config)

    def encode(texts):
        return infer_texts(model, tokenizer, texts, settings, device, embeddings=True)

    def predict(texts):
        return predict_labels(classifier, encode(texts))

    start = perf_counter()
    train_embeddings = encode(split.train["text"].tolist())
    classifier.fit(train_embeddings, split.train["target"])
    fit_seconds = perf_counter() - start
    val_texts = split.val["text"].tolist()
    prediction = predict(val_texts)
    return [{
        "stage": "embeddings",
        "representation": model_id,
        "preprocessing": "model_tokenizer",
        "classifier": head,
        "features": train_embeddings.shape[1],
        "device": str(device),
        "fit_seconds": fit_seconds,
        "inference_ms_per_tweet": benchmark_ms(predict, val_texts, config["benchmark"]),
        **classification_metrics(split.val["target"], prediction),
    }]


def training_batches(frame, tokenizer, settings, seed: int) -> DataLoader:
    tokens = tokenizer(
        frame["text"].tolist(), truncation=True, max_length=settings["max_length"]
    )
    examples = [
        {**{name: values[i] for name, values in tokens.items()}, "labels": label}
        for i, label in enumerate(frame["target"].astype(int))
    ]
    return DataLoader(
        examples,
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )


def run_finetune(split, config: dict, cache_dir: Path, device) -> tuple[list[dict], list[dict]]:
    settings = config["finetune"]
    set_seed(config["seed"])
    model_id = settings["model_id"]
    parameters = model_parameters(settings, cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **parameters)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, **parameters
    ).to(device)
    val_texts = split.val["text"].tolist()

    def predict(texts):
        return infer_texts(model, tokenizer, texts, settings, device)

    start = perf_counter()
    batches = training_batches(split.train, tokenizer, settings, config["seed"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    steps = len(batches) * settings["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(steps * settings["warmup_ratio"]), num_training_steps=steps
    )

    history = []
    best_f1 = -1.0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in batches:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * batch["labels"].size(0)

        model.eval()
        metrics = classification_metrics(split.val["target"], predict(val_texts))
        history.append({"epoch": epoch, "train_loss": total_loss / len(split.train), **metrics})
        print(
            f"Epoch {epoch}/{settings['epochs']}: "
            f"loss={history[-1]['train_loss']:.4f}, F1={metrics['f1']:.4f}",
            flush=True,
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    fit_seconds = perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    prediction = predict(val_texts)
    row = {
        "stage": "finetune",
        "representation": model_id,
        "preprocessing": "model_tokenizer",
        "classifier": "trainable_linear_head",
        "features": model.config.hidden_size,
        "device": str(device),
        "fit_seconds": fit_seconds,
        "best_epoch": best_epoch,
        "inference_ms_per_tweet": benchmark_ms(predict, val_texts, config["benchmark"]),
        **classification_metrics(split.val["target"], prediction),
    }
    return [row], history
