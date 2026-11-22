import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .classic import run_classic
from .data import make_split

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("classic", "embeddings", "finetune")


def main() -> None:
    parser = argparse.ArgumentParser(description="Disaster Tweets classification experiments")
    parser.add_argument("stage", nargs="?", default="all", choices=("all", *STAGES))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.device:
        config["device"] = args.device

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train_path = ROOT / config["data"]["train_csv"]
    if not train_path.is_file():
        parser.error(f"Put the labeled Kaggle CSV at {train_path}.")
    split = make_split(
        pd.read_csv(train_path),
        val_size=config["data"].get("val_size", 0.2),
        seed=config["seed"],
        group_folds=config["data"].get("validation_group_folds"),
    )
    print(f"Train: {len(split.train)}  Validation: {len(split.val)}", flush=True)

    run_info = {
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "val_ids_sha256": hashlib.sha256(
            split.val["id"].to_csv(index=False, lineterminator="\n").encode()
        ).hexdigest(),
        "train_size": len(split.train),
        "val_size": len(split.val),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "nltk", "torch", "transformers")
        },
        "config": config,
    }
    run_hash = hashlib.sha256(json.dumps(run_info, sort_keys=True).encode()).hexdigest()[:16]
    output_dir = ROOT / config["results_dir"]
    stage_dir = output_dir / ".runs" / run_hash
    stage_dir.mkdir(parents=True, exist_ok=True)
    requested_stages = STAGES if args.stage == "all" else (args.stage,)

    with threadpool_limits(limits=config["benchmark"]["cpu_threads"]):
        for stage in requested_stages:
            print(f"\nRunning {stage}", flush=True)
            if stage == "classic":
                rows = run_classic(split, config)
            else:
                import torch

                from .transformer import run_embeddings, run_finetune, select_device

                torch.set_num_threads(config["benchmark"]["cpu_threads"])
                device = select_device(config["device"])
                cache_dir = ROOT / config["model_cache"] / "hub"
                if stage == "embeddings":
                    classic_result = stage_dir / "classic.csv"
                    if not classic_result.is_file():
                        parser.error("Run classic first with the same data, config, and device option.")
                    ranking = pd.read_csv(classic_result).sort_values("f1", ascending=False, kind="stable")
                    head = ranking.iloc[0]["classifier"]
                    print(f"Selected classifier: {head} ({device})", flush=True)
                    rows = run_embeddings(split, config, cache_dir, head, device)
                else:
                    rows, history = run_finetune(split, config, cache_dir, device)
                    pd.DataFrame(history).to_csv(stage_dir / "finetune_history.csv", index=False)

            frame = pd.DataFrame(rows)
            frame.to_csv(stage_dir / f"{stage}.csv", index=False)
            best = frame.loc[frame["f1"].idxmax()]
            print(f"Best {stage} F1: {best['f1']:.4f}", flush=True)

    completed = [stage for stage in STAGES if (stage_dir / f"{stage}.csv").exists()]
    results = pd.concat([pd.read_csv(stage_dir / f"{stage}.csv") for stage in completed])
    results.to_csv(output_dir / "metrics.csv", index=False, float_format="%.8f")
    history_path = stage_dir / "finetune_history.csv"
    if history_path.is_file():
        pd.read_csv(history_path).to_csv(
            output_dir / "finetune_history.csv", index=False, float_format="%.8f"
        )
    run_info["stages"] = completed
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Metrics: {output_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
