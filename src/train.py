"""Train the guide efficiency model.

    python -m src.train --config configs/base.yaml
    python -m src.train --config configs/base.yaml --csv data/doench2016.csv
    python -m src.train --config configs/base.yaml --no-features   # ablation
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from .data import (
    encode_dataset,
    load_csv,
    make_guide_dataset,
    split_indices,
)
from .metrics import evaluate, format_report
from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def predict(model, one_hot, features, indices, device, batch_size=256):
    model.eval()
    outputs = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        x = torch.from_numpy(one_hot[chunk]).float().to(device)
        f = torch.from_numpy(features[chunk]).float().to(device)
        outputs.append(model.predict_efficiency(x, f).cpu().numpy())
    return np.concatenate(outputs)


def baseline_scores(features, y, train_idx, test_idx, seed):
    """Ridge and gradient boosting on the biophysical features alone.

    Reported because on guide efficiency these baselines are genuinely
    competitive -- most of the signal is GC content, poly-T and a handful of
    position effects, all of which are in the feature vector. The network has
    to earn its place above them.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features[train_idx])
    X_train = scaler.transform(features[train_idx])
    X_test = scaler.transform(features[test_idx])

    results = {}
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X_train, y[train_idx])
    results["ridge_features"] = evaluate(y[test_idx], ridge.predict(X_test))

    boosted = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                        learning_rate=0.05, random_state=seed)
    boosted.fit(X_train, y[train_idx])
    results["gradient_boosting_features"] = evaluate(
        y[test_idx], boosted.predict(X_test)
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-features", action="store_true",
                        help="Ablate the biophysical branch.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.no_features:
        cfg["model"]["use_features"] = False
    if args.out_dir:
        cfg["out_dir"] = args.out_dir

    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  use_features={cfg['model']['use_features']}  "
          f"out_dir={out_dir}")

    if args.csv:
        guides, contexts, y = load_csv(args.csv)
    else:
        guides, contexts, y = make_guide_dataset(cfg["data"]["n_guides"],
                                                 cfg["seed"])
    print(f"{len(guides)} guides, efficiency {y.min():.3f}-{y.max():.3f} "
          f"(mean {y.mean():.3f})")

    from .sequence import has_polyt
    n_polyt = sum(has_polyt(g) for g in guides)
    print(f"{n_polyt} guides ({n_polyt/len(guides):.1%}) contain a poly-T "
          f"terminator and cannot work at all")

    one_hot, features = encode_dataset(guides, contexts)
    train_idx, val_idx, test_idx = split_indices(
        len(guides), cfg["data"]["val_fraction"], cfg["data"]["test_fraction"],
        cfg["seed"],
    )
    print(f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    model = build_model(cfg).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e3:.1f}k")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    total_steps = cfg["train"]["epochs"] * max(
        1, math.ceil(len(train_idx) / cfg["train"]["batch_size"])
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: cosine_with_warmup(s, cfg["train"]["warmup_steps"], total_steps),
    )
    # The target is a bounded rate, so BCE on the sigmoid output is the natural
    # loss -- it saturates correctly at 0 and 1, where MSE would keep pushing
    # logits to infinity.
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(cfg["seed"])

    history, best_spearman, patience, best_state = [], -2.0, 0, None
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        order = rng.permutation(train_idx)
        total, seen = 0.0, 0
        for start in tqdm(range(0, len(order), cfg["train"]["batch_size"]),
                          desc=f"epoch {epoch}", leave=False):
            chunk = order[start : start + cfg["train"]["batch_size"]]
            if len(chunk) < 2:      # BatchNorm needs more than one sample
                continue
            x = torch.from_numpy(one_hot[chunk]).float().to(device)
            f = torch.from_numpy(features[chunk]).float().to(device)
            target = torch.from_numpy(y[chunk]).float().to(device)

            loss = criterion(model(x, f), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg["train"]["grad_clip"])
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(chunk)
            seen += len(chunk)

        val = evaluate(y[val_idx],
                       predict(model, one_hot, features, val_idx, device))
        print(f"epoch {epoch:3d}  loss {total/max(1,seen):.4f}  "
              f"val_Spearman {val['spearman']:.4f}  "
              f"val_RMSE {val['rmse']:.4f}  "
              f"val_prec@5 {val['precision@5']:.3f}")
        history.append({"epoch": epoch, "loss": total / max(1, seen), "val": val})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2,
                                                         default=float))

        # Select on Spearman, since ranking is what the tool is used for.
        if val["spearman"] > best_spearman:
            best_spearman, patience = val["spearman"], 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["train"]["early_stopping_patience"]:
                print(f"early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(y[test_idx],
                    predict(model, one_hot, features, test_idx, device))
    print("\n" + format_report(test, "GuideEfficiencyNet"))

    baselines = baseline_scores(features, y, train_idx, test_idx, cfg["seed"])
    for name, result in baselines.items():
        print("\n" + format_report(result, name))

    best_baseline = max(baselines.values(), key=lambda r: r["spearman"])
    delta = test["spearman"] - best_baseline["spearman"]
    print(f"\nnetwork vs best feature baseline Spearman: {delta:+.4f} "
          f"({'network wins' if delta > 0 else 'baseline wins'})")
    print(f"\npicking the single top-ranked guide gives a true efficiency of "
          f"{test['top1_efficiency']:.3f}, against {test['mean_efficiency']:.3f} "
          f"for a guide chosen at random")

    torch.save({"model": model.state_dict(), "config": cfg},
               out_dir / "best.pt")
    (out_dir / "results.json").write_text(json.dumps(
        {"model": test, "baselines": baselines}, indent=2, default=float
    ))
    print(f"\nsaved {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
