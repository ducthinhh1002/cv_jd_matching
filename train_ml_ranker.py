from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xgboost as xgb

from score_candidates import BASIC_ML_FEATURES, IT_ML_FEATURES


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def feature_names(feature_set: str) -> tuple[str, ...]:
    return BASIC_ML_FEATURES if feature_set == "basic" else IT_ML_FEATURES


def build_matrix(
    rows: list[dict[str, Any]],
    *,
    features: tuple[str, ...],
    group_by: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    usable = [row for row in rows if "relevance_label" in row and stringify(row.get(group_by))]
    if not usable:
        raise SystemExit("No labelled rows with group id found.")

    X = np.array([[float(row.get(name, 0.0)) for name in features] for row in usable], dtype=np.float32)
    y = np.array([float(row.get("relevance_label", 1)) for row in usable], dtype=np.float32)
    groups = np.array([stringify(row.get(group_by)) for row in usable])
    return X, y, groups


def sort_by_group(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(groups)
    X_sorted = X[order]
    y_sorted = y[order]
    groups_sorted = groups[order]
    _, group_sizes = np.unique(groups_sorted, return_counts=True)
    return X_sorted, y_sorted, groups_sorted, group_sizes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost learning-to-rank model from scored pairs.")
    parser.add_argument("--train-scores", required=True, type=Path)
    parser.add_argument("--valid-scores", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--feature-set", choices=("basic", "it"), default="it")
    parser.add_argument("--group-by", choices=("jd_id", "cv_id"), default="jd_id")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    if not args.train_scores.exists():
        raise SystemExit(f"Train scores not found: {args.train_scores}")
    if args.valid_scores and not args.valid_scores.exists():
        raise SystemExit(f"Valid scores not found: {args.valid_scores}")

    features = feature_names(args.feature_set)
    train_rows = read_jsonl(args.train_scores)
    X_train, y_train, groups_train = build_matrix(train_rows, features=features, group_by=args.group_by)
    X_train, y_train, groups_train, train_group_sizes = sort_by_group(X_train, y_train, groups_train)
    logging.info(
        "Loaded train: %s rows, %s groups, features=%s.",
        len(X_train),
        len(train_group_sizes),
        ", ".join(features),
    )

    eval_set = None
    eval_group = None
    valid_summary: dict[str, Any] | None = None
    if args.valid_scores:
        valid_rows = read_jsonl(args.valid_scores)
        X_valid, y_valid, groups_valid = build_matrix(valid_rows, features=features, group_by=args.group_by)
        X_valid, y_valid, groups_valid, valid_group_sizes = sort_by_group(X_valid, y_valid, groups_valid)
        eval_set = [(X_valid, y_valid)]
        eval_group = [valid_group_sizes]
        valid_summary = {
            "rows": int(len(X_valid)),
            "groups": int(len(valid_group_sizes)),
            "group_ids": sorted(set(groups_valid.tolist())),
        }
        logging.info("Loaded valid: %s rows, %s groups.", len(X_valid), len(valid_group_sizes))

    model = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric=["ndcg@5", "ndcg@10"],
        tree_method="hist",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.seed,
    )

    logging.info("Training XGBRanker...")
    fit_kwargs: dict[str, Any] = {"group": train_group_sizes, "verbose": False}
    if eval_set is not None and eval_group is not None:
        fit_kwargs["eval_set"] = eval_set
        fit_kwargs["eval_group"] = eval_group
    model.fit(X_train, y_train, **fit_kwargs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output)
    logging.info("Model saved to %s", args.output)

    importances = model.feature_importances_.tolist()
    metadata = {
        "feature_set": args.feature_set,
        "features": list(features),
        "group_by": args.group_by,
        "train": {
            "rows": int(len(X_train)),
            "groups": int(len(train_group_sizes)),
            "group_ids": sorted(set(groups_train.tolist())),
        },
        "valid": valid_summary,
        "params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "seed": args.seed,
        },
        "feature_importances": {
            name: float(value) for name, value in zip(features, importances, strict=False)
        },
    }

    metadata_output = args.metadata_output or args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Metadata saved to %s", metadata_output)
    for name, value in metadata["feature_importances"].items():
        logging.info("  %-24s %.4f", name, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
