"""Evaluate Stage 3 ranking output against a labelled synthetic dataset.

Metrics computed:
  - Precision@K   — fraction of top-K that are relevant (label >= threshold)
  - NDCG@K        — normalised discounted cumulative gain
  - MAP           — mean average precision over all JDs
  - Spearman's ρ  — rank correlation between predicted score and ground-truth label

The evaluation expects two inputs:
  1. --predictions  Output of score_candidates.py (JSONL with cv_id, jd_id, score, relevance_label)
  2. --ground-truth  Output of generate_synthetic_eval.py (JSONL with pair_id, cv_id, jd_id, relevance_label)

If --predictions already contains relevance_label (because score_candidates.py was run with
--pairs-input synthetic_eval_dataset.jsonl), --ground-truth can be omitted.

Usage:
  python evaluate_ranking.py \\
    --predictions .\\artifacts\\stage3_scores_eval.jsonl \\
    --output .\\artifacts\\eval_results.json

  python evaluate_ranking.py \\
    --predictions .\\artifacts\\stage3_scores_eval.jsonl \\
    --ground-truth .\\artifacts\\synthetic_eval_dataset.jsonl \\
    --output .\\artifacts\\eval_results.json \\
    --k 5 --relevant-threshold 4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 3 ranking output: P@K, NDCG@K, MAP, Spearman's ρ."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="JSONL output from score_candidates.py (must contain cv_id, jd_id, score).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help=(
            "Optional JSONL from generate_synthetic_eval.py. "
            "If omitted, relevance_label must already be in --predictions."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="JSON file to write evaluation results to.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Cut-off for Precision@K and NDCG@K (default 5).",
    )
    parser.add_argument(
        "--relevant-threshold",
        type=int,
        default=4,
        help="Minimum relevance_label (1-5) to count as relevant for P@K (default 4).",
    )
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help=(
            "Also compute BM25-style (random baseline) and "
            "pure-semantic metrics for comparison."
        ),
    )
    parser.add_argument(
        "--group-by",
        choices=("jd_id", "cv_id"),
        default="jd_id",
        help="Query/group id used for ranking evaluation. Default ranks CVs per JD.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def build_ground_truth(gt_rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    """Map (cv_id, jd_id) → relevance_label."""
    lookup: dict[tuple[str, str], int] = {}
    for row in gt_rows:
        cv_id = stringify(row.get("cv_id"))
        jd_id = stringify(row.get("jd_id"))
        label = row.get("relevance_label")
        if cv_id and jd_id and isinstance(label, (int, float)):
            lookup[(cv_id, jd_id)] = int(label)
    return lookup


def attach_labels(
    predictions: list[dict[str, Any]],
    ground_truth: dict[tuple[str, str], int] | None,
) -> list[dict[str, Any]]:
    """Add/override relevance_label from ground_truth if provided."""
    result: list[dict[str, Any]] = []
    missing = 0
    for pred in predictions:
        cv_id = stringify(pred.get("cv_id"))
        jd_id = stringify(pred.get("jd_id"))
        if ground_truth is not None:
            label = ground_truth.get((cv_id, jd_id))
            if label is None:
                missing += 1
                continue
            pred = {**pred, "relevance_label": label}
        elif "relevance_label" not in pred:
            missing += 1
            continue
        result.append(pred)
    if missing:
        logging.warning("%d predictions had no ground-truth label and were skipped.", missing)
    return result


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def precision_at_k(ranked_labels: list[int], k: int, threshold: int) -> float:
    """Fraction of top-K items with label >= threshold."""
    top_k = ranked_labels[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for lbl in top_k if lbl >= threshold)
    return relevant / len(top_k)


def dcg_at_k(ranked_labels: list[int], k: int) -> float:
    """Discounted Cumulative Gain using (2^rel - 1) / log2(rank+1)."""
    dcg = 0.0
    for rank, label in enumerate(ranked_labels[:k], start=1):
        dcg += (2 ** label - 1) / math.log2(rank + 1)
    return dcg


def ndcg_at_k(ranked_labels: list[int], ideal_labels: list[int], k: int) -> float:
    """Normalised DCG@K."""
    ideal_dcg = dcg_at_k(sorted(ideal_labels, reverse=True), k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked_labels, k) / ideal_dcg


def average_precision(ranked_labels: list[int], threshold: int) -> float:
    """Average Precision: area under the precision-recall curve."""
    relevant_count = 0
    precision_sum = 0.0
    total_relevant = sum(1 for lbl in ranked_labels if lbl >= threshold)
    if total_relevant == 0:
        return 0.0
    for rank, label in enumerate(ranked_labels, start=1):
        if label >= threshold:
            relevant_count += 1
            precision_sum += relevant_count / rank
    return precision_sum / total_relevant


def spearman_rho(scores: list[float], labels: list[int]) -> float:
    """Spearman's rank correlation coefficient."""
    n = len(scores)
    if n < 2:
        return 0.0

    def rank_list(values: list[float]) -> list[float]:
        indexed = sorted(range(n), key=lambda i: values[i], reverse=True)
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and values[indexed[j]] == values[indexed[i]]:
                j += 1
            avg_rank = (i + j + 1) / 2.0
            for k_idx in range(i, j):
                ranks[indexed[k_idx]] = avg_rank
            i = j
        return ranks

    score_ranks = rank_list(scores)
    label_ranks = rank_list([float(lbl) for lbl in labels])

    d2_sum = sum((sr - lr) ** 2 for sr, lr in zip(score_ranks, label_ranks))
    return 1.0 - (6 * d2_sum) / (n * (n ** 2 - 1))


# ---------------------------------------------------------------------------
# Per-system evaluation
# ---------------------------------------------------------------------------

def evaluate_system(
    predictions_by_group: dict[str, list[dict[str, Any]]],
    k: int,
    threshold: int,
    score_key: str = "score",
) -> dict[str, Any]:
    """Evaluate one scoring system identified by `score_key`."""
    precision_scores: list[float] = []
    ndcg_scores: list[float] = []
    ap_scores: list[float] = []
    all_pred_scores: list[float] = []
    all_true_labels: list[int] = []

    for group_id, preds in predictions_by_group.items():
        # Sort by the chosen score descending
        ranked = sorted(preds, key=lambda p: p.get(score_key, 0.0), reverse=True)
        ranked_labels = [int(p["relevance_label"]) for p in ranked]
        ideal_labels = sorted(ranked_labels, reverse=True)
        ranked_scores = [float(p.get(score_key, 0.0)) for p in ranked]

        precision_scores.append(precision_at_k(ranked_labels, k, threshold))
        ndcg_scores.append(ndcg_at_k(ranked_labels, ideal_labels, k))
        ap_scores.append(average_precision(ranked_labels, threshold))
        all_pred_scores.extend(ranked_scores)
        all_true_labels.extend(ranked_labels)

    rho = spearman_rho(all_pred_scores, all_true_labels)

    return {
        f"precision_at_{k}": round(statistics.mean(precision_scores), 4) if precision_scores else 0.0,
        f"ndcg_at_{k}": round(statistics.mean(ndcg_scores), 4) if ndcg_scores else 0.0,
        "map": round(statistics.mean(ap_scores), 4) if ap_scores else 0.0,
        "spearman_rho": round(rho, 4),
        "num_groups_evaluated": len(predictions_by_group),
        "num_jds_evaluated": len(predictions_by_group),
        "num_pairs_evaluated": sum(len(v) for v in predictions_by_group.values()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if not args.predictions.exists():
        raise SystemExit(f"Predictions file not found: {args.predictions}")

    logging.info("Loading predictions from %s", args.predictions)
    predictions = load_jsonl(args.predictions)
    logging.info("Loaded %d prediction rows.", len(predictions))

    ground_truth: dict[tuple[str, str], int] | None = None
    if args.ground_truth:
        if not args.ground_truth.exists():
            raise SystemExit(f"Ground-truth file not found: {args.ground_truth}")
        logging.info("Loading ground truth from %s", args.ground_truth)
        gt_rows = load_jsonl(args.ground_truth)
        ground_truth = build_ground_truth(gt_rows)
        logging.info("Loaded %d ground-truth labels.", len(ground_truth))

    labelled = attach_labels(predictions, ground_truth)
    logging.info("%d labelled prediction rows ready for evaluation.", len(labelled))

    if not labelled:
        raise SystemExit(
            "No labelled predictions available. "
            "Either include relevance_label in predictions or provide --ground-truth."
        )

    predictions_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in labelled:
        group_id = stringify(pred.get(args.group_by))
        if not group_id:
            logging.warning("Skipping prediction without %s: %s", args.group_by, pred.get("pair_id", ""))
            continue
        predictions_by_group[group_id].append(pred)

    logging.info("Evaluating across %d groups by %s.", len(predictions_by_group), args.group_by)

    # --- Proposed system ---
    proposed_results = evaluate_system(predictions_by_group, k=args.k, threshold=args.relevant_threshold)

    output: dict[str, Any] = {
        "k": args.k,
        "relevant_threshold": args.relevant_threshold,
        "group_by": args.group_by,
        "proposed_system": proposed_results,
    }

    # --- Optional baselines using alternative score fields ---
    if args.compare_baselines:
        # Semantic-only baseline (uses semantic_score field)
        has_semantic = any("semantic_score" in p for p in labelled)
        if has_semantic:
            semantic_results = evaluate_system(
                predictions_by_group, k=args.k, threshold=args.relevant_threshold, score_key="semantic_score"
            )
            output["pure_semantic_baseline"] = semantic_results

        # O*NET-only baseline
        has_onet = any("onet_score" in p for p in labelled)
        if has_onet:
            onet_results = evaluate_system(
                predictions_by_group, k=args.k, threshold=args.relevant_threshold, score_key="onet_score"
            )
            output["pure_onet_baseline"] = onet_results

        # Random baseline (hard constraint only as a proxy for keyword approach)
        has_hard = any("hard_constraint_score" in p for p in labelled)
        if has_hard:
            hard_results = evaluate_system(
                predictions_by_group, k=args.k, threshold=args.relevant_threshold, score_key="hard_constraint_score"
            )
            output["hard_constraint_baseline"] = hard_results

    # --- Per-group breakdown ---
    per_group_details: list[dict[str, Any]] = []
    for group_id, preds in sorted(predictions_by_group.items()):
        ranked = sorted(preds, key=lambda p: p.get("score", 0.0), reverse=True)
        ranked_labels = [int(p["relevance_label"]) for p in ranked]
        ideal_labels = sorted(ranked_labels, reverse=True)
        ranked_scores = [float(p.get("score", 0.0)) for p in ranked]
        per_group_details.append({
            "group_by": args.group_by,
            "group_id": group_id,
            args.group_by: group_id,
            f"precision_at_{args.k}": round(precision_at_k(ranked_labels, args.k, args.relevant_threshold), 4),
            f"ndcg_at_{args.k}": round(ndcg_at_k(ranked_labels, ideal_labels, args.k), 4),
            "average_precision": round(average_precision(ranked_labels, args.relevant_threshold), 4),
            "spearman_rho": round(spearman_rho(ranked_scores, ranked_labels), 4),
            "ranked_labels": ranked_labels,
            "num_relevant": sum(1 for lbl in ranked_labels if lbl >= args.relevant_threshold),
        })
    output["per_group"] = per_group_details
    output["per_jd"] = per_group_details

    # --- Save ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Pretty print summary ---
    p = proposed_results
    logging.info("=" * 60)
    logging.info("Evaluation Results (k=%d, relevant≥%d)", args.k, args.relevant_threshold)
    logging.info("=" * 60)
    logging.info("  Proposed System")
    logging.info("    Precision@%d  : %.4f", args.k, p[f"precision_at_{args.k}"])
    logging.info("    NDCG@%d       : %.4f", args.k, p[f"ndcg_at_{args.k}"])
    logging.info("    MAP           : %.4f", p["map"])
    logging.info("    Spearman ρ    : %.4f", p["spearman_rho"])
    logging.info("    Groups evaluated : %d", p["num_groups_evaluated"])

    if args.compare_baselines:
        for baseline_key in ("pure_semantic_baseline", "pure_onet_baseline", "hard_constraint_baseline"):
            if baseline_key in output:
                b = output[baseline_key]
                logging.info("  %s", baseline_key)
                logging.info("    Precision@%d  : %.4f", args.k, b[f"precision_at_{args.k}"])
                logging.info("    NDCG@%d       : %.4f", args.k, b[f"ndcg_at_{args.k}"])
                logging.info("    MAP           : %.4f", b["map"])
                logging.info("    Spearman ρ    : %.4f", b["spearman_rho"])

    logging.info("Results saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
