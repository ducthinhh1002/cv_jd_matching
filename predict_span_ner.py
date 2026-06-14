from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from djinni_gemini_ner import split_text, stringify
from train_span_ner import (
    NO_ENTITY_LABEL,
    SpanBasedNerModel,
    SpanFeatureDataset,
    enumerate_candidate_spans,
    make_collate_fn,
    move_batch_to_device,
    require_transformers,
)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def read_jsonl(path: Path, max_documents: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
            if max_documents and len(rows) >= max_documents:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"{path} already exists. Pass --overwrite to replace it.")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def completed_record_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            record_id = stringify(row.get("record_id") or row.get("id"))
            if record_id:
                completed.add(record_id)
    return completed


def batched(rows: list[dict[str, Any]], batch_size: int) -> Sequence[list[dict[str, Any]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def choose_device(explicit_device: str | None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_prediction_features(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_chars_per_example: int,
    chunk_overlap_chars: int,
    max_length: int,
    max_span_width: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    features: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}

    for row in rows:
        record_id = stringify(row.get("record_id") or row.get("id"))
        text = stringify(row.get("text"))
        document_type = stringify(row.get("document_type") or row.get("doc_type") or "unknown")
        if not record_id or not text:
            continue

        for chunk_index, (chunk_start, chunk_text) in enumerate(
            split_text(text, max_chars_per_example, chunk_overlap_chars)
        ):
            encoded = tokenizer(
                chunk_text,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            offset_mapping = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
            span_starts, span_ends, span_widths, _ = enumerate_candidate_spans(
                offset_mapping=offset_mapping,
                gold_token_spans={},
                max_span_width=max_span_width,
            )
            if not span_starts:
                continue

            feature_id = f"{record_id}::chunk-{chunk_index}"
            features.append(
                {
                    "feature_id": feature_id,
                    "record_id": record_id,
                    "document_type": document_type,
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "span_starts": span_starts,
                    "span_ends": span_ends,
                    "span_widths": span_widths,
                    "labels": [0] * len(span_starts),
                }
            )
            metadata[feature_id] = {
                "record_id": record_id,
                "chunk_start": chunk_start,
                "chunk_text": chunk_text,
                "offset_mapping": offset_mapping,
                "span_starts": span_starts,
                "span_ends": span_ends,
            }
    return features, metadata


def overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return not (int(left["end"]) <= int(right["start"]) or int(left["start"]) >= int(right["end"]))


def select_entities(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_by_label: dict[str, list[dict[str, Any]]] = {}
    for candidate in sorted(candidates, key=lambda item: float(item["score"]), reverse=True):
        label = stringify(candidate.get("label"))
        kept = selected_by_label.setdefault(label, [])
        if any(overlaps(candidate, other) for other in kept):
            continue
        kept.append(candidate)

    selected = [entity for entities in selected_by_label.values() for entity in entities]
    return sorted(selected, key=lambda item: (int(item["start"]), int(item["end"]), stringify(item["label"])))


@torch.no_grad()
def predict_entities(
    *,
    model: SpanBasedNerModel,
    data_loader: DataLoader,
    metadata: dict[str, dict[str, Any]],
    id_to_label: dict[int, str],
    device: torch.device,
    threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    model.eval()
    by_record: dict[str, list[dict[str, Any]]] = {}

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            span_starts=batch["span_starts"],
            span_ends=batch["span_ends"],
            span_widths=batch["span_widths"],
        )
        probabilities = F.softmax(outputs["logits"], dim=-1)
        scores, predictions = probabilities.max(dim=-1)

        for row_index, feature_id in enumerate(batch["feature_ids"]):
            info = metadata[feature_id]
            record_id = info["record_id"]
            offset_mapping = info["offset_mapping"]
            chunk_start = int(info["chunk_start"])
            chunk_text = info["chunk_text"]
            span_starts = info["span_starts"]
            span_ends = info["span_ends"]

            for span_index, (token_start, token_end) in enumerate(zip(span_starts, span_ends, strict=False)):
                label_id = int(predictions[row_index, span_index].item())
                label = id_to_label.get(label_id, NO_ENTITY_LABEL)
                score = float(scores[row_index, span_index].item())
                if label == NO_ENTITY_LABEL or score < threshold:
                    continue
                char_start = int(offset_mapping[token_start][0])
                char_end = int(offset_mapping[token_end][1])
                if char_end <= char_start:
                    continue
                text = chunk_text[char_start:char_end]
                by_record.setdefault(record_id, []).append(
                    {
                        "label": label,
                        "text": text,
                        "start": chunk_start + char_start,
                        "end": chunk_start + char_end,
                        "normalized": text.lower().strip(),
                        "score": score,
                    }
                )

    return {record_id: select_entities(entities) for record_id, entities in by_record.items()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run span-based RoBERTa NER inference on JSONL documents.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL with record_id, document_type, text.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL with predicted entities.")
    parser.add_argument(
        "--checkpoint",
        default=Path("artifacts/span_ner_roberta_base_openai_cleaned_v6/span_ner.pt"),
        type=Path,
    )
    parser.add_argument("--threshold", type=float, default=0.65, help="Minimum entity probability.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--document-batch-size",
        type=int,
        default=256,
        help="Number of source documents to featurize, predict, and write per chunk.",
    )
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-chars-per-example", type=int, default=None)
    parser.add_argument("--chunk-overlap-chars", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-span-width", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append to an existing output and skip completed rows.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.output.exists() and not args.overwrite and not args.resume:
        raise SystemExit(f"{args.output} already exists. Pass --overwrite or --resume.")
    if args.overwrite and args.resume:
        raise SystemExit("--overwrite and --resume are mutually exclusive.")

    AutoModel, AutoTokenizer = require_transformers()
    del AutoModel

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = checkpoint.get("training_args") or {}
    model_name = checkpoint.get("model_name") or training_args.get("model_name") or "roberta-base"
    max_span_width = int(args.max_span_width or checkpoint.get("max_span_width") or 8)
    max_length = int(args.max_length or training_args.get("max_length") or 256)
    max_chars = int(args.max_chars_per_example or training_args.get("max_chars_per_example") or 1200)
    overlap = int(args.chunk_overlap_chars or training_args.get("chunk_overlap_chars") or 120)
    width_dim = int(checkpoint.get("width_embedding_dim") or training_args.get("width_embedding_dim") or 32)
    hidden_dim = int(checkpoint.get("classifier_hidden_dim") or training_args.get("classifier_hidden_dim") or 256)

    label_to_id = {str(label): int(index) for label, index in checkpoint["label_to_id"].items()}
    id_to_label = {int(index): str(label) for index, label in checkpoint["id_to_label"].items()}

    tokenizer_dir = args.checkpoint.parent
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if tokenizer_dir.exists() else model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = SpanBasedNerModel(
        model_name=model_name,
        num_labels=len(label_to_id),
        max_span_width=max_span_width,
        width_embedding_dim=width_dim,
        classifier_hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    rows = read_jsonl(args.input, max_documents=args.max_documents)
    completed_ids = completed_record_ids(args.output) if args.resume else set()
    if completed_ids:
        rows = [row for row in rows if stringify(row.get("record_id") or row.get("id")) not in completed_ids]
        logging.info("Resume mode: skipping %s already completed documents.", len(completed_ids))

    if args.overwrite and args.output.exists():
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_documents = 0
    total_features = 0
    total_entities = 0
    output_mode = "a" if args.resume else "w"
    with args.output.open(output_mode, encoding="utf-8") as handle:
        for batch_index, row_batch in enumerate(batched(rows, args.document_batch_size), start=1):
            features, metadata = build_prediction_features(
                row_batch,
                tokenizer,
                max_chars_per_example=max_chars,
                chunk_overlap_chars=overlap,
                max_length=max_length,
                max_span_width=max_span_width,
            )
            logging.info(
                "Prepared batch %s: %s prediction features from %s documents.",
                batch_index,
                len(features),
                len(row_batch),
            )

            entities_by_record: dict[str, list[dict[str, Any]]] = {}
            if features:
                data_loader = DataLoader(
                    SpanFeatureDataset(features),
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=make_collate_fn(tokenizer.pad_token_id),
                )
                entities_by_record = predict_entities(
                    model=model,
                    data_loader=data_loader,
                    metadata=metadata,
                    id_to_label=id_to_label,
                    device=device,
                    threshold=args.threshold,
                )

            for row in row_batch:
                record_id = stringify(row.get("record_id") or row.get("id"))
                row["entities"] = entities_by_record.get(record_id, [])
                row["ner_checkpoint"] = str(args.checkpoint)
                row["ner_threshold"] = args.threshold
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_documents += 1
                total_entities += len(row["entities"])
            handle.flush()
            total_features += len(features)
            logging.info(
                "Wrote %s/%s pending documents (%s features, %s entities so far).",
                total_documents,
                len(rows),
                total_features,
                total_entities,
            )

    logging.info("Wrote %s documents with %s predicted entities to %s.", total_documents, total_entities, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
