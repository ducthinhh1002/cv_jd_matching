from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


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


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"{path} already exists. Pass --overwrite.")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split ranking pairs into train/valid/test JSONL files.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--valid-output", required=True, type=Path)
    parser.add_argument("--test-output", required=True, type=Path)
    parser.add_argument("--group-by", choices=("jd_id", "cv_id"), default="jd_id")
    parser.add_argument("--valid-groups", type=int, default=None)
    parser.add_argument("--test-groups", type=int, default=None)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    rows = read_jsonl(args.input)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = stringify(row.get(args.group_by))
        if group_id:
            by_group[group_id].append(row)

    groups = sorted(by_group)
    rng = random.Random(args.seed)
    rng.shuffle(groups)

    group_count = len(groups)
    test_count = args.test_groups if args.test_groups is not None else max(1, round(group_count * args.test_fraction))
    valid_count = args.valid_groups if args.valid_groups is not None else max(1, round(group_count * args.valid_fraction))
    if test_count + valid_count >= group_count:
        raise SystemExit(
            f"Invalid split: {group_count} groups, valid={valid_count}, test={test_count}. "
            "Need at least one train group."
        )

    test_groups = set(groups[:test_count])
    valid_groups = set(groups[test_count : test_count + valid_count])
    train_groups = set(groups[test_count + valid_count :])

    train_rows = [row for group in sorted(train_groups) for row in by_group[group]]
    valid_rows = [row for group in sorted(valid_groups) for row in by_group[group]]
    test_rows = [row for group in sorted(test_groups) for row in by_group[group]]

    write_jsonl(args.train_output, train_rows, args.overwrite)
    write_jsonl(args.valid_output, valid_rows, args.overwrite)
    write_jsonl(args.test_output, test_rows, args.overwrite)

    logging.info(
        "Split %s rows across %s %s groups -> train=%s rows/%s groups, valid=%s rows/%s groups, test=%s rows/%s groups.",
        len(rows),
        group_count,
        args.group_by,
        len(train_rows),
        len(train_groups),
        len(valid_rows),
        len(valid_groups),
        len(test_rows),
        len(test_groups),
    )
    logging.info("Valid groups: %s", sorted(valid_groups))
    logging.info("Test groups: %s", sorted(test_groups))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
