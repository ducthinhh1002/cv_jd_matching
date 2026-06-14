from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import random
import re
import zipfile
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree


DEGREE_KEYWORDS = (
    "phd",
    "doctorate",
    "master",
    "mba",
    "bachelor",
    "bs",
    "ba",
    "b.sc",
    "m.sc",
    "degree",
)

CERT_KEYWORDS = (
    "aws",
    "azure",
    "gcp",
    "cissp",
    "security+",
    "pmp",
    "scrum master",
    "oracle certified",
    "microsoft certified",
    "ccna",
    "cka",
)

KAGGLE_RESUME_COLUMNS = {
    "career_objective",
    "skills",
    "educational_institution_name",
    "degree_names",
    "passing_years",
    "major_field_of_studies",
    "educational_results",
    "professional_company_names",
    "positions",
    "responsibilities",
    "certification_providers",
    "certification_skills",
    "issue_dates",
    "expiry_dates",
    "languages",
    "proficiency_levels",
    "extra_curricular_activity_types",
    "extra_curricular_organization_names",
    "role_positions",
}

KAGGLE_JOB_COLUMNS = {
    "job_position_name",
    "educational_requirements",
    "educationaL_requirements",
    "experiencere_requirement",
    "experience_requirement",
    "age_requirement",
    "skills_required",
    "responsibilities.1",
    "job_responsibilities",
}

KAGGLE_IT_JOB_TITLE_PATTERN = re.compile(
    r"\b("
    r"software|developer|full\s*stack|front[-\s]?end|back[-\s]?end|"
    r"ios|android|devops|data\s+(?:engineer|science|scientist|analyst)|"
    r"machine\s+learning|\bml\b|ai\s+engineer|generative\s+ai|"
    r"database\s+administrator|\bdba\b|system\s+administrator|"
    r"network\s+support|information\s+technology|\bit\b"
    r")\b",
    re.IGNORECASE,
)


def stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def canonical_header(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip()).lower()


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"{path} already exists. Pass --overwrite to replace it.")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_docx_text(path: Path) -> str:
    """Extract plain text from a .docx using only stdlib zip/xml modules."""
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def infer_qualification_facts(text: str) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    lowered = text.lower()

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", lowered):
        facts.append(
            {
                "fact_type": "EXPERIENCE_YEARS",
                "text": text[match.start() : match.end()],
                "start": match.start(),
                "end": match.end(),
                "operator": ">=" if "+" in match.group(0) else "=",
                "value": match.group(1),
                "unit": "years",
                "normalized": match.group(1),
                "is_mandatory": False,
            }
        )

    for keyword in DEGREE_KEYWORDS:
        start = lowered.find(keyword)
        if start >= 0:
            end = start + len(keyword)
            facts.append(
                {
                    "fact_type": "DEGREE",
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "value": keyword,
                    "normalized": keyword,
                    "is_mandatory": False,
                }
            )

    for keyword in CERT_KEYWORDS:
        start = lowered.find(keyword)
        if start >= 0:
            end = start + len(keyword)
            facts.append(
                {
                    "fact_type": "CERTIFICATION",
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "value": keyword,
                    "normalized": keyword,
                    "is_mandatory": False,
                }
            )
    return facts


def parse_vanetik_rankings(path: Path) -> tuple[list[list[int]], list[list[int]]]:
    text = path.read_text(encoding="utf-8")
    cleaned = re.sub(r"#.*", "", text)
    arrays: list[list[list[int]]] = []
    for name in ("ANNOTATOR_1_RANKINGS", "ANNOTATOR_2_RANKINGS"):
        match = re.search(rf"{name}\s*=\s*(\[[\s\S]*?\])\s*(?:ANNOTATOR_|$)", cleaned)
        if not match:
            raise SystemExit(f"Could not parse {name} from {path}")
        raw = match.group(1).strip()
        # Regex above stops at the first closing bracket if not expanded; balance it manually.
        start = cleaned.find("[", cleaned.find(name))
        depth = 0
        end = None
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            raise SystemExit(f"Unbalanced ranking array for {name}")
        arrays.append(ast.literal_eval(cleaned[start:end]))
    return arrays[0], arrays[1]


def ranking_to_scores(ranked_vacancies: list[int], vacancy_count: int = 5) -> dict[int, float]:
    scores = {vacancy_id: 1.0 for vacancy_id in range(1, vacancy_count + 1)}
    seen: set[int] = set()
    for position, vacancy_id in enumerate(ranked_vacancies, start=1):
        if vacancy_id in seen or vacancy_id < 1 or vacancy_id > vacancy_count:
            continue
        seen.add(vacancy_id)
        scores[vacancy_id] = float(vacancy_count + 1 - position)
    return scores


def prepare_vanetik(args: argparse.Namespace) -> int:
    source_dir = args.source_dir
    vacancies_path = source_dir / "5_vacancies.csv"
    annotations_path = source_dir / "annotations-for-the-first-30-vacancies.txt"
    cv_dir = source_dir / "CV"

    if not vacancies_path.exists() or not annotations_path.exists() or not cv_dir.exists():
        raise SystemExit(f"Vanetik dataset folder is incomplete: {source_dir}")

    docs: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    with vacancies_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        vacancies = list(reader)

    for index, row in enumerate(vacancies, start=1):
        text = stringify(row.get("job_description"))
        title = stringify(row.get("job_title"))
        full_text = f"Position: {title}\n\nJob Description:\n{text}".strip()
        docs.append(
            {
                "record_id": f"vanetik_jd_{index}",
                "document_type": "jd",
                "source_dataset": "vanetik_vacancy_resume_matching",
                "source_row_index": index,
                "text": full_text,
                "qualification_facts": infer_qualification_facts(full_text),
            }
        )

    annotator_1, annotator_2 = parse_vanetik_rankings(annotations_path)
    max_cv = min(30, len(annotator_1), len(annotator_2))
    for cv_index in range(1, max_cv + 1):
        cv_path = cv_dir / f"{cv_index}.docx"
        if not cv_path.exists():
            logging.warning("Missing CV file: %s", cv_path)
            continue
        cv_text = extract_docx_text(cv_path)
        cv_id = f"vanetik_cv_{cv_index}"
        docs.append(
            {
                "record_id": cv_id,
                "document_type": "cv",
                "source_dataset": "vanetik_vacancy_resume_matching",
                "source_row_index": cv_index,
                "text": cv_text,
                "qualification_facts": infer_qualification_facts(cv_text),
            }
        )

        scores_1 = ranking_to_scores(annotator_1[cv_index - 1], vacancy_count=len(vacancies))
        scores_2 = ranking_to_scores(annotator_2[cv_index - 1], vacancy_count=len(vacancies))
        for vacancy_index in range(1, len(vacancies) + 1):
            avg_score = (scores_1[vacancy_index] + scores_2[vacancy_index]) / 2.0
            pairs.append(
                {
                    "pair_id": f"vanetik_cv_{cv_index}_jd_{vacancy_index}",
                    "cv_id": cv_id,
                    "jd_id": f"vanetik_jd_{vacancy_index}",
                    "relevance_label": max(1, min(5, int(round(avg_score)))),
                    "human_score": avg_score,
                    "annotator_1_score": scores_1[vacancy_index],
                    "annotator_2_score": scores_2[vacancy_index],
                    "source_dataset": "vanetik_vacancy_resume_matching",
                }
            )

    write_jsonl(args.docs_output, docs, args.overwrite)
    write_jsonl(args.pairs_output, pairs, args.overwrite)
    logging.info("Wrote %s docs and %s pairs.", len(docs), len(pairs))
    return 0


def normalize_score_to_label(value: Any) -> int:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 1
    if score > 1.0:
        score = score / 100.0
    score = max(0.0, min(1.0, score))
    return max(1, min(5, int(round(1 + 4 * score))))


def join_fields(row: dict[str, str], columns: list[str]) -> str:
    sections: list[str] = []
    for column in columns:
        value = stringify(row.get(column)).strip()
        if not value or value.lower() in {"nan", "none", "[]"}:
            continue
        sections.append(f"{column}: {value}")
    return "\n".join(sections)


def matches_optional_regex(value: str, pattern: re.Pattern[str] | None) -> bool:
    if pattern is None:
        return True
    return bool(pattern.search(value))


def prepare_kaggle(args: argparse.Namespace) -> int:
    if not args.input_csv.exists():
        raise SystemExit(f"Kaggle CSV not found: {args.input_csv}")

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        original_fields = reader.fieldnames or []
        rows = [{canonical_header(key): value for key, value in row.items()} for row in reader]

    fields = [canonical_header(field) for field in original_fields]
    score_col = next((field for field in fields if field.lower() == "matched_score"), None)
    if not score_col:
        raise SystemExit("Kaggle CSV must contain a matched_score column.")

    resume_columns = [field for field in fields if field in {canonical_header(c) for c in KAGGLE_RESUME_COLUMNS}]
    job_columns = [field for field in fields if field in {canonical_header(c) for c in KAGGLE_JOB_COLUMNS}]
    if not resume_columns or not job_columns:
        raise SystemExit(
            "Could not detect resume/job columns in Kaggle CSV. "
            f"Detected resume={resume_columns}, job={job_columns}"
        )

    docs_by_id: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    job_title_pattern = re.compile(args.job_title_regex, re.IGNORECASE) if args.job_title_regex else None
    filtered_out = 0

    for index, row in enumerate(rows, start=1):
        job_title = stringify(row.get("job_position_name")).strip()
        if args.it_only and not KAGGLE_IT_JOB_TITLE_PATTERN.search(job_title):
            filtered_out += 1
            continue
        if not matches_optional_regex(job_title, job_title_pattern):
            filtered_out += 1
            continue

        resume_text = join_fields(row, resume_columns)
        job_text = join_fields(row, job_columns)
        if not resume_text or not job_text:
            continue

        cv_id = stable_id("kaggle_cv", resume_text)
        jd_id = stable_id("kaggle_jd", job_text)
        docs_by_id.setdefault(
            cv_id,
            {
                "record_id": cv_id,
                "document_type": "cv",
                "source_dataset": "kaggle_resume_data_for_ranking",
                "text": resume_text,
                "qualification_facts": infer_qualification_facts(resume_text),
            },
        )
        docs_by_id.setdefault(
            jd_id,
            {
                "record_id": jd_id,
                "document_type": "jd",
                "source_dataset": "kaggle_resume_data_for_ranking",
                "text": job_text,
                "source_job_title": job_title,
                "qualification_facts": infer_qualification_facts(job_text),
            },
        )
        matched_score = row.get(score_col)
        pairs.append(
            {
                "pair_id": f"kaggle_pair_{index}",
                "cv_id": cv_id,
                "jd_id": jd_id,
                "matched_score": float(matched_score) if stringify(matched_score) else None,
                "relevance_label": normalize_score_to_label(matched_score),
                "source_job_title": job_title,
                "source_dataset": "kaggle_resume_data_for_ranking",
            }
        )

    if filtered_out:
        logging.info("Filtered out %s Kaggle rows by job title.", filtered_out)

    if args.max_pairs_per_jd:
        rng = random.Random(args.sample_seed)
        pairs_by_jd: dict[str, list[dict[str, Any]]] = {}
        for pair in pairs:
            pairs_by_jd.setdefault(pair["jd_id"], []).append(pair)

        sampled_pairs: list[dict[str, Any]] = []
        for jd_id in sorted(pairs_by_jd):
            candidates = pairs_by_jd[jd_id]
            rng.shuffle(candidates)
            sampled_pairs.extend(candidates[: args.max_pairs_per_jd])
        sampled_pair_ids = {pair["pair_id"] for pair in sampled_pairs}
        pairs = [pair for pair in pairs if pair["pair_id"] in sampled_pair_ids]

        referenced_docs = {pair["cv_id"] for pair in pairs} | {pair["jd_id"] for pair in pairs}
        docs_by_id = {doc_id: doc for doc_id, doc in docs_by_id.items() if doc_id in referenced_docs}
        logging.info(
            "Sampled Kaggle benchmark to at most %s pairs per JD using seed %s.",
            args.max_pairs_per_jd,
            args.sample_seed,
        )

    docs = list(docs_by_id.values())
    write_jsonl(args.docs_output, docs, args.overwrite)
    write_jsonl(args.pairs_output, pairs, args.overwrite)
    jd_counts: dict[str, int] = {}
    for pair in pairs:
        jd_counts[pair["jd_id"]] = jd_counts.get(pair["jd_id"], 0) + 1
    multi_candidate_jds = sum(1 for count in jd_counts.values() if count > 1)
    logging.info(
        "Wrote %s unique docs and %s pairs across %s unique JDs (%s with >1 candidate).",
        len(docs),
        len(pairs),
        len(jd_counts),
        multi_candidate_jds,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare external CV-JD benchmark datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    vanetik = subparsers.add_parser("vanetik", help="Prepare NataliaVanetik GitHub benchmark.")
    vanetik.add_argument("--source-dir", required=True, type=Path)
    vanetik.add_argument("--docs-output", required=True, type=Path)
    vanetik.add_argument("--pairs-output", required=True, type=Path)
    vanetik.add_argument("--overwrite", action="store_true")
    vanetik.add_argument("--verbose", action="store_true")

    kaggle = subparsers.add_parser("kaggle-ranking", help="Prepare Kaggle resume-data-for-ranking CSV.")
    kaggle.add_argument("--input-csv", required=True, type=Path)
    kaggle.add_argument("--docs-output", required=True, type=Path)
    kaggle.add_argument("--pairs-output", required=True, type=Path)
    kaggle.add_argument(
        "--it-only",
        action="store_true",
        help="Keep only IT/software/data/network/admin job titles from the Kaggle ranking CSV.",
    )
    kaggle.add_argument(
        "--job-title-regex",
        default=None,
        help="Optional regex filter applied to job_position_name before sampling.",
    )
    kaggle.add_argument(
        "--max-pairs-per-jd",
        type=int,
        default=None,
        help="Optional deterministic cap per JD for faster benchmark subsets.",
    )
    kaggle.add_argument("--sample-seed", type=int, default=13)
    kaggle.add_argument("--overwrite", action="store_true")
    kaggle.add_argument("--verbose", action="store_true")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    if args.command == "vanetik":
        return prepare_vanetik(args)
    if args.command == "kaggle-ranking":
        return prepare_kaggle(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
