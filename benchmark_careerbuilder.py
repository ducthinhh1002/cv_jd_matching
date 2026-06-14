from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import random
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from prepare_external_benchmark import infer_qualification_facts
from score_candidates import (
    DocumentEmbedder,
    compute_onet_score,
    compute_hard_constraint_score,
    compute_role_alignment_score,
    compute_seniority_score,
    compute_tech_skill_features,
)

csv.field_size_limit(2**31 - 1)


IT_PATTERN = re.compile(
    r"\b("
    r"software|developer|programmer|engineer|data|database|dba|devops|"
    r"network|systems?|administrator|security|python|java|javascript|"
    r"sql|oracle|sap|qa|tester|web|cloud|linux|windows|server|"
    r"information technology|technical support|help desk"
    r")\b",
    re.IGNORECASE,
)

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}")
TAG_RE = re.compile(r"<[^>]+>")

GENERIC_ONET_TERMS = {
    "work",
    "worker",
    "workers",
    "manager",
    "managers",
    "specialist",
    "specialists",
    "engineer",
    "engineers",
    "assistant",
    "assistants",
    "associate",
    "associates",
    "representative",
    "representatives",
    "customer",
    "service",
    "services",
    "sales",
}


@dataclass(frozen=True)
class UserKey:
    user_id: str
    window_id: str


@dataclass
class BenchmarkPair:
    user_key: UserKey
    job_id: str
    label: int


FeatureVector = dict[str, dict[tuple[UserKey, str], float]]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def clean_html(text: str) -> str:
    text = html.unescape(stringify(text))
    text = TAG_RE.sub(" ", text)
    text = text.replace("\\r", " ").replace("\\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_score_map(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if abs(hi - lo) < 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if len(token) > 1]


def read_tsv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        yield from reader


def build_job_text(row: dict[str, str]) -> str:
    parts = [
        row.get("Title", ""),
        clean_html(row.get("Description", "")),
        clean_html(row.get("Requirements", "")),
        row.get("City", ""),
        row.get("State", ""),
        row.get("Country", ""),
    ]
    return "\n".join(part for part in parts if stringify(part).strip())


def build_user_profile(
    user: dict[str, str],
    history_titles: list[str],
    applied_job_texts: list[str] | None = None,
    max_applied_chars: int = 4000,
) -> str:
    fields = [
        f"Location: {user.get('City', '')}, {user.get('State', '')}, {user.get('Country', '')}",
        f"Degree: {user.get('DegreeType', '')}",
        f"Major: {user.get('Major', '')}",
        f"Total years experience: {user.get('TotalYearsExperience', '')}",
        f"Currently employed: {user.get('CurrentlyEmployed', '')}",
        f"Managed others: {user.get('ManagedOthers', '')}",
        "Work history: " + "; ".join(history_titles[:12]),
    ]
    if applied_job_texts:
        applied_text = "\n\n".join(applied_job_texts)
        fields.append("Previously applied jobs:\n" + applied_text[:max_applied_chars])
    return "\n".join(field for field in fields if field.strip())


def load_users(path: Path) -> dict[UserKey, dict[str, str]]:
    users: dict[UserKey, dict[str, str]] = {}
    for row in read_tsv(path):
        if row.get("Split") != "Train":
            continue
        users[UserKey(row["UserID"], row["WindowID"])] = row
    logging.info("Loaded %s train users.", len(users))
    return users


def load_user_history(path: Path) -> dict[UserKey, list[str]]:
    history: dict[UserKey, list[tuple[int, str]]] = defaultdict(list)
    for row in read_tsv(path):
        if row.get("Split") != "Train":
            continue
        key = UserKey(row["UserID"], row["WindowID"])
        try:
            sequence = int(row.get("Sequence") or 0)
        except ValueError:
            sequence = 0
        title = stringify(row.get("JobTitle")).strip()
        if title:
            history[key].append((sequence, title))
    result = {
        key: [title for _, title in sorted(items)]
        for key, items in history.items()
    }
    logging.info("Loaded history for %s users.", len(result))
    return result


def load_applications(path: Path) -> tuple[dict[UserKey, list[str]], dict[str, set[str]], set[str]]:
    apps_by_user: dict[UserKey, list[str]] = defaultdict(list)
    jobs_by_window: dict[str, set[str]] = defaultdict(set)
    all_job_ids: set[str] = set()
    for row in read_tsv(path):
        if row.get("Split") != "Train":
            continue
        key = UserKey(row["UserID"], row["WindowID"])
        job_id = row["JobID"]
        apps_by_user[key].append(job_id)
        jobs_by_window[key.window_id].add(job_id)
        all_job_ids.add(job_id)
    logging.info(
        "Loaded %s train application groups and %s unique applied jobs.",
        len(apps_by_user),
        len(all_job_ids),
    )
    return apps_by_user, jobs_by_window, all_job_ids


def is_it_job(row: dict[str, str]) -> bool:
    text = " ".join([row.get("Title", ""), row.get("Description", ""), row.get("Requirements", "")])
    return bool(IT_PATTERN.search(clean_html(text)))


def load_jobs_from_zip(
    zip_path: Path,
    needed_ids: set[str] | None = None,
    *,
    only_app_job_ids: set[str] | None = None,
    job_filter: str = "all",
) -> dict[str, dict[str, str]]:
    jobs: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("jobs.tsv") as raw:
            text = (line.decode("utf-8", errors="replace") for line in raw)
            reader = csv.DictReader(text, delimiter="\t")
            for row in reader:
                job_id = row["JobID"]
                if needed_ids is not None and job_id not in needed_ids:
                    continue
                if only_app_job_ids is not None and job_id not in only_app_job_ids:
                    continue
                if job_filter == "it" and not is_it_job(row):
                    continue
                jobs[job_id] = row
                if needed_ids is not None and len(jobs) >= len(needed_ids):
                    break
    logging.info("Loaded %s jobs from %s.", len(jobs), zip_path)
    return jobs


def build_benchmark_pairs(
    *,
    apps_by_user: dict[UserKey, list[str]],
    jobs_by_window: dict[str, set[str]],
    users: dict[UserKey, dict[str, str]],
    rng: random.Random,
    num_users: int,
    negatives_per_user: int,
    positives_per_user: int,
    available_job_ids: set[str] | None = None,
) -> list[BenchmarkPair]:
    eligible: list[UserKey] = []
    for key, job_ids in apps_by_user.items():
        positives = [job_id for job_id in job_ids if available_job_ids is None or job_id in available_job_ids]
        if key in users and len(set(positives)) >= positives_per_user:
            eligible.append(key)
    rng.shuffle(eligible)

    pairs: list[BenchmarkPair] = []
    for key in eligible:
        if len({pair.user_key for pair in pairs}) >= num_users:
            break
        applied = list(dict.fromkeys(apps_by_user[key]))
        positives = [job_id for job_id in applied if available_job_ids is None or job_id in available_job_ids]
        if len(positives) < positives_per_user:
            continue
        rng.shuffle(positives)
        chosen_positives = positives[:positives_per_user]
        pool = list(jobs_by_window.get(key.window_id, set()))
        if available_job_ids is not None:
            pool = [job_id for job_id in pool if job_id in available_job_ids]
        negative_pool = [job_id for job_id in pool if job_id not in set(applied)]
        if len(negative_pool) < negatives_per_user:
            continue
        chosen_negatives = rng.sample(negative_pool, negatives_per_user)
        pairs.extend(BenchmarkPair(key, job_id, 1) for job_id in chosen_positives)
        pairs.extend(BenchmarkPair(key, job_id, 0) for job_id in chosen_negatives)

    sampled_users = len({pair.user_key for pair in pairs})
    logging.info("Built benchmark with %s users and %s pairs.", sampled_users, len(pairs))
    return pairs


def bm25_scores(query_texts: dict[UserKey, str], job_texts: dict[str, str], pairs: list[BenchmarkPair]) -> dict[tuple[UserKey, str], float]:
    candidate_job_ids = sorted({pair.job_id for pair in pairs})
    doc_tokens = {job_id: tokenize(job_texts[job_id]) for job_id in candidate_job_ids}
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens.values():
        doc_freq.update(set(tokens))
    doc_count = len(doc_tokens)
    avg_len = statistics.mean([len(tokens) for tokens in doc_tokens.values()] or [1])
    k1 = 1.5
    b = 0.75
    scores: dict[tuple[UserKey, str], float] = {}
    query_tokens_by_user = {key: Counter(tokenize(text)) for key, text in query_texts.items()}
    term_counts_by_job = {job_id: Counter(tokens) for job_id, tokens in doc_tokens.items()}

    for pair in pairs:
        q_tokens = query_tokens_by_user[pair.user_key]
        d_counts = term_counts_by_job[pair.job_id]
        doc_len = max(1, len(doc_tokens[pair.job_id]))
        score = 0.0
        for token, q_weight in q_tokens.items():
            tf = d_counts.get(token, 0)
            if tf <= 0:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * doc_len / avg_len)
            score += idf * (tf * (k1 + 1.0) / denom) * min(2.0, 1.0 + math.log1p(q_weight))
        scores[(pair.user_key, pair.job_id)] = score
    return scores


def semantic_scores(
    query_texts: dict[UserKey, str],
    job_texts: dict[str, str],
    pairs: list[BenchmarkPair],
    *,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> dict[tuple[UserKey, str], float]:
    embedder = DocumentEmbedder(model_name, device=device, batch_size=batch_size)
    user_keys = sorted({pair.user_key for pair in pairs}, key=lambda key: (key.window_id, key.user_id))
    job_ids = sorted({pair.job_id for pair in pairs})
    user_embeddings = embedder.encode([query_texts[key][:2000] for key in user_keys])
    job_embeddings = embedder.encode([job_texts[job_id][:2000] for job_id in job_ids])
    user_index = {key: user_embeddings[index] for index, key in enumerate(user_keys)}
    job_index = {job_id: job_embeddings[index] for index, job_id in enumerate(job_ids)}

    scores: dict[tuple[UserKey, str], float] = {}
    for pair in pairs:
        sim = float(torch.dot(user_index[pair.user_key], job_index[pair.job_id]).clamp(-1.0, 1.0))
        scores[(pair.user_key, pair.job_id)] = (sim + 1.0) / 2.0
    return scores


def load_onet_terms(path: Path, max_terms: int) -> list[str]:
    terms: set[str] = set()
    allowed = {"occupation", "technology_skill", "tool", "skill", "knowledge"}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if stringify(row.get("entry_type")) not in allowed:
                continue
            candidates = [row.get("normalized_text"), row.get("title")]
            candidates.extend(row.get("alias_normalized") or [])
            for candidate in candidates:
                term = re.sub(r"\s+", " ", stringify(candidate).lower()).strip()
                if not term or len(term) < 4 or len(term) > 60:
                    continue
                if term in GENERIC_ONET_TERMS:
                    continue
                if len(term.split()) > 6:
                    continue
                terms.add(term)
    sorted_terms = sorted(terms, key=lambda item: (-len(item), item))[:max_terms]
    logging.info("Loaded %s O*NET lexical terms.", len(sorted_terms))
    return sorted_terms


def extract_onet_terms(text: str, terms: list[str]) -> set[str]:
    normalized = " " + re.sub(r"\s+", " ", clean_html(text).lower()) + " "
    found: set[str] = set()
    for term in terms:
        if f" {term} " in normalized:
            found.add(term)
    return found


def onet_scores(
    query_texts: dict[UserKey, str],
    job_texts: dict[str, str],
    pairs: list[BenchmarkPair],
    terms: list[str],
) -> dict[tuple[UserKey, str], float]:
    user_terms = {key: extract_onet_terms(text, terms) for key, text in query_texts.items()}
    job_terms = {job_id: extract_onet_terms(text, terms) for job_id, text in job_texts.items()}
    scores: dict[tuple[UserKey, str], float] = {}
    for pair in pairs:
        q_terms = user_terms[pair.user_key]
        d_terms = job_terms[pair.job_id]
        if not d_terms:
            score = 0.0
        else:
            score = len(q_terms & d_terms) / len(d_terms)
        scores[(pair.user_key, pair.job_id)] = score
    return scores


def onet_semantic_scores(
    query_texts: dict[UserKey, str],
    job_texts: dict[str, str],
    pairs: list[BenchmarkPair],
    *,
    onet_index: Path,
    model_name: str,
    device: str | None,
    batch_size: int,
    max_terms: int,
) -> dict[tuple[UserKey, str], float]:
    """Semantic O*NET proxy: compare texts to a compact O*NET term bank."""
    if not onet_index.exists():
        return { (pair.user_key, pair.job_id): 0.0 for pair in pairs }
    terms = load_onet_terms(onet_index, max_terms)
    if not terms:
        return { (pair.user_key, pair.job_id): 0.0 for pair in pairs }

    embedder = DocumentEmbedder(model_name, device=device, batch_size=batch_size)
    term_embeddings = embedder.encode(terms)
    user_keys = sorted({pair.user_key for pair in pairs}, key=lambda key: (key.window_id, key.user_id))
    job_ids = sorted({pair.job_id for pair in pairs})
    user_embeddings = embedder.encode([query_texts[key][:2000] for key in user_keys])
    job_embeddings = embedder.encode([job_texts[job_id][:2000] for job_id in job_ids])

    def top_term_vector(embedding: torch.Tensor, top_k: int = 12) -> torch.Tensor:
        similarities = torch.matmul(term_embeddings, embedding)
        k = min(top_k, len(similarities))
        indices = torch.topk(similarities, k=k).indices
        vec = term_embeddings[indices].mean(dim=0)
        return F.normalize(vec, dim=0)

    user_vecs = {key: top_term_vector(user_embeddings[index]) for index, key in enumerate(user_keys)}
    job_vecs = {job_id: top_term_vector(job_embeddings[index]) for index, job_id in enumerate(job_ids)}
    scores: dict[tuple[UserKey, str], float] = {}
    for pair in pairs:
        sim = float(torch.dot(user_vecs[pair.user_key], job_vecs[pair.job_id]).clamp(-1.0, 1.0))
        scores[(pair.user_key, pair.job_id)] = (sim + 1.0) / 2.0
    return scores


def row_for_text(record_id: str, doc_type: str, text: str) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "document_type": doc_type,
        "source_job_title": text.splitlines()[0][:160] if text.splitlines() else "",
        "text": text,
        "qualification_facts": infer_qualification_facts(text),
        "entities": [],
        "onet_mappings": [],
    }


def combine_weighted_scores(
    pairs: list[BenchmarkPair],
    feature_maps: FeatureVector,
    weights: dict[str, float],
) -> dict[tuple[UserKey, str], float]:
    scores: dict[tuple[UserKey, str], float] = {}
    total = sum(max(0.0, weight) for weight in weights.values())
    if total <= 0:
        total = 1.0
    for pair in pairs:
        key = (pair.user_key, pair.job_id)
        value = 0.0
        for feature_name, weight in weights.items():
            value += max(0.0, weight) * feature_maps[feature_name].get(key, 0.0)
        scores[key] = value / total
    return scores


def reciprocal_rank_fusion_scores(
    pairs: list[BenchmarkPair],
    score_maps: FeatureVector,
    weights: dict[str, float],
    *,
    rank_k: int = 60,
) -> dict[tuple[UserKey, str], float]:
    grouped: dict[UserKey, list[BenchmarkPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.user_key].append(pair)

    fused: dict[tuple[UserKey, str], float] = {}
    active_weights = {name: max(0.0, weight) for name, weight in weights.items() if weight > 0}
    if not active_weights:
        active_weights = {"bm25": 1.0}

    for user_key, group in grouped.items():
        user_scores = {(pair.user_key, pair.job_id): 0.0 for pair in group}
        for feature_name, weight in active_weights.items():
            feature_scores = score_maps[feature_name]
            ranked = sorted(
                group,
                key=lambda pair: (-feature_scores.get((pair.user_key, pair.job_id), 0.0), pair.job_id),
            )
            for rank, pair in enumerate(ranked, start=1):
                user_scores[(pair.user_key, pair.job_id)] += weight / (rank_k + rank)
        normalized = normalize_score_map({job_id: value for (_, job_id), value in user_scores.items()})
        for pair in group:
            fused[(pair.user_key, pair.job_id)] = normalized.get(pair.job_id, 0.0)
    return fused


def tune_rrf_weights(
    pairs: list[BenchmarkPair],
    score_maps: FeatureVector,
    *,
    ks: list[int],
) -> tuple[dict[str, float], dict[str, Any]]:
    best_weights: dict[str, float] = {}
    best_metrics: dict[str, Any] = {}
    best_score = -1.0
    main_grid = [0.0, 0.5, 1.0, 2.0]
    skill_grid = [0.0, 0.25, 0.5, 1.0]
    for bm25_w in main_grid:
        for semantic_w in main_grid:
            for proposed_w in main_grid:
                for skill_w in skill_grid:
                    for tech_w in skill_grid:
                        weights = {
                            "bm25": bm25_w,
                            "sbert_cosine": semantic_w,
                            "proposed_hybrid": proposed_w,
                            "skill_coverage": skill_w,
                            "tech_skill": tech_w,
                        }
                        if sum(weights.values()) <= 0:
                            continue
                        scores = reciprocal_rank_fusion_scores(pairs, score_maps, weights)
                        metrics = evaluate(pairs, scores, ks=ks)
                        objective = metrics.get("map", 0.0) + metrics.get("ndcg_at_5", 0.0)
                        if objective > best_score:
                            best_score = objective
                            best_weights = weights
                            best_metrics = metrics
    return best_weights, best_metrics


def cross_encoder_score_pairs(
    text_pairs: list[tuple[str, str]],
    *,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> list[float]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required for cross-encoder reranking.") from exc

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Loading CrossEncoder reranker %s on %s.", model_name, resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(resolved_device)
    model.eval()

    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(text_pairs), batch_size):
            batch = text_pairs[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(resolved_device)
            logits = model(**encoded).logits
            if logits.ndim == 1 or logits.shape[-1] == 1:
                batch_scores = torch.sigmoid(logits.reshape(-1))
            else:
                batch_scores = torch.softmax(logits, dim=-1)[:, -1]
            scores.extend(float(value) for value in batch_scores.cpu())
    return scores


def rerank_top_k_with_cross_encoder(
    pairs: list[BenchmarkPair],
    base_scores: dict[tuple[UserKey, str], float],
    query_texts: dict[UserKey, str],
    job_texts: dict[str, str],
    *,
    model_name: str,
    device: str | None,
    batch_size: int,
    top_k: int,
    cross_encoder_weight: float,
) -> dict[tuple[UserKey, str], float]:
    grouped: dict[UserKey, list[BenchmarkPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.user_key].append(pair)

    reranked = {key: value for key, value in base_scores.items()}
    text_pairs: list[tuple[str, str]] = []
    score_targets: list[tuple[UserKey, str]] = []
    for user_key, group in grouped.items():
        ranked = sorted(
            group,
            key=lambda pair: (-base_scores.get((pair.user_key, pair.job_id), 0.0), pair.job_id),
        )[:top_k]
        for pair in ranked:
            text_pairs.append((query_texts[user_key][:1800], job_texts[pair.job_id][:1800]))
            score_targets.append((user_key, pair.job_id))

    raw_scores = cross_encoder_score_pairs(
        text_pairs,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    raw_by_key = dict(zip(score_targets, raw_scores, strict=True))

    for user_key, group in grouped.items():
        candidate_keys = [
            (pair.user_key, pair.job_id)
            for pair in sorted(
                group,
                key=lambda item: (-base_scores.get((item.user_key, item.job_id), 0.0), item.job_id),
            )[:top_k]
        ]
        normalized = normalize_score_map({job_id: raw_by_key[(user_key, job_id)] for _, job_id in candidate_keys})
        for _, job_id in candidate_keys:
            key = (user_key, job_id)
            base = base_scores.get(key, 0.0)
            reranked[key] = (1.0 - cross_encoder_weight) * base + cross_encoder_weight * normalized.get(job_id, 0.0)
    return reranked


def proposed_scores(
    query_texts: dict[UserKey, str],
    job_texts: dict[str, str],
    pairs: list[BenchmarkPair],
    *,
    bm25: dict[tuple[UserKey, str], float],
    semantic: dict[tuple[UserKey, str], float],
    onet: dict[tuple[UserKey, str], float],
) -> dict[tuple[UserKey, str], float]:
    bm25_norm_by_user: dict[UserKey, dict[str, float]] = defaultdict(dict)
    for key in {pair.user_key for pair in pairs}:
        raw = {pair.job_id: bm25[(pair.user_key, pair.job_id)] for pair in pairs if pair.user_key == key}
        bm25_norm_by_user[key] = normalize_score_map(raw)

    user_rows = {key: row_for_text(f"user_{key.user_id}_{key.window_id}", "cv", text) for key, text in query_texts.items()}
    job_rows = {job_id: row_for_text(f"job_{job_id}", "jd", text) for job_id, text in job_texts.items()}
    scores: dict[tuple[UserKey, str], float] = {}

    for pair in pairs:
        urow = user_rows[pair.user_key]
        jrow = job_rows[pair.job_id]
        skill_features = compute_tech_skill_features(urow, jrow)
        role_score, _, _ = compute_role_alignment_score(urow, jrow)
        seniority_score, _, _ = compute_seniority_score(urow, jrow)
        hard_score = compute_hard_constraint_score(urow, jrow)
        skill_score = float(skill_features["required_skill_coverage"])
        score = (
            0.45 * semantic[(pair.user_key, pair.job_id)]
            + 0.18 * onet[(pair.user_key, pair.job_id)]
            + 0.15 * bm25_norm_by_user[pair.user_key].get(pair.job_id, 0.0)
            + 0.10 * skill_score
            + 0.07 * role_score
            + 0.03 * seniority_score
            + 0.02 * hard_score
        )
        scores[(pair.user_key, pair.job_id)] = score
    return scores


def diagnostic_feature_scores(
    query_texts: dict[UserKey, str],
    job_texts: dict[str, str],
    pairs: list[BenchmarkPair],
) -> dict[str, dict[tuple[UserKey, str], float]]:
    user_rows = {key: row_for_text(f"user_{key.user_id}_{key.window_id}", "cv", text) for key, text in query_texts.items()}
    job_rows = {job_id: row_for_text(f"job_{job_id}", "jd", text) for job_id, text in job_texts.items()}
    skill_coverage: dict[tuple[UserKey, str], float] = {}
    tech_skill: dict[tuple[UserKey, str], float] = {}
    role_alignment: dict[tuple[UserKey, str], float] = {}
    seniority: dict[tuple[UserKey, str], float] = {}
    hard_constraint: dict[tuple[UserKey, str], float] = {}

    for pair in pairs:
        key = (pair.user_key, pair.job_id)
        urow = user_rows[pair.user_key]
        jrow = job_rows[pair.job_id]
        skill_features = compute_tech_skill_features(urow, jrow)
        role_score, _, _ = compute_role_alignment_score(urow, jrow)
        seniority_score, _, _ = compute_seniority_score(urow, jrow)
        hard_score = compute_hard_constraint_score(urow, jrow)
        skill_coverage[key] = float(skill_features["required_skill_coverage"])
        tech_skill[key] = float(skill_features["tech_skill_score"])
        role_alignment[key] = role_score
        seniority[key] = seniority_score
        hard_constraint[key] = hard_score
    return {
        "skill_coverage": skill_coverage,
        "tech_skill": tech_skill,
        "role_alignment": role_alignment,
        "seniority": seniority,
        "hard_constraint": hard_constraint,
    }


def user_split(
    pairs: list[BenchmarkPair],
    *,
    valid_fraction: float,
    rng: random.Random,
) -> tuple[list[BenchmarkPair], list[BenchmarkPair]]:
    users = sorted({pair.user_key for pair in pairs}, key=lambda key: (key.window_id, key.user_id))
    rng.shuffle(users)
    valid_count = max(1, round(len(users) * valid_fraction))
    valid_users = set(users[:valid_count])
    valid = [pair for pair in pairs if pair.user_key in valid_users]
    test = [pair for pair in pairs if pair.user_key not in valid_users]
    return valid, test


def tune_weights(
    pairs: list[BenchmarkPair],
    feature_maps: FeatureVector,
    *,
    ks: list[int],
) -> tuple[dict[str, float], dict[str, Any]]:
    grid = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
    best_weights: dict[str, float] = {}
    best_metrics: dict[str, Any] = {}
    best_score = -1.0
    feature_names = ["bm25", "sbert_cosine", "onet_entity", "skill_coverage", "role_alignment", "seniority", "hard_constraint"]
    for semantic_w in grid:
        for bm25_w in grid:
            for onet_w in grid:
                for skill_w in (0.0, 0.1, 0.2, 0.35):
                    weights = {
                        "sbert_cosine": semantic_w,
                        "bm25": bm25_w,
                        "onet_entity": onet_w,
                        "skill_coverage": skill_w,
                        "role_alignment": 0.1,
                        "seniority": 0.05,
                        "hard_constraint": 0.02,
                    }
                    if sum(weights.values()) <= 0:
                        continue
                    scores = combine_weighted_scores(pairs, feature_maps, weights)
                    metrics = evaluate(pairs, scores, ks=ks)
                    objective = metrics.get("map", 0.0) + metrics.get("ndcg_at_5", 0.0)
                    if objective > best_score:
                        best_score = objective
                        best_weights = weights
                        best_metrics = metrics
    return best_weights, best_metrics


def tune_it_skill_weights(
    pairs: list[BenchmarkPair],
    feature_maps: FeatureVector,
    *,
    ks: list[int],
) -> tuple[dict[str, float], dict[str, Any]]:
    grid = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
    tech_grid = [0.0, 0.05, 0.1, 0.2, 0.35]
    best_weights: dict[str, float] = {}
    best_metrics: dict[str, Any] = {}
    best_score = -1.0
    for semantic_w in grid:
        for bm25_w in grid:
            for onet_w in grid:
                for skill_w in (0.0, 0.1, 0.2, 0.35):
                    for tech_w in tech_grid:
                        weights = {
                            "sbert_cosine": semantic_w,
                            "bm25": bm25_w,
                            "onet_entity": onet_w,
                            "skill_coverage": skill_w,
                            "tech_skill": tech_w,
                            "role_alignment": 0.1,
                            "seniority": 0.05,
                            "hard_constraint": 0.02,
                        }
                        if sum(weights.values()) <= 0:
                            continue
                        scores = combine_weighted_scores(pairs, feature_maps, weights)
                        metrics = evaluate(pairs, scores, ks=ks)
                        objective = metrics.get("map", 0.0) + metrics.get("ndcg_at_5", 0.0)
                        if objective > best_score:
                            best_score = objective
                            best_weights = weights
                            best_metrics = metrics
    return best_weights, best_metrics


def evaluate(
    pairs: list[BenchmarkPair],
    scores: dict[tuple[UserKey, str], float],
    *,
    ks: list[int],
) -> dict[str, Any]:
    by_user: dict[UserKey, list[BenchmarkPair]] = defaultdict(list)
    for pair in pairs:
        by_user[pair.user_key].append(pair)

    result: dict[str, Any] = {
        "num_users": len(by_user),
        "num_pairs": len(pairs),
        "num_positive_pairs": sum(pair.label for pair in pairs),
    }
    reciprocal_ranks: list[float] = []
    average_precisions: list[float] = []
    precision_at: dict[int, list[float]] = {k: [] for k in ks}
    recall_at: dict[int, list[float]] = {k: [] for k in ks}
    ndcg_at: dict[int, list[float]] = {k: [] for k in ks}

    for key, group in by_user.items():
        ranked = sorted(
            group,
            key=lambda pair: (-scores.get((pair.user_key, pair.job_id), 0.0), pair.job_id),
        )
        labels = [pair.label for pair in ranked]
        positives = sum(labels)
        if positives <= 0:
            continue
        first_positive_rank = next((index for index, label in enumerate(labels, start=1) if label), None)
        reciprocal_ranks.append(1.0 / first_positive_rank if first_positive_rank else 0.0)

        hit_count = 0
        precision_sum = 0.0
        for rank, label in enumerate(labels, start=1):
            if label:
                hit_count += 1
                precision_sum += hit_count / rank
        average_precisions.append(precision_sum / positives)

        for k in ks:
            top = labels[:k]
            hits = sum(top)
            precision_at[k].append(hits / len(top) if top else 0.0)
            recall_at[k].append(hits / positives)
            dcg = sum((1.0 / math.log2(rank + 1)) for rank, label in enumerate(top, start=1) if label)
            ideal_hits = min(positives, k)
            ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg_at[k].append(dcg / ideal_dcg if ideal_dcg else 0.0)

    result["map"] = round(statistics.mean(average_precisions), 6) if average_precisions else 0.0
    result["mrr"] = round(statistics.mean(reciprocal_ranks), 6) if reciprocal_ranks else 0.0
    for k in ks:
        result[f"precision_at_{k}"] = round(statistics.mean(precision_at[k]), 6) if precision_at[k] else 0.0
        result[f"recall_at_{k}"] = round(statistics.mean(recall_at[k]), 6) if recall_at[k] else 0.0
        result[f"ndcg_at_{k}"] = round(statistics.mean(ndcg_at[k]), 6) if ndcg_at[k] else 0.0
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ranking methods on CareerBuilder job recommendation data.")
    parser.add_argument("--data-dir", type=Path, default=Path("job-recommendation"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/careerbuilder_benchmark"))
    parser.add_argument("--num-users", type=int, default=100)
    parser.add_argument("--negatives-per-user", type=int, default=30)
    parser.add_argument("--positives-per-user", type=int, default=1)
    parser.add_argument(
        "--enrich-profile-with-apps",
        action="store_true",
        help="Append text of previously applied jobs to the user profile.",
    )
    parser.add_argument("--max-profile-apps", type=int, default=5)
    parser.add_argument("--max-profile-app-chars", type=int, default=4000)
    parser.add_argument(
        "--tune-weights",
        action="store_true",
        help="Tune hybrid weights on a user-level validation split and report test metrics.",
    )
    parser.add_argument(
        "--tune-it-skill-weights",
        action="store_true",
        help="Tune an IT-domain hybrid score that includes explicit tech-skill overlap.",
    )
    parser.add_argument(
        "--tune-it-rrf",
        action="store_true",
        help="Tune an IT-domain reciprocal-rank fusion over lexical, semantic, and skill signals.",
    )
    parser.add_argument(
        "--cross-encoder-rerank",
        action="store_true",
        help="Rerank each user's top candidates from the strongest proposed method with a CrossEncoder.",
    )
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross-encoder-top-k", type=int, default=10)
    parser.add_argument("--cross-encoder-weight", type=float, default=0.25)
    parser.add_argument("--cross-encoder-batch-size", type=int, default=16)
    parser.add_argument("--valid-fraction", type=float, default=0.25)
    parser.add_argument("--job-filter", choices=("all", "it"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds for repeated runs.")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--onet-index", type=Path, default=Path("artifacts/onet_index.jsonl"))
    parser.add_argument("--max-onet-terms", type=int, default=25000)
    parser.add_argument(
        "--onet-scoring",
        choices=("lexical", "semantic", "hybrid"),
        default="hybrid",
        help="O*NET baseline/proposed component: lexical term overlap, semantic term-bank matching, or max of both.",
    )
    parser.add_argument(
        "--semantic-onet-terms",
        type=int,
        default=4000,
        help="Number of O*NET terms to embed for semantic O*NET scoring.",
    )
    parser.add_argument("--ks", default="1,5,10")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def aggregate_runs(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted(run_summaries[0]["metrics"])
    aggregate: dict[str, Any] = {}
    for method in methods:
        metric_names = sorted(run_summaries[0]["metrics"][method])
        aggregate[method] = {}
        for metric_name in metric_names:
            values = [summary["metrics"][method][metric_name] for summary in run_summaries]
            if not all(isinstance(value, (int, float)) for value in values):
                continue
            aggregate[method][metric_name] = {
                "mean": round(statistics.mean(values), 6),
                "std": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
            }
    return aggregate


def run_one_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    users: dict[UserKey, dict[str, str]],
    history: dict[UserKey, list[str]],
    apps_by_user: dict[UserKey, list[str]],
    jobs_by_window: dict[str, set[str]],
    app_job_ids: set[str],
    ks: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    rng = random.Random(seed)
    jobs_zip = args.data_dir / "jobs.zip"
    local_jobs_by_window = {window: set(job_ids) for window, job_ids in jobs_by_window.items()}

    available_job_ids: set[str] | None = None
    preloaded_jobs: dict[str, dict[str, str]] = {}
    if args.job_filter == "it":
        preloaded_jobs = load_jobs_from_zip(
            jobs_zip,
            only_app_job_ids=app_job_ids,
            job_filter="it",
        )
        available_job_ids = set(preloaded_jobs)
        filtered_by_window: dict[str, set[str]] = defaultdict(set)
        for job_id, row in preloaded_jobs.items():
            filtered_by_window[row["WindowID"]].add(job_id)
        local_jobs_by_window = filtered_by_window
        logging.info("IT filter retained %s applied jobs.", len(available_job_ids))

    pairs = build_benchmark_pairs(
        apps_by_user=apps_by_user,
        jobs_by_window=local_jobs_by_window,
        users=users,
        rng=rng,
        num_users=args.num_users,
        negatives_per_user=args.negatives_per_user,
        positives_per_user=args.positives_per_user,
        available_job_ids=available_job_ids,
    )
    if not pairs:
        raise SystemExit("No benchmark pairs could be built.")

    needed_job_ids = {pair.job_id for pair in pairs}
    if args.enrich_profile_with_apps:
        for pair in pairs:
            applied = [job_id for job_id in apps_by_user[pair.user_key] if job_id != pair.job_id]
            needed_job_ids.update(applied[: args.max_profile_apps])

    if args.job_filter == "it":
        missing_preload_ids = needed_job_ids - set(preloaded_jobs)
        if missing_preload_ids:
            extra_jobs = load_jobs_from_zip(jobs_zip, needed_ids=missing_preload_ids)
            preloaded_jobs.update(extra_jobs)
        jobs = {job_id: preloaded_jobs[job_id] for job_id in needed_job_ids if job_id in preloaded_jobs}
    else:
        jobs = load_jobs_from_zip(jobs_zip, needed_ids=needed_job_ids)

    missing_jobs = {pair.job_id for pair in pairs} - set(jobs)
    if missing_jobs:
        logging.warning("Dropping %s pairs with missing job text.", len(missing_jobs))
        pairs = [pair for pair in pairs if pair.job_id in jobs]

    job_texts = {job_id: build_job_text(row) for job_id, row in jobs.items()}
    query_texts: dict[UserKey, str] = {}
    for key in {pair.user_key for pair in pairs}:
        applied_texts: list[str] = []
        if args.enrich_profile_with_apps:
            for job_id in apps_by_user[key][: args.max_profile_apps]:
                if job_id in job_texts:
                    applied_texts.append(job_texts[job_id])
        query_texts[key] = build_user_profile(
            users[key],
            history.get(key, []),
            applied_texts,
            max_applied_chars=args.max_profile_app_chars,
        )

    logging.info("Computing BM25 baseline.")
    bm25 = bm25_scores(query_texts, job_texts, pairs)
    logging.info("Computing SBERT cosine baseline.")
    semantic = semantic_scores(
        query_texts,
        job_texts,
        pairs,
        model_name=args.embedding_model,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
    )
    logging.info("Computing O*NET lexical entity baseline.")
    terms = load_onet_terms(args.onet_index, args.max_onet_terms)
    onet_lexical = onet_scores(query_texts, job_texts, pairs, terms)
    onet_semantic: dict[tuple[UserKey, str], float] | None = None
    if args.onet_scoring in {"semantic", "hybrid"}:
        logging.info("Computing O*NET semantic term-bank baseline.")
        onet_semantic = onet_semantic_scores(
            query_texts,
            job_texts,
            pairs,
            onet_index=args.onet_index,
            model_name=args.embedding_model,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
            max_terms=args.semantic_onet_terms,
        )
    if args.onet_scoring == "semantic" and onet_semantic is not None:
        onet = onet_semantic
    elif args.onet_scoring == "hybrid" and onet_semantic is not None:
        onet = {
            key: max(onet_lexical.get(key, 0.0), onet_semantic.get(key, 0.0))
            for key in set(onet_lexical) | set(onet_semantic)
        }
    else:
        onet = onet_lexical
    logging.info("Computing diagnostic features.")
    diagnostics = diagnostic_feature_scores(query_texts, job_texts, pairs)
    logging.info("Computing proposed hybrid score.")
    proposed = proposed_scores(query_texts, job_texts, pairs, bm25=bm25, semantic=semantic, onet=onet)

    feature_maps: FeatureVector = {
        "bm25": bm25,
        "sbert_cosine": semantic,
        "onet_entity": onet,
        **diagnostics,
    }
    methods = {
        "bm25": bm25,
        "sbert_cosine": semantic,
        "onet_entity": onet,
        "proposed_hybrid": proposed,
    }
    if onet_semantic is not None:
        methods["onet_semantic"] = onet_semantic
    tuned_weights: dict[str, float] | None = None
    tuned_valid_metrics: dict[str, Any] | None = None
    it_skill_tuned_weights: dict[str, float] | None = None
    it_skill_tuned_valid_metrics: dict[str, Any] | None = None
    it_rrf_weights: dict[str, float] | None = None
    it_rrf_valid_metrics: dict[str, Any] | None = None
    eval_pairs = pairs
    if args.tune_weights or args.tune_it_skill_weights or args.tune_it_rrf:
        valid_pairs, test_pairs = user_split(pairs, valid_fraction=args.valid_fraction, rng=rng)
        eval_pairs = test_pairs
    if args.tune_weights:
        tuned_weights, tuned_valid_metrics = tune_weights(valid_pairs, feature_maps, ks=ks)
        logging.info("Tuned weights on %s valid pairs: %s", len(valid_pairs), tuned_weights)
        methods["proposed_tuned"] = combine_weighted_scores(test_pairs, feature_maps, tuned_weights)
    if args.tune_it_skill_weights:
        it_skill_tuned_weights, it_skill_tuned_valid_metrics = tune_it_skill_weights(valid_pairs, feature_maps, ks=ks)
        logging.info(
            "IT skill tuned weights on %s valid pairs: %s",
            len(valid_pairs),
            it_skill_tuned_weights,
        )
        methods["proposed_it_skill_tuned"] = combine_weighted_scores(
            test_pairs,
            feature_maps,
            it_skill_tuned_weights,
        )
    if args.tune_it_rrf:
        rrf_maps: FeatureVector = {
            **feature_maps,
            "proposed_hybrid": proposed,
        }
        it_rrf_weights, it_rrf_valid_metrics = tune_rrf_weights(valid_pairs, rrf_maps, ks=ks)
        logging.info("IT RRF tuned weights on %s valid pairs: %s", len(valid_pairs), it_rrf_weights)
        methods["proposed_it_rrf"] = reciprocal_rank_fusion_scores(test_pairs, rrf_maps, it_rrf_weights)
    if args.cross_encoder_rerank:
        base_name = (
            "proposed_it_rrf"
            if "proposed_it_rrf" in methods
            else "proposed_it_skill_tuned"
            if "proposed_it_skill_tuned" in methods
            else "proposed_tuned"
            if "proposed_tuned" in methods
            else "proposed_hybrid"
        )
        logging.info(
            "CrossEncoder reranking top-%s from base=%s.",
            args.cross_encoder_top_k,
            base_name,
        )
        methods["proposed_cross_encoder_rerank"] = rerank_top_k_with_cross_encoder(
            eval_pairs,
            methods[base_name],
            query_texts,
            job_texts,
            model_name=args.cross_encoder_model,
            device=args.embedding_device,
            batch_size=args.cross_encoder_batch_size,
            top_k=args.cross_encoder_top_k,
            cross_encoder_weight=args.cross_encoder_weight,
        )

    results = {name: evaluate(eval_pairs, score_map, ks=ks) for name, score_map in methods.items()}
    summary = {
        "dataset": "CareerBuilder Job Recommendation Challenge",
        "framing": "implicit-feedback job recommendation",
        "job_filter": args.job_filter,
        "num_users_requested": args.num_users,
        "negatives_per_user": args.negatives_per_user,
        "positives_per_user": args.positives_per_user,
        "enrich_profile_with_apps": args.enrich_profile_with_apps,
        "onet_scoring": args.onet_scoring,
        "seed": seed,
        "metrics": results,
        "tuned_weights": tuned_weights,
        "tuned_valid_metrics": tuned_valid_metrics,
        "it_skill_tuned_weights": it_skill_tuned_weights,
        "it_skill_tuned_valid_metrics": it_skill_tuned_valid_metrics,
        "it_rrf_weights": it_rrf_weights,
        "it_rrf_valid_metrics": it_rrf_valid_metrics,
        "cross_encoder_config": {
            "enabled": args.cross_encoder_rerank,
            "model": args.cross_encoder_model,
            "top_k": args.cross_encoder_top_k,
            "weight": args.cross_encoder_weight,
        },
        "notes": [
            "Positive labels are observed applications.",
            "Sampled negatives are unobserved user-job pairs, not guaranteed human negatives.",
            "O*NET baseline uses lexical O*NET term extraction from user/job text.",
        ],
    }

    write_jsonl(
        output_dir / f"pairs_seed{seed}.jsonl",
        (
            {
                "user_id": pair.user_key.user_id,
                "window_id": pair.user_key.window_id,
                "job_id": pair.job_id,
                "label": pair.label,
            }
            for pair in pairs
        ),
    )
    write_jsonl(
        output_dir / f"scores_seed{seed}.jsonl",
        (
            {
                "user_id": pair.user_key.user_id,
                "window_id": pair.user_key.window_id,
                "job_id": pair.job_id,
                "label": pair.label,
                "bm25": round(bm25[(pair.user_key, pair.job_id)], 6),
                "sbert_cosine": round(semantic[(pair.user_key, pair.job_id)], 6),
                "onet_entity": round(onet[(pair.user_key, pair.job_id)], 6),
                "onet_lexical": round(onet_lexical[(pair.user_key, pair.job_id)], 6),
                "onet_semantic": round(onet_semantic[(pair.user_key, pair.job_id)], 6) if onet_semantic else None,
                "proposed_hybrid": round(proposed[(pair.user_key, pair.job_id)], 6),
            }
            for pair in pairs
        ),
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise SystemExit(f"{summary_path} exists. Pass --overwrite.")

    seeds = [int(value.strip()) for value in args.seeds.split(",")] if args.seeds else [args.seed]
    ks = [int(value.strip()) for value in args.ks.split(",") if value.strip()]

    users = load_users(args.data_dir / "users.tsv")
    history = load_user_history(args.data_dir / "user_history.tsv")
    apps_by_user, jobs_by_window, app_job_ids = load_applications(args.data_dir / "apps.tsv")

    run_summaries = []
    for seed in seeds:
        logging.info("=" * 60)
        logging.info("Running CareerBuilder benchmark seed=%s", seed)
        run_summaries.append(
            run_one_seed(
                args,
                seed=seed,
                users=users,
                history=history,
                apps_by_user=apps_by_user,
                jobs_by_window=jobs_by_window,
                app_job_ids=app_job_ids,
                ks=ks,
                output_dir=args.output_dir,
            )
        )

    summary: dict[str, Any]
    if len(run_summaries) == 1:
        summary = run_summaries[0]
    else:
        summary = {
            "dataset": "CareerBuilder Job Recommendation Challenge",
            "framing": "implicit-feedback job recommendation",
            "job_filter": args.job_filter,
            "num_users_requested": args.num_users,
            "negatives_per_user": args.negatives_per_user,
            "positives_per_user": args.positives_per_user,
            "enrich_profile_with_apps": args.enrich_profile_with_apps,
            "onet_scoring": args.onet_scoring,
            "seeds": seeds,
            "runs": run_summaries,
            "metrics_mean_std": aggregate_runs(run_summaries),
        }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info("Results saved to %s", summary_path)
    metrics = summary.get("metrics") or {
        method: values
        for method, values in (summary.get("metrics_mean_std") or {}).items()
    }
    for name, values in metrics.items():
        if "map" in values and isinstance(values["map"], dict):
            logging.info(
                "%-16s MAP=%.4f±%.4f NDCG@5=%.4f±%.4f",
                name,
                values["map"]["mean"],
                values["map"]["std"],
                values.get("ndcg_at_5", {}).get("mean", 0.0),
                values.get("ndcg_at_5", {}).get("std", 0.0),
            )
        elif "map" in values:
            logging.info(
                "%-16s MAP=%.4f MRR=%.4f NDCG@5=%.4f Recall@5=%.4f",
                name,
                values["map"],
                values["mrr"],
                values.get("ndcg_at_5", 0.0),
                values.get("recall_at_5", 0.0),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
