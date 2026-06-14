"""Stage 3: Multi-Factor CV-JD Scoring.

Combines three scoring components:
  S_final = alpha * S_semantic + beta * S_onet + gamma * S_hard

  S_semantic  — cosine similarity of document-level embeddings
  S_onet      — O*NET importance-weighted entity overlap
  S_hard      — sigmoid-based satisfaction of hard constraints (experience, degree, cert)

Inputs are Stage-2 JSONL files (produced by map_entities_to_onet.py) that already
carry `onet_mappings` and `qualification_facts` fields alongside the raw text.

Usage example (rank all CVs against every JD):
  python score_candidates.py \\
    --stage2-input .\\artifacts\\stage2_onet_mapped_semantic_v6_full.jsonl \\
    --output .\\artifacts\\stage3_scores.jsonl \\
    --top-n 10

Usage example (evaluate against a labelled synthetic dataset):
  python score_candidates.py \\
    --stage2-input .\\artifacts\\stage2_onet_mapped_semantic_v6_full.jsonl \\
    --pairs-input .\\artifacts\\synthetic_eval_dataset.jsonl \\
    --output .\\artifacts\\stage3_scores_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Degree hierarchy (lower index = lower qualification)
# ---------------------------------------------------------------------------
DEGREE_RANK: dict[str, int] = {
    "high school": 0,
    "associate": 1,
    "bachelor": 2,
    "bs": 2,
    "ba": 2,
    "undergraduate": 2,
    "master": 3,
    "ms": 3,
    "ma": 3,
    "mba": 3,
    "postgraduate": 3,
    "phd": 4,
    "doctorate": 4,
    "doctoral": 4,
}

TECH_SKILL_PATTERNS: dict[str, tuple[str, ...]] = {
    "python": (r"\bpython\b",),
    "java": (r"\bjava\b",),
    "javascript": (r"\bjavascript\b", r"\bjs\b"),
    "typescript": (r"\btypescript\b", r"\bts\b"),
    "react": (r"\breact(?:\.js|js)?\b",),
    "angular": (r"\bangular\b",),
    "vue": (r"\bvue(?:\.js|js)?\b",),
    "node": (r"\bnode(?:\.js|js)?\b",),
    "django": (r"\bdjango\b",),
    "flask": (r"\bflask\b",),
    "fastapi": (r"\bfastapi\b",),
    "php": (r"\bphp\b",),
    "laravel": (r"\blaravel\b",),
    "csharp": (r"c#", r"\bc sharp\b", r"\bcsharp\b"),
    "dotnet": (r"\.net\b", r"\bdotnet\b", r"\basp\.net\b"),
    "c++": (r"\bc\+\+\b", r"\bcpp\b"),
    "go": (r"\bgolang\b", r"\bgo\b"),
    "swift": (r"\bswift\b",),
    "objective-c": (r"\bobjective[-\s]?c\b",),
    "kotlin": (r"\bkotlin\b",),
    "sql": (r"\bsql\b",),
    "mysql": (r"\bmysql\b",),
    "postgresql": (r"\bpostgres(?:ql)?\b",),
    "mongodb": (r"\bmongo(?:db)?\b",),
    "oracle": (r"\boracle\b",),
    "redis": (r"\bredis\b",),
    "elasticsearch": (r"\belasticsearch\b", r"\belastic\s*search\b"),
    "aws": (r"\baws\b", r"\bamazon web services\b"),
    "azure": (r"\bazure\b",),
    "gcp": (r"\bgcp\b", r"\bgoogle cloud\b"),
    "docker": (r"\bdocker\b",),
    "kubernetes": (r"\bkubernetes\b", r"\bk8s\b"),
    "linux": (r"\blinux\b",),
    "windows-server": (r"\bwindows server\b",),
    "active-directory": (r"\bactive directory\b", r"\bad\b"),
    "vmware": (r"\bvmware\b",),
    "git": (r"\bgit\b", r"\bgithub\b", r"\bgitlab\b",),
    "ci-cd": (r"\bci/cd\b", r"\bcontinuous integration\b", r"\bcontinuous delivery\b"),
    "jenkins": (r"\bjenkins\b",),
    "terraform": (r"\bterraform\b",),
    "ansible": (r"\bansible\b",),
    "spark": (r"\bspark\b", r"\bpyspark\b"),
    "hadoop": (r"\bhadoop\b",),
    "airflow": (r"\bairflow\b",),
    "kafka": (r"\bkafka\b",),
    "pandas": (r"\bpandas\b",),
    "numpy": (r"\bnumpy\b",),
    "scikit-learn": (r"\bscikit[-\s]?learn\b", r"\bsklearn\b"),
    "tensorflow": (r"\btensorflow\b",),
    "pytorch": (r"\bpytorch\b", r"\btorch\b"),
    "machine-learning": (r"\bmachine learning\b", r"\bml\b"),
    "deep-learning": (r"\bdeep learning\b",),
    "nlp": (r"\bnlp\b", r"\bnatural language processing\b"),
    "llm": (r"\bllm\b", r"\blarge language model", r"\bgenerative ai\b"),
    "data-science": (r"\bdata science\b",),
    "data-analysis": (r"\bdata analysis\b", r"\banalytics\b"),
    "tableau": (r"\btableau\b",),
    "power-bi": (r"\bpower\s*bi\b",),
    "rest-api": (r"\brest(?:ful)?\s+api", r"\bapi\b"),
    "graphql": (r"\bgraphql\b",),
    "html": (r"\bhtml\b",),
    "css": (r"\bcss\b",),
    "ios": (r"\bios\b",),
    "android": (r"\bandroid\b",),
    "networking": (r"\bnetwork(?:ing)?\b", r"\btcp/ip\b", r"\bdns\b"),
    "cisco": (r"\bcisco\b", r"\bccna\b"),
    "firewall": (r"\bfirewall\b",),
}

ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "software": (r"\bsoftware engineer", r"\bsoftware developer", r"\bdeveloper\b", r"\bprogrammer\b"),
    "fullstack": (r"\bfull[-\s]?stack\b",),
    "frontend": (r"\bfront[-\s]?end\b", r"\bfrontend\b"),
    "backend": (r"\bback[-\s]?end\b", r"\bbackend\b"),
    "ios": (r"\bios\b", r"\bswift\b"),
    "android": (r"\bandroid\b", r"\bkotlin\b"),
    "ai_ml": (r"\bai engineer\b", r"\bmachine learning\b", r"\bml engineer\b", r"\bgenerative ai\b"),
    "data_engineer": (r"\bdata engineer\b", r"\betl\b", r"\bdata pipeline"),
    "data_science": (r"\bdata scien", r"\bdata scientist\b", r"\banalyst\b"),
    "devops": (r"\bdevops\b", r"\bsite reliability\b", r"\bsre\b"),
    "dba": (r"\bdatabase administrator\b", r"\bdba\b"),
    "sysadmin": (r"\bsystem administrator\b", r"\bsysadmin\b", r"\bserver administrator\b"),
    "network": (r"\bnetwork support\b", r"\bnetwork engineer\b", r"\bnetwork administrator\b"),
    "it_support": (r"\bit support\b", r"\bservice desk\b", r"\bit executive\b"),
}

ROLE_CLUSTERS: tuple[set[str], ...] = (
    {"software", "fullstack", "frontend", "backend", "ios", "android"},
    {"ai_ml", "data_engineer", "data_science"},
    {"devops", "dba", "sysadmin", "network", "it_support"},
)

SENIORITY_PATTERNS: tuple[tuple[int, str], ...] = (
    (0, r"\bintern\b|\binternship\b"),
    (1, r"\bjunior\b|\bentry[-\s]?level\b|\btrainee\b"),
    (2, r"\bmid[-\s]?level\b|\bassociate\b"),
    (3, r"\bsenior\b|\bsr\.?\b"),
    (4, r"\blead\b|\bprincipal\b|\bstaff\b"),
    (5, r"\bmanager\b|\bhead\b|\bdirector\b"),
)

MANDATORY_CUE_RE = re.compile(
    r"\b(required|mandatory|must|minimum|min\.?|at least)\b",
    re.IGNORECASE,
)

BASIC_ML_FEATURES = (
    "semantic_score",
    "onet_score",
    "hard_constraint_score",
)

IT_ML_FEATURES = (
    "semantic_score",
    "onet_score",
    "hard_constraint_score",
    "hard_penalty_factor",
    "tech_skill_score",
    "required_skill_coverage",
    "role_alignment_score",
    "seniority_score",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3: Multi-Factor CV-JD scoring pipeline."
    )
    parser.add_argument(
        "--stage2-input",
        required=True,
        type=Path,
        help="Stage 2 JSONL file containing both CV and JD rows with onet_mappings.",
    )
    parser.add_argument(
        "--pairs-input",
        type=Path,
        default=None,
        help=(
            "Optional JSONL file with {cv_id, jd_id, relevance_label?} pairs to score. "
            "If omitted, all CV x JD combinations are scored (up to --max-pairs)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL with per-pair scores.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Weight for semantic similarity component (default 0.4).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.4,
        help="Weight for O*NET importance component (default 0.4).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        help="Weight for hard constraint component (default 0.2).",
    )
    parser.add_argument(
        "--weight-profile",
        choices=("generic", "it"),
        default="generic",
        help=(
            "Preset linear weights. 'generic' uses alpha/beta/gamma args; "
            "'it' uses semantic-heavy IT benchmark weights 0.95/0.05/0."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model used to encode document texts.",
    )
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Device for the embedding model (cpu / cuda). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size for text embedding.",
    )
    parser.add_argument(
        "--use-cross-encoder",
        action="store_true",
        help="Use CrossEncoder instead of BiEncoder for semantic matching.",
    )
    parser.add_argument(
        "--ml-ranker",
        type=Path,
        default=None,
        help="Path to an XGBoost model (JSON) to use instead of the linear alpha/beta/gamma formula.",
    )
    parser.add_argument(
        "--ml-feature-set",
        choices=("basic", "it"),
        default="it",
        help="Feature vector used by --ml-ranker. 'it' includes skill, role, seniority, and penalty features.",
    )
    parser.add_argument(
        "--hard-constraint-mode",
        choices=("auto", "additive", "penalty", "off"),
        default="auto",
        help=(
            "How hard constraints affect the final score. 'additive' uses gamma in the weighted sum; "
            "'penalty' multiplies the base score only when mandatory constraints are detected; "
            "'off' keeps the diagnostics but does not affect score."
        ),
    )
    parser.add_argument(
        "--hard-penalty-strength",
        type=float,
        default=0.5,
        help="Penalty strength for mandatory hard constraints in penalty mode (default 0.5).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Rank top-N CVs per JD in the output (only used in all-pairs mode).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on total CV x JD pairs (for smoke runs).",
    )
    parser.add_argument(
        "--hard-k",
        type=float,
        default=5.0,
        help="Steepness parameter k for the sigmoid hard-constraint formula.",
    )
    parser.add_argument(
        "--hard-theta",
        type=float,
        default=0.6,
        help="Inflection-point parameter θ for the sigmoid hard-constraint formula.",
    )
    parser.add_argument(
        "--onet-correction-factor",
        type=float,
        default=1.0,
        help="Additive correction factor in the O*NET score denominator to avoid zero-division.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stage2_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def split_cv_jd(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition Stage 2 rows by doc_type."""
    cvs: list[dict[str, Any]] = []
    jds: list[dict[str, Any]] = []
    for row in rows:
        doc_type = stringify(row.get("document_type") or row.get("doc_type")).lower()
        if doc_type == "cv":
            cvs.append(row)
        elif doc_type == "jd":
            jds.append(row)
        else:
            logging.debug("Skipping row with unknown doc_type=%r  id=%s", doc_type, row.get("record_id"))
    return cvs, jds


def load_pairs(path: Path) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            pairs.append(json.loads(stripped))
    return pairs


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

class DocumentEmbedder:
    """Thin wrapper around a HuggingFace transformer for mean-pool text embedding."""

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "transformers is required. Install with: pip install transformers"
            ) from exc

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        logging.info("Loading embedding model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, 1), dtype=torch.float32)
        all_embeddings: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state  # (B, L, H)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embeddings.append(F.normalize(pooled, dim=-1).cpu())
        return torch.cat(all_embeddings, dim=0)


class CrossEncoderEmbedder:
    """Thin wrapper around a HuggingFace CrossEncoder for sequence classification."""

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "transformers is required. Install with: pip install transformers"
            ) from exc

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        logging.info("Loading Cross-Encoder model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        all_scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**encoded)
            logits = outputs.logits.squeeze(-1)
            # Sigmoid to get probability/score in [0, 1]
            if logits.dim() == 0:
                scores = [float(torch.sigmoid(logits).cpu())]
            else:
                scores = torch.sigmoid(logits).cpu().tolist()
            all_scores.extend(scores)
        return all_scores


# ---------------------------------------------------------------------------
# Component 1 — Semantic Similarity
# ---------------------------------------------------------------------------

def compute_semantic_scores(
    cv_embeddings: torch.Tensor,
    jd_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Returns an (n_cv, n_jd) cosine similarity matrix in [0, 1]."""
    # Both are already L2-normalised, so dot-product == cosine similarity in [-1, 1]
    similarity = torch.matmul(cv_embeddings, jd_embeddings.T)
    # Map to [0, 1]
    return (similarity + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Component 2 — O*NET Importance Score
# ---------------------------------------------------------------------------

_DEFAULT_IMPORTANCE = 2.5  # mid-point of O*NET 1–5 scale when importance is absent


def _extract_mapped_entities(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return onet_mappings entries that have at least one candidate."""
    mappings = row.get("onet_mappings") or []
    return [m for m in mappings if isinstance(m, dict) and m.get("candidates")]


def _best_candidate(mapping: dict[str, Any]) -> dict[str, Any]:
    candidates = mapping.get("candidates") or []
    return candidates[0] if candidates else {}


def _entity_importance(mapping: dict[str, Any]) -> float:
    """Return the O*NET importance of the best candidate (default if missing)."""
    best = _best_candidate(mapping)
    importance = safe_float(best.get("importance"))
    return importance if importance is not None else _DEFAULT_IMPORTANCE


def _entity_soc_code(mapping: dict[str, Any]) -> str:
    return stringify(_best_candidate(mapping).get("onetsoc_code"))


def compute_onet_score(
    cv_row: dict[str, Any],
    jd_row: dict[str, Any],
    correction_factor: float = 1.0,
) -> float:
    """
    O*NET Importance Score.

    For each JD entity that was mapped to O*NET we check whether the CV contains
    a matching entity (same SOC code family or matching normalized text). The
    importance scores of matched JD entities are summed and normalised by the
    maximum possible score plus a correction factor (see slide formula).
    """
    jd_mappings = _extract_mapped_entities(jd_row)
    if not jd_mappings:
        return 1.0  # no O*NET requirements → full score

    cv_mappings = _extract_mapped_entities(cv_row)

    # Build a set of (normalised entity text, soc code) from CV
    cv_entity_texts: set[str] = {
        stringify(m.get("entity_text")).lower().strip() for m in cv_mappings
    }
    cv_soc_codes: set[str] = {_entity_soc_code(m) for m in cv_mappings if _entity_soc_code(m)}
    cv_element_ids: set[str] = set()
    cv_commodity_codes: set[str] = set()
    for m in cv_mappings:
        best = _best_candidate(m)
        eid = stringify(best.get("element_id"))
        if eid:
            cv_element_ids.add(eid)
        ccode = stringify(best.get("commodity_code"))
        if ccode:
            cv_commodity_codes.add(ccode)

    matched_importance = 0.0
    max_possible = 0.0

    for jd_m in jd_mappings:
        imp = _entity_importance(jd_m)
        max_possible += imp

        jd_text = stringify(jd_m.get("entity_text")).lower().strip()
        jd_soc = _entity_soc_code(jd_m)
        best_jd = _best_candidate(jd_m)
        jd_eid = stringify(best_jd.get("element_id"))
        jd_commodity = stringify(best_jd.get("commodity_code"))

        # Exact match
        exact_match = (
            jd_text in cv_entity_texts
            or (jd_eid and jd_eid in cv_element_ids)
            or (jd_soc and any(c.startswith(jd_soc[:6]) for c in cv_soc_codes))
        )
        if exact_match:
            matched_importance += imp
            continue

        # Soft match
        credit = 0.0
        # 1. Commodity Code (Technology Skills) sharing first 6 digits
        if jd_commodity and any(c.startswith(jd_commodity[:6]) for c in cv_commodity_codes):
            credit = max(credit, 0.5)
        # 2. Element ID (Skills/Knowledge) sharing prefix (e.g. 2.A.1)
        elif jd_eid and any(c.rsplit('.', 1)[0] == jd_eid.rsplit('.', 1)[0] for c in cv_element_ids if '.' in c and '.' in jd_eid):
            credit = max(credit, 0.5)
        # 3. SOC Code (Role) sharing first 2 digits
        elif jd_soc and any(c.startswith(jd_soc[:2]) for c in cv_soc_codes):
            credit = max(credit, 0.25)
            
        matched_importance += imp * credit

    score = matched_importance / (max_possible + correction_factor)
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Component 3 — Hard Constraint Score
# ---------------------------------------------------------------------------

def _sigmoid(x: float, k: float, theta: float) -> float:
    """Sigmoid scoring: σ(k * (x/threshold - θ))."""
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - theta)))
    except OverflowError:
        return 0.0 if k * (x - theta) < 0 else 1.0


def _degree_rank(text: str) -> int:
    lowered = text.lower()
    for keyword, rank in sorted(DEGREE_RANK.items(), key=lambda kv: -len(kv[0])):
        if keyword in lowered:
            return rank
    return -1  # unknown


def _parse_experience_years(facts: list[dict[str, Any]]) -> float | None:
    """Return the maximum experience years mentioned in qualification facts."""
    best: float | None = None
    for fact in facts:
        if stringify(fact.get("fact_type")).upper() != "EXPERIENCE_YEARS":
            continue
        val = safe_float(fact.get("value"))
        if val is not None:
            best = max(best, val) if best is not None else val
    return best


def _parse_degree(facts: list[dict[str, Any]]) -> int:
    """Return the highest degree rank mentioned in qualification facts."""
    best = -1
    for fact in facts:
        if stringify(fact.get("fact_type")).upper() not in {"DEGREE", "EDUCATION"}:
            continue
        rank = _degree_rank(stringify(fact.get("text")) + " " + stringify(fact.get("normalized")))
        best = max(best, rank)
    return best


def _parse_certifications(facts: list[dict[str, Any]]) -> set[str]:
    certs: set[str] = set()
    for fact in facts:
        if stringify(fact.get("fact_type")).upper() != "CERTIFICATION":
            continue
        text = stringify(fact.get("normalized") or fact.get("text")).lower().strip()
        if text:
            certs.add(text)
    return certs


def compute_hard_constraint_score(
    cv_row: dict[str, Any],
    jd_row: dict[str, Any],
    k: float = 5.0,
    theta: float = 0.6,
) -> float:
    """
    Hard Constraint Satisfaction Score.

    Each applicable constraint is scored individually and the results are averaged.
    If the JD has no parseable hard constraints, returns 1.0 (no penalty).
    """
    jd_facts = jd_row.get("qualification_facts") or []
    cv_facts = cv_row.get("qualification_facts") or []

    scores: list[float] = []

    # --- Experience ---
    jd_exp = _parse_experience_years(jd_facts)
    if jd_exp is not None and jd_exp > 0:
        cv_exp = _parse_experience_years(cv_facts)
        if cv_exp is None:
            cv_exp = 0.0
        ratio = cv_exp / jd_exp
        scores.append(_sigmoid(ratio, k, theta))

    # --- Degree ---
    jd_degree = _parse_degree(jd_facts)
    if jd_degree >= 0:
        cv_degree = _parse_degree(cv_facts)
        if cv_degree < 0:
            cv_degree = 0
        # Ratio of ranks on a 0-4 scale; clamp to [0, 1]
        ratio = cv_degree / max(jd_degree, 1)
        scores.append(_sigmoid(ratio, k, theta))

    # --- Certifications ---
    jd_certs = _parse_certifications(jd_facts)
    if jd_certs:
        cv_certs = _parse_certifications(cv_facts)
        # Fraction of required certs that the CV holds
        matched = len(jd_certs & cv_certs)
        ratio = matched / len(jd_certs)
        scores.append(_sigmoid(ratio, k, theta))

    if not scores:
        return 1.0  # no hard constraints in JD
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# IT-specific explicit matching features
# ---------------------------------------------------------------------------

def _combined_text(row: dict[str, Any]) -> str:
    return (
        stringify(row.get("source_job_title"))
        + "\n"
        + stringify(row.get("text"))
        + "\n"
        + " ".join(stringify(e.get("text") or e.get("normalized")) for e in (row.get("entities") or []))
    )


def _match_patterns(text: str, patterns_by_name: dict[str, tuple[str, ...]]) -> set[str]:
    matched: set[str] = set()
    for name, patterns in patterns_by_name.items():
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            matched.add(name)
    return matched


def extract_tech_skills(row: dict[str, Any]) -> set[str]:
    return _match_patterns(_combined_text(row), TECH_SKILL_PATTERNS)


def compute_tech_skill_features(cv_row: dict[str, Any], jd_row: dict[str, Any]) -> dict[str, Any]:
    cv_skills = extract_tech_skills(cv_row)
    jd_skills = extract_tech_skills(jd_row)
    if not jd_skills:
        return {
            "tech_skill_score": 1.0,
            "required_skill_coverage": 1.0,
            "matched_tech_skills": [],
            "missing_required_skills": [],
            "cv_tech_skill_count": len(cv_skills),
            "jd_tech_skill_count": 0,
        }

    matched = cv_skills & jd_skills
    union = cv_skills | jd_skills
    jaccard = len(matched) / len(union) if union else 1.0
    coverage = len(matched) / len(jd_skills)
    return {
        "tech_skill_score": jaccard,
        "required_skill_coverage": coverage,
        "matched_tech_skills": sorted(matched),
        "missing_required_skills": sorted(jd_skills - cv_skills),
        "cv_tech_skill_count": len(cv_skills),
        "jd_tech_skill_count": len(jd_skills),
    }


def extract_roles(row: dict[str, Any], *, title_only: bool = False) -> set[str]:
    text = stringify(row.get("source_job_title")) if title_only else _combined_text(row)
    return _match_patterns(text, ROLE_PATTERNS)


def _same_role_cluster(left: set[str], right: set[str]) -> bool:
    return any(left & cluster and right & cluster for cluster in ROLE_CLUSTERS)


def compute_role_alignment_score(cv_row: dict[str, Any], jd_row: dict[str, Any]) -> tuple[float, set[str], set[str]]:
    jd_roles = extract_roles(jd_row, title_only=True) or extract_roles(jd_row)
    cv_roles = extract_roles(cv_row)
    if not jd_roles:
        return 1.0, cv_roles, jd_roles
    if cv_roles & jd_roles:
        return 1.0, cv_roles, jd_roles
    if cv_roles and _same_role_cluster(cv_roles, jd_roles):
        return 0.7, cv_roles, jd_roles
    if not cv_roles:
        return 0.5, cv_roles, jd_roles
    return 0.2, cv_roles, jd_roles


def extract_seniority(row: dict[str, Any], *, title_only: bool = False) -> int | None:
    text = stringify(row.get("source_job_title")) if title_only else _combined_text(row)
    matches = [level for level, pattern in SENIORITY_PATTERNS if re.search(pattern, text, flags=re.IGNORECASE)]
    return max(matches) if matches else None


def compute_seniority_score(cv_row: dict[str, Any], jd_row: dict[str, Any]) -> tuple[float, int | None, int | None]:
    jd_level = extract_seniority(jd_row, title_only=True)
    cv_level = extract_seniority(cv_row)
    if jd_level is None:
        return 1.0, cv_level, jd_level
    if cv_level is None:
        return 0.7, cv_level, jd_level
    if cv_level >= jd_level:
        return 1.0, cv_level, jd_level
    if jd_level - cv_level == 1:
        return 0.7, cv_level, jd_level
    return 0.3, cv_level, jd_level


def hard_constraint_count(jd_row: dict[str, Any]) -> int:
    facts = jd_row.get("qualification_facts") or []
    supported = {"EXPERIENCE_YEARS", "DEGREE", "EDUCATION", "CERTIFICATION"}
    return sum(1 for fact in facts if stringify(fact.get("fact_type")).upper() in supported)


def has_mandatory_hard_constraints(jd_row: dict[str, Any]) -> bool:
    return hard_constraint_count(jd_row) > 0 and bool(MANDATORY_CUE_RE.search(_combined_text(jd_row)))


def compute_hard_penalty_factor(
    *,
    hard_score: float,
    jd_row: dict[str, Any],
    strength: float,
) -> tuple[float, int, bool]:
    count = hard_constraint_count(jd_row)
    mandatory = has_mandatory_hard_constraints(jd_row)
    if count == 0 or not mandatory:
        return 1.0, count, mandatory
    strength = max(0.0, min(1.0, strength))
    factor = 1.0 - strength * max(0.0, 1.0 - hard_score)
    return max(0.0, min(1.0, factor)), count, mandatory


def build_ml_features(result: dict[str, Any], feature_set: str) -> list[float]:
    names = BASIC_ML_FEATURES if feature_set == "basic" else IT_ML_FEATURES
    return [float(result.get(name, 0.0)) for name in names]


# ---------------------------------------------------------------------------
# Final score
# ---------------------------------------------------------------------------

def combine_scores(
    semantic: float,
    onet: float,
    hard: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    total_weight = alpha + beta + gamma
    if total_weight <= 0:
        return 0.0
    return (alpha * semantic + beta * onet + gamma * hard) / total_weight


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        rid = stringify(row.get("record_id"))
        if rid:
            index[rid] = row
    return index


def score_pair(
    cv_row: dict[str, Any],
    jd_row: dict[str, Any],
    *,
    semantic_score: float,
    alpha: float,
    beta: float,
    gamma: float,
    k: float,
    theta: float,
    correction_factor: float,
    hard_constraint_mode: str,
    hard_penalty_strength: float,
    ml_feature_set: str,
    ml_model: Any = None,
) -> dict[str, Any]:
    """Compute all three components and combine for one CV-JD pair."""
    semantic = semantic_score

    onet = compute_onet_score(cv_row, jd_row, correction_factor=correction_factor)
    hard = compute_hard_constraint_score(cv_row, jd_row, k=k, theta=theta)
    hard_penalty, hard_count, hard_is_mandatory = compute_hard_penalty_factor(
        hard_score=hard,
        jd_row=jd_row,
        strength=hard_penalty_strength,
    )
    skill_features = compute_tech_skill_features(cv_row, jd_row)
    role_alignment, cv_roles, jd_roles = compute_role_alignment_score(cv_row, jd_row)
    seniority, cv_seniority, jd_seniority = compute_seniority_score(cv_row, jd_row)

    result = {
        "cv_id": stringify(cv_row.get("record_id")),
        "jd_id": stringify(jd_row.get("record_id")),
        "semantic_score": round(semantic, 6),
        "onet_score": round(onet, 6),
        "hard_constraint_score": round(hard, 6),
        "hard_penalty_factor": round(hard_penalty, 6),
        "hard_constraint_count": hard_count,
        "hard_constraints_mandatory": hard_is_mandatory,
        "tech_skill_score": round(float(skill_features["tech_skill_score"]), 6),
        "required_skill_coverage": round(float(skill_features["required_skill_coverage"]), 6),
        "matched_tech_skills": skill_features["matched_tech_skills"],
        "missing_required_skills": skill_features["missing_required_skills"],
        "cv_tech_skill_count": skill_features["cv_tech_skill_count"],
        "jd_tech_skill_count": skill_features["jd_tech_skill_count"],
        "role_alignment_score": round(role_alignment, 6),
        "cv_roles": sorted(cv_roles),
        "jd_roles": sorted(jd_roles),
        "seniority_score": round(seniority, 6),
        "cv_seniority": cv_seniority,
        "jd_seniority": jd_seniority,
    }
    
    if ml_model is not None:
        import numpy as np
        X = np.array([build_ml_features(result, ml_feature_set)])
        final = float(ml_model.predict(X)[0])
    else:
        if hard_constraint_mode == "additive":
            final = combine_scores(semantic, onet, hard, alpha, beta, gamma)
        else:
            final = combine_scores(semantic, onet, 0.0, alpha, beta, 0.0)
            if hard_constraint_mode == "penalty":
                final *= hard_penalty

    result["score"] = round(final, 6)
    result["score_breakdown"] = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "hard_constraint_mode": hard_constraint_mode,
        "hard_penalty_strength": hard_penalty_strength,
        "ml_feature_set": ml_feature_set,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if not args.stage2_input.exists():
        raise SystemExit(f"Stage 2 input not found: {args.stage2_input}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output}. Pass --overwrite.")

    if args.weight_profile == "it":
        alpha, beta, gamma = 0.95, 0.05, 0.0
    else:
        alpha, beta, gamma = args.alpha, args.beta, args.gamma
    hard_constraint_mode = args.hard_constraint_mode
    if hard_constraint_mode == "auto":
        hard_constraint_mode = "off" if args.weight_profile == "it" else "additive"
    if abs(alpha + beta + gamma) < 1e-9:
        raise SystemExit("Weights alpha+beta+gamma must not all be zero.")
    logging.info("Using score weights alpha=%.4f beta=%.4f gamma=%.4f", alpha, beta, gamma)
    logging.info("Using hard constraint mode: %s", hard_constraint_mode)

    logging.info("Loading Stage 2 data from %s", args.stage2_input)
    all_rows = load_stage2_jsonl(args.stage2_input)
    cv_rows, jd_rows = split_cv_jd(all_rows)
    logging.info("Found %d CVs and %d JDs", len(cv_rows), len(jd_rows))

    ml_model = None
    if args.ml_ranker:
        import xgboost as xgb
        if not args.ml_ranker.exists():
            raise SystemExit(f"ML Ranker model not found: {args.ml_ranker}")
        ml_model = xgb.XGBRanker()
        ml_model.load_model(args.ml_ranker)
        logging.info("Loaded ML Ranker from %s", args.ml_ranker)

    if not cv_rows:
        raise SystemExit("No CV documents found (doc_type='cv') in the Stage 2 input.")
    if not jd_rows:
        raise SystemExit("No JD documents found (doc_type='jd') in the Stage 2 input.")

    # Build lookup by record_id for fast pair resolution
    cv_index = build_index_by_id(cv_rows)
    jd_index = build_index_by_id(jd_rows)

    # Determine which pairs to score
    if args.pairs_input:
        if not args.pairs_input.exists():
            raise SystemExit(f"Pairs input not found: {args.pairs_input}")
        raw_pairs = load_pairs(args.pairs_input)
        pairs_to_score: list[tuple[str, str, dict[str, Any]]] = []
        for pair in raw_pairs:
            cv_id = stringify(pair.get("cv_id"))
            jd_id = stringify(pair.get("jd_id"))
            
            # Synthetic dataset provides self-contained records. If they exist, index them.
            if "cv_record" in pair and cv_id not in cv_index:
                cv_index[cv_id] = pair["cv_record"]
                cv_rows.append(pair["cv_record"])
            if "jd_record" in pair and jd_id not in jd_index:
                jd_index[jd_id] = pair["jd_record"]
                jd_rows.append(pair["jd_record"])
                
            if cv_id not in cv_index:
                logging.warning("cv_id %r not found in Stage 2 data — skipping pair.", cv_id)
                continue
            if jd_id not in jd_index:
                logging.warning("jd_id %r not found in Stage 2 data — skipping pair.", jd_id)
                continue
            pairs_to_score.append((cv_id, jd_id, pair))
        logging.info("Loaded %d explicit pairs from %s.", len(pairs_to_score), args.pairs_input)
    else:
        # All CV × JD combinations (use top-N per JD mode)
        pairs_to_score = []
        for jd in jd_rows:
            jd_id = stringify(jd.get("record_id"))
            for cv in cv_rows:
                cv_id = stringify(cv.get("record_id"))
                pairs_to_score.append((cv_id, jd_id, {}))
                if args.max_pairs and len(pairs_to_score) >= args.max_pairs:
                    break
            if args.max_pairs and len(pairs_to_score) >= args.max_pairs:
                break
        logging.info("All-pairs mode: %d pairs to score.", len(pairs_to_score))

    # Embed all texts
    def _text(row: dict[str, Any]) -> str:
        return stringify(row.get("text"))[:2000]
        
    cross_score_map = {}
    cv_emb_map = {}
    jd_emb_map = {}
    
    if args.use_cross_encoder:
        logging.info("Using CrossEncoder logic...")
        cross_embedder = CrossEncoderEmbedder(
            args.embedding_model, 
            device=args.embedding_device, 
            batch_size=args.embedding_batch_size
        )
        pair_texts = []
        for cv_id, jd_id, _ in pairs_to_score:
            pair_texts.append((_text(cv_index[cv_id]), _text(jd_index[jd_id])))
        logging.info("Scoring %d pairs with CrossEncoder...", len(pair_texts))
        cross_scores = cross_embedder.score_pairs(pair_texts)
        cross_score_map = {(cv_id, jd_id): s for (cv_id, jd_id, _), s in zip(pairs_to_score, cross_scores)}
    else:
        embedder = DocumentEmbedder(
            args.embedding_model,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )

        # Collect unique IDs that appear in the pairs
        needed_cv_ids = {cv_id for cv_id, _, _ in pairs_to_score}
        needed_jd_ids = {jd_id for _, jd_id, _ in pairs_to_score}

        cv_id_list = [rid for rid in cv_index if rid in needed_cv_ids]
        jd_id_list = [rid for rid in jd_index if rid in needed_jd_ids]

        logging.info("Embedding %d CVs …", len(cv_id_list))
        cv_embeddings_tensor = embedder.encode([_text(cv_index[rid]) for rid in cv_id_list])
        cv_emb_map = {rid: cv_embeddings_tensor[i] for i, rid in enumerate(cv_id_list)}

        logging.info("Embedding %d JDs …", len(jd_id_list))
        jd_embeddings_tensor = embedder.encode([_text(jd_index[rid]) for rid in jd_id_list])
        jd_emb_map = {rid: jd_embeddings_tensor[i] for i, rid in enumerate(jd_id_list)}

    # Score pairs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    # Group by JD for top-N ranking when in all-pairs mode
    jd_results: dict[str, list[dict[str, Any]]] = {}

    processed = 0
    with args.output.open("w", encoding="utf-8") as out_handle:
        for cv_id, jd_id, extra_fields in pairs_to_score:
            if args.use_cross_encoder:
                semantic_val = cross_score_map.get((cv_id, jd_id), 0.0)
            else:
                cv_emb = cv_emb_map.get(cv_id)
                jd_emb = jd_emb_map.get(jd_id)
                if cv_emb is None or jd_emb is None:
                    continue
                sim = float(torch.dot(cv_emb, jd_emb).clamp(-1.0, 1.0))
                semantic_val = (sim + 1.0) / 2.0

            result = score_pair(
                cv_index[cv_id],
                jd_index[jd_id],
                semantic_score=semantic_val,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                k=args.hard_k,
                theta=args.hard_theta,
                correction_factor=args.onet_correction_factor,
                hard_constraint_mode=hard_constraint_mode,
                hard_penalty_strength=args.hard_penalty_strength,
                ml_feature_set=args.ml_feature_set,
                ml_model=ml_model,
            )
            # Pass through ground truth label if present (from synthetic eval)
            if "relevance_label" in extra_fields:
                result["relevance_label"] = extra_fields["relevance_label"]
            if extra_fields.get("pair_id"):
                result["pair_id"] = extra_fields["pair_id"]

            if args.pairs_input:
                # Write immediately in explicit-pairs mode
                out_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                jd_results.setdefault(jd_id, []).append(result)

            processed += 1
            if processed % 1000 == 0:
                logging.info("Scored %d pairs …", processed)

        if not args.pairs_input:
            # Write top-N per JD
            written = 0
            for jd_id, results in jd_results.items():
                results.sort(key=lambda r: r["score"], reverse=True)
                for rank, result in enumerate(results[: args.top_n], start=1):
                    result["rank"] = rank
                    out_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    written += 1
            logging.info("Wrote top-%d results for %d JDs (%d rows total).", args.top_n, len(jd_results), written)

    logging.info("Done. Scored %d pairs → %s", processed, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
