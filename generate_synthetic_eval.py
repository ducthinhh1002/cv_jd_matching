"""Generate a synthetic CV-JD evaluation dataset using the OpenAI API.

Produces 1,000 labelled CV-JD pairs across 20 IT roles with 5 relevance levels
(1=irrelevant → 5=perfect match).  Each pair carries a ground-truth `relevance_label`
that the Stage 3 scorer is later evaluated against.

Dataset design:
  20 IT roles × 10 JDs per role × 5 relevance levels = 1,000 pairs

The script is resumable: already-generated pairs are skipped on restart.

Usage:
  python generate_synthetic_eval.py \\
    --output .\\artifacts\\synthetic_eval_dataset.jsonl \\
    --api-key sk-...

Smoke run (faster, cheaper):
  python generate_synthetic_eval.py \\
    --output .\\artifacts\\synthetic_eval_smoke.jsonl \\
    --roles 2 --jds-per-role 2 --relevance-levels 1 2 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Canonical 20 IT roles + brief description used to seed JD generation
# ---------------------------------------------------------------------------
DEFAULT_IT_ROLES: list[dict[str, str]] = [
    {"role": "Software Engineer", "focus": "backend systems and APIs"},
    {"role": "Frontend Developer", "focus": "React/Vue web interfaces"},
    {"role": "Data Scientist", "focus": "machine learning and predictive modelling"},
    {"role": "Data Engineer", "focus": "ETL pipelines and data warehousing"},
    {"role": "ML Engineer", "focus": "deploying and optimising ML models in production"},
    {"role": "DevOps Engineer", "focus": "CI/CD, Kubernetes, and cloud infrastructure"},
    {"role": "Cloud Architect", "focus": "AWS/GCP/Azure solution design"},
    {"role": "Cybersecurity Analyst", "focus": "threat detection and incident response"},
    {"role": "Network Engineer", "focus": "LAN/WAN, routing, and switching"},
    {"role": "Mobile Developer", "focus": "iOS and Android application development"},
    {"role": "QA Engineer", "focus": "automated and manual software testing"},
    {"role": "Database Administrator", "focus": "PostgreSQL, MySQL, and performance tuning"},
    {"role": "Product Manager", "focus": "agile roadmap and cross-functional delivery"},
    {"role": "Business Intelligence Analyst", "focus": "Power BI, Tableau, and SQL reporting"},
    {"role": "Technical Lead", "focus": "software architecture and team mentoring"},
    {"role": "Site Reliability Engineer", "focus": "uptime, latency SLOs, and on-call response"},
    {"role": "Blockchain Developer", "focus": "smart contracts and decentralised applications"},
    {"role": "AR/VR Developer", "focus": "Unity and Unreal Engine immersive experiences"},
    {"role": "Embedded Systems Engineer", "focus": "C/C++ firmware and microcontroller programming"},
    {"role": "IT Project Manager", "focus": "PMP, budget management, and stakeholder reporting"},
]

# Relevance level definitions — deliberately precise to reduce LLM ambiguity.
# Level 1 intentionally still has SOME it keywords so S_onet is non-trivially low
# (avoids trivially-easy discrimination that inflates metrics).
RELEVANCE_DESCRIPTIONS: dict[int, str] = {
    1: (
        "The CV is irrelevant to the job. "
        "The candidate has IT literacy (e.g., basic computer skills, MS Office) but works in a "
        "completely unrelated profession such as accounting, teaching, or healthcare. "
        "They have NONE of the specific technical skills, programming languages, or tools mentioned in the JD. "
        "Their experience years and degree may happen to match numerically, but the domain is entirely wrong."
    ),
    2: (
        "The CV is weakly relevant. The candidate is an IT professional but in a DIFFERENT specialisation "
        "from the JD (e.g., a QA tester for a Data Science role, or a network admin for a frontend role). "
        "They share at most 1-2 peripheral tools with the JD. "
        "Their seniority or years of experience is significantly below the JD requirement (at least 3 years short). "
        "Their degree field is adjacent but not matching (e.g., Information Systems vs Computer Science)."
    ),
    3: (
        "The CV is partially relevant. The candidate works in the same broad IT domain as the JD "
        "and shares roughly half the required technologies, but has clear gaps: "
        "missing 2-3 key tools/frameworks explicitly listed in the JD, "
        "AND has 1-2 fewer years of experience than required. "
        "They may have the right degree but lack certifications or vice versa."
    ),
    4: (
        "The CV is a strong match. The candidate covers most JD requirements: "
        "correct role family, 4+ of the key technologies listed, "
        "and experience within 1 year of the requirement. "
        "Only 1 minor gap exists — either one secondary tool is missing OR the degree is one level below."
    ),
    5: (
        "The CV is a perfect match. The candidate satisfies EVERY requirement in the JD: "
        "exact job title or very close synonym, ALL key technologies listed, "
        "experience years >= required, degree >= required level, "
        "and any certifications mentioned. No gaps whatsoever."
    ),
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

JD_SYSTEM_PROMPT = (
    "You are a professional recruitment content writer with 10 years of experience "
    "writing IT job descriptions. Return ONLY valid JSON, no markdown fences."
)

# Variant seeds ensure each of the 10 JDs per role is meaningfully different.
_JD_VARIANT_SEEDS: list[dict[str, str]] = [
    {"seniority": "junior",  "exp": "1-2",  "twist": "startup environment, focus on learning velocity"},
    {"seniority": "mid",     "exp": "3-4",  "twist": "product company, strong emphasis on code quality"},
    {"seniority": "senior",  "exp": "5-7",  "twist": "fintech domain, compliance and security awareness required"},
    {"seniority": "lead",    "exp": "7-10", "twist": "team lead responsibilities, mentoring junior devs"},
    {"seniority": "mid",     "exp": "3-5",  "twist": "remote-first company, async collaboration tools valued"},
    {"seniority": "senior",  "exp": "5-8",  "twist": "e-commerce scale, high-traffic system experience needed"},
    {"seniority": "junior",  "exp": "0-2",  "twist": "fresh graduate welcome, strong CS fundamentals required"},
    {"seniority": "senior",  "exp": "6-9",  "twist": "enterprise software, legacy modernisation experience a plus"},
    {"seniority": "mid",     "exp": "2-4",  "twist": "AI-powered product company, interest in ML pipelines valued"},
    {"seniority": "lead",    "exp": "8+",   "twist": "system architect track, cross-team technical ownership"},
]

JD_USER_PROMPT = """\
Write a realistic, detailed job description for a {role} position with focus on {focus}.

Variant context for this JD:
- Seniority: {seniority}
- Experience required: {exp} years
- Company context: {twist}

IMPORTANT: Choose a SPECIFIC and DIFFERENT technology stack than the generic one for this role.
For example, do NOT always use Python/Django for backend — sometimes use Go/Gin, Java/Spring, 
or Node.js/NestJS. Pick a concrete, internally consistent stack.

Return JSON:
{{
  "jd_text": "<400-600 word job description with sections: About the Role, Key Requirements, Nice to Have, Benefits>",
  "required_experience_years": <integer, must match the variant context above>,
  "required_degree": "<one of: none, Bachelor, Master, PhD>",
  "required_certifications": ["<cert1 or empty list>"],
  "key_technologies": ["<tech1>", "<tech2>", "<tech3>", "<tech4>"],
  "seniority": "{seniority}"
}}"""

CV_SYSTEM_PROMPT = (
    "You are a professional resume writer. Generate realistic IT candidate CVs. "
    "Return ONLY valid JSON, no markdown fences."
)

CV_USER_PROMPT = """\
Generate a realistic CV for a candidate applying to the following job description.
The CV MUST match relevance level {level}/5 — read the definition carefully.

--- JOB DESCRIPTION ---
{jd_text}
--- END JD ---

RELEVANCE LEVEL {level}/5 — follow this EXACTLY:
{level_description}

IMPORTANT rules:
- The CV must be internally consistent: dates, job titles, skills, and education must all align.
- Do NOT copy sentences from the JD verbatim into the CV text.
- Do NOT mention the relevance level number anywhere in the CV text.
- Use a unique candidate name, city, and career background (not a stereotype).
- The Experience section must include at least 2 concrete past roles with company names and dates.

Also extract structured facts from the CV you generate:
- candidate_experience_years: total years of professional IT work experience as an integer
- candidate_degree: highest degree (one of: none, Bachelor, Master, PhD)
- candidate_certifications: list of any professional certifications mentioned (empty list if none)

Return JSON:
{{
  "cv_text": "<350-600 word CV with sections: Summary, Experience, Education, Skills>",
  "candidate_name": "<First Last>",
  "relevance_label": {level},
  "relevance_reason": "<1-2 sentence explanation of the match/mismatch with the JD>",
  "candidate_experience_years": <integer>,
  "candidate_degree": "<none | Bachelor | Master | PhD>",
  "candidate_certifications": ["<cert or empty list>"]
}}"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CV-JD pairs for Stage 3 evaluation."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL file with labelled CV-JD pairs.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. Falls back to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--roles",
        type=int,
        default=20,
        help="Number of IT roles to use (1-20). Default 20.",
    )
    parser.add_argument(
        "--jds-per-role",
        type=int,
        default=10,
        help="JDs to generate per role (default 10).",
    )
    parser.add_argument(
        "--relevance-levels",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Which relevance levels to generate CVs for (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="OpenAI sampling temperature (default 0.7). Lower = more reliable JSON.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per API call on transient errors.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="Seconds to wait between retries.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.5,
        help="Seconds to sleep between API requests to avoid rate limits.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

def get_openai_client(api_key: str | None) -> Any:
    try:
        import openai
    except ImportError as exc:
        raise SystemExit(
            "openai package is required. Install with: pip install openai"
        ) from exc

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OpenAI API key required. Pass --api-key or set OPENAI_API_KEY env var."
        )
    return openai.OpenAI(api_key=key)


def call_openai_json(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    """Call OpenAI chat completion and parse the response as JSON."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logging.warning("JSON parse error (attempt %d): %s", attempt + 1, exc)
        except Exception as exc:  # noqa: BLE001
            logging.warning("API error (attempt %d): %s", attempt + 1, exc)
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    raise RuntimeError(f"All {max_retries} attempts failed.")


# ---------------------------------------------------------------------------
# Resume support — track already-generated pair IDs
# ---------------------------------------------------------------------------

def load_existing_pair_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
                pid = row.get("pair_id")
                if pid:
                    ids.add(str(pid))
            except json.JSONDecodeError:
                pass
    return ids


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    roles = DEFAULT_IT_ROLES[: max(1, min(args.roles, len(DEFAULT_IT_ROLES)))]
    relevance_levels = sorted(set(int(lvl) for lvl in args.relevance_levels if 1 <= int(lvl) <= 5))
    if not relevance_levels:
        raise SystemExit("No valid relevance levels specified (must be 1-5).")

    total_pairs = len(roles) * args.jds_per_role * len(relevance_levels)
    logging.info(
        "Plan: %d roles × %d JDs × %d levels = %d pairs",
        len(roles),
        args.jds_per_role,
        len(relevance_levels),
        total_pairs,
    )

    client = get_openai_client(args.api_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = load_existing_pair_ids(args.output)
    if existing_ids:
        logging.info("Resuming: %d pairs already generated.", len(existing_ids))

    generated = 0
    skipped = 0

    with args.output.open("a", encoding="utf-8") as out_handle:
        for role_info in roles:
            role = role_info["role"]
            focus = role_info["focus"]
            role_slug = role.lower().replace(" ", "_")

            for jd_idx in range(1, args.jds_per_role + 1):
                jd_pair_id_prefix = f"{role_slug}_jd{jd_idx}"

                # Check if ALL levels for this JD are already done (skip JD generation)
                all_cv_ids = {
                    f"{jd_pair_id_prefix}_lv{lvl}" for lvl in relevance_levels
                }
                if all_cv_ids.issubset(existing_ids):
                    skipped += len(relevance_levels)
                    logging.debug("Skipping JD %s (all levels done).", jd_pair_id_prefix)
                    continue

                # Generate the JD — use variant seed for diversity
                seed = _JD_VARIANT_SEEDS[(jd_idx - 1) % len(_JD_VARIANT_SEEDS)]
                logging.info("Generating JD: %s variant %d (%s) …", role, jd_idx, seed["seniority"])
                try:
                    jd_data = call_openai_json(
                        client,
                        model=args.model,
                        system_prompt=JD_SYSTEM_PROMPT,
                        user_prompt=JD_USER_PROMPT.format(
                            role=role,
                            focus=focus,
                            seniority=seed["seniority"],
                            exp=seed["exp"],
                            twist=seed["twist"],
                        ),
                        temperature=args.temperature,
                        max_retries=args.max_retries,
                        retry_delay=args.retry_delay,
                    )
                except RuntimeError as exc:
                    logging.error("Failed to generate JD for %s variant %d: %s", role, jd_idx, exc)
                    continue
                time.sleep(args.request_delay)

                jd_text = str(jd_data.get("jd_text") or "")
                if len(jd_text) < 100:
                    logging.warning("JD text too short for %s variant %d — skipping.", role, jd_idx)
                    continue

                jd_record: dict[str, Any] = {
                    "record_id": f"synthetic_jd_{jd_pair_id_prefix}",
                    "document_type": "jd",
                    "text": jd_text,
                    "role": role,
                    "jd_variant": jd_idx,
                    "required_experience_years": jd_data.get("required_experience_years"),
                    "required_degree": jd_data.get("required_degree"),
                    "key_technologies": jd_data.get("key_technologies") or [],
                    "seniority": jd_data.get("seniority"),
                    "qualification_facts": _build_jd_facts(jd_data),
                }

                # Generate CVs for each relevance level
                for level in relevance_levels:
                    pair_id = f"{jd_pair_id_prefix}_lv{level}"
                    if pair_id in existing_ids:
                        logging.debug("Skipping %s (already done).", pair_id)
                        skipped += 1
                        continue

                    logging.info("  Generating CV level %d for %s variant %d …", level, role, jd_idx)
                    try:
                        cv_data = call_openai_json(
                            client,
                            model=args.model,
                            system_prompt=CV_SYSTEM_PROMPT,
                            user_prompt=CV_USER_PROMPT.format(
                                level=level,
                                jd_text=jd_text[:1500],  # keep prompt under token limit
                                level_description=RELEVANCE_DESCRIPTIONS[level],
                            ),
                            temperature=args.temperature,
                            max_retries=args.max_retries,
                            retry_delay=args.retry_delay,
                        )
                    except RuntimeError as exc:
                        logging.error("Failed CV level %d for %s: %s", level, pair_id, exc)
                        continue
                    time.sleep(args.request_delay)

                    cv_text = str(cv_data.get("cv_text") or "")
                    if len(cv_text) < 80:
                        logging.warning("CV text too short for %s — skipping.", pair_id)
                        continue

                    pair_row: dict[str, Any] = {
                        "pair_id": pair_id,
                        "cv_id": f"synthetic_cv_{pair_id}",
                        "jd_id": jd_record["record_id"],
                        "relevance_label": level,
                        "relevance_reason": str(cv_data.get("relevance_reason") or ""),
                        "candidate_name": str(cv_data.get("candidate_name") or ""),
                        "role": role,
                        "jd_variant": jd_idx,
                        "cv_record": {
                            "record_id": f"synthetic_cv_{pair_id}",
                            "document_type": "cv",
                            "text": cv_text,
                            # qualification_facts populated from LLM-extracted structured fields
                            "qualification_facts": _build_cv_facts(cv_data),
                        },
                        "jd_record": jd_record,
                    }

                    out_handle.write(json.dumps(pair_row, ensure_ascii=False) + "\n")
                    out_handle.flush()
                    generated += 1
                    existing_ids.add(pair_id)
                    logging.info(
                        "  [%d/%d] Generated pair %s (relevance=%d)",
                        generated + skipped,
                        total_pairs,
                        pair_id,
                        level,
                    )

    logging.info(
        "Done. Generated %d new pairs, skipped %d. Output: %s",
        generated,
        skipped,
        args.output,
    )
    return 0


def _build_jd_facts(jd_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert structured JD metadata into qualification_facts format."""
    facts: list[dict[str, Any]] = []
    exp = jd_data.get("required_experience_years")
    if exp is not None:
        try:
            facts.append({
                "fact_type": "EXPERIENCE_YEARS",
                "text": f"{int(exp)} years of experience",
                "value": str(int(exp)),
                "unit": "years",
                "normalized": f"{int(exp)} years of experience",
                "operator": ">=",
                "is_mandatory": True,
            })
        except (TypeError, ValueError):
            pass
    degree = str(jd_data.get("required_degree") or "").strip()
    if degree and degree.lower() not in ("", "none"):
        facts.append({
            "fact_type": "DEGREE",
            "text": degree,
            "normalized": degree.lower(),
            "is_mandatory": True,
        })
    for cert in jd_data.get("required_certifications") or []:
        cert_text = str(cert).strip()
        if cert_text:
            facts.append({
                "fact_type": "CERTIFICATION",
                "text": cert_text,
                "normalized": cert_text.lower(),
                "is_mandatory": True,
            })
    return facts


def _build_cv_facts(cv_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build qualification_facts from LLM-extracted CV structured fields."""
    facts: list[dict[str, Any]] = []
    exp = cv_data.get("candidate_experience_years")
    if exp is not None:
        try:
            facts.append({
                "fact_type": "EXPERIENCE_YEARS",
                "text": f"{int(exp)} years of experience",
                "value": str(int(exp)),
                "unit": "years",
                "normalized": f"{int(exp)} years of experience",
                "operator": "=",
                "is_mandatory": False,
            })
        except (TypeError, ValueError):
            pass
    degree = str(cv_data.get("candidate_degree") or "").strip()
    if degree and degree.lower() not in ("", "none"):
        facts.append({
            "fact_type": "DEGREE",
            "text": degree,
            "normalized": degree.lower(),
            "is_mandatory": False,
        })
    for cert in cv_data.get("candidate_certifications") or []:
        cert_text = str(cert).strip()
        if cert_text:
            facts.append({
                "fact_type": "CERTIFICATION",
                "text": cert_text,
                "normalized": cert_text.lower(),
                "is_mandatory": False,
            })
    return facts


if __name__ == "__main__":
    raise SystemExit(main())
