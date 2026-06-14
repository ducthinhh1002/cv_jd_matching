"""Fix collapsed tech-stack diversity for Software Engineer and AR/VR Developer.

Removes existing pairs for these 2 roles from the dataset, then regenerates them
with explicitly pinned technology stacks per JD variant to guarantee diversity.

Usage:
  python fix_collapsed_roles.py --api-key sk-...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import shared helpers from the main generator
# ---------------------------------------------------------------------------
from generate_synthetic_eval import (
    CV_SYSTEM_PROMPT,
    CV_USER_PROMPT,
    RELEVANCE_DESCRIPTIONS,
    _build_cv_facts,
    _build_jd_facts,
    call_openai_json,
    configure_logging,
    get_openai_client,
    load_existing_pair_ids,
)

# ---------------------------------------------------------------------------
# Pre-pinned tech stacks — one unique set per variant, guaranteed distinct
# ---------------------------------------------------------------------------

FIXED_ROLE_VARIANTS: dict[str, list[dict[str, Any]]] = {
    "Software Engineer": [
        # (seniority, exp, twist, stack)
        {"seniority": "junior",  "exp": "1-2",  "twist": "startup, learning velocity matters",
         "stack": ["Python", "FastAPI", "PostgreSQL", "Redis"],
         "desc": "backend REST APIs with Python"},
        {"seniority": "mid",     "exp": "3-4",  "twist": "product company, code quality focus",
         "stack": ["Java", "Spring Boot", "MySQL", "Kafka"],
         "desc": "Java microservices and event-driven architecture"},
        {"seniority": "senior",  "exp": "5-7",  "twist": "fintech, compliance and security",
         "stack": ["Go", "Gin", "PostgreSQL", "gRPC"],
         "desc": "high-performance Go services with gRPC"},
        {"seniority": "lead",    "exp": "7-10", "twist": "team lead, mentoring focus",
         "stack": ["C#", ".NET", "SQL Server", "Azure Service Bus"],
         "desc": ".NET platform and Azure cloud"},
        {"seniority": "mid",     "exp": "3-5",  "twist": "remote-first, async collaboration",
         "stack": ["Node.js", "NestJS", "MongoDB", "GraphQL"],
         "desc": "Node.js GraphQL backend"},
        {"seniority": "senior",  "exp": "5-8",  "twist": "e-commerce, high-traffic systems",
         "stack": ["Kotlin", "Ktor", "Cassandra", "RabbitMQ"],
         "desc": "Kotlin server-side with Cassandra"},
        {"seniority": "junior",  "exp": "0-2",  "twist": "fresh grad welcome, strong CS fundamentals",
         "stack": ["Ruby", "Ruby on Rails", "PostgreSQL", "Sidekiq"],
         "desc": "Ruby on Rails full-stack development"},
        {"seniority": "senior",  "exp": "6-9",  "twist": "enterprise, legacy modernisation",
         "stack": ["Scala", "Akka", "Elasticsearch", "Apache Spark"],
         "desc": "Scala distributed backend with Akka"},
        {"seniority": "mid",     "exp": "2-4",  "twist": "AI product company, ML pipelines",
         "stack": ["Python", "Django", "Celery", "AWS Lambda"],
         "desc": "Python Django with serverless AWS"},
        {"seniority": "lead",    "exp": "8+",   "twist": "architect track, cross-team ownership",
         "stack": ["Rust", "Actix-Web", "PostgreSQL", "Kubernetes"],
         "desc": "systems-level Rust backend"},
    ],
    "AR/VR Developer": [
        {"seniority": "junior",  "exp": "1-2",  "twist": "gaming startup, ship fast",
         "stack": ["Unity", "C#", "ARCore", "Blender"],
         "desc": "Unity-based mobile AR games"},
        {"seniority": "mid",     "exp": "3-4",  "twist": "enterprise XR training platform",
         "stack": ["Unreal Engine", "C++", "MetaXR SDK", "Lumen"],
         "desc": "Unreal Engine enterprise VR training"},
        {"seniority": "senior",  "exp": "5-7",  "twist": "medical simulation, precision required",
         "stack": ["Unity", "C#", "OpenXR", "MRTK"],
         "desc": "medical AR/VR simulation with MRTK"},
        {"seniority": "lead",    "exp": "7-10", "twist": "WebXR platform, browser-first",
         "stack": ["Three.js", "WebXR API", "JavaScript", "Babylon.js"],
         "desc": "browser-based WebXR experiences"},
        {"seniority": "mid",     "exp": "3-5",  "twist": "retail virtual try-on",
         "stack": ["ARKit", "Swift", "RealityKit", "SceneKit"],
         "desc": "iOS AR shopping experience with ARKit"},
        {"seniority": "senior",  "exp": "5-8",  "twist": "industrial digital twin",
         "stack": ["Unreal Engine", "C++", "NVIDIA Omniverse", "USD"],
         "desc": "industrial digital twins with Omniverse"},
        {"seniority": "junior",  "exp": "0-2",  "twist": "education platform, fresh grad welcome",
         "stack": ["Unity", "C#", "Vuforia", "Photon SDK"],
         "desc": "educational AR apps with Vuforia"},
        {"seniority": "senior",  "exp": "6-9",  "twist": "social VR, multiplayer",
         "stack": ["Unreal Engine", "Blueprint", "Mirror Networking", "Vivox"],
         "desc": "social VR with multiplayer networking"},
        {"seniority": "mid",     "exp": "2-4",  "twist": "spatial computing, Apple Vision Pro",
         "stack": ["visionOS", "Swift", "RealityKit", "SwiftUI"],
         "desc": "spatial computing apps for Apple Vision Pro"},
        {"seniority": "lead",    "exp": "8+",   "twist": "platform architect, SDK design",
         "stack": ["C++", "OpenXR", "Vulkan", "HLSL"],
         "desc": "low-level XR runtime and graphics programming"},
    ],
}

JD_SYSTEM_PROMPT = (
    "You are a professional recruitment content writer with 10 years of experience "
    "writing IT job descriptions. Return ONLY valid JSON, no markdown fences."
)

JD_USER_PROMPT_FIXED = """\
Write a realistic, detailed job description for a {role} position.
Focus area: {desc}

Variant context:
- Seniority: {seniority}
- Experience required: {exp} years
- Company context: {twist}

REQUIRED technology stack (use EXACTLY these technologies, do not substitute):
{stack_str}

Return JSON:
{{
  "jd_text": "<400-600 word job description: About the Role, Key Requirements, Nice to Have, Benefits>",
  "required_experience_years": <integer matching the variant context>,
  "required_degree": "<one of: none, Bachelor, Master, PhD>",
  "required_certifications": ["<cert or empty list>"],
  "key_technologies": {stack_json},
  "seniority": "{seniority}"
}}"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix collapsed roles in synthetic eval dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/synthetic_eval_dataset.jsonl"),
        help="Path to the existing synthetic eval dataset.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--roles", nargs="+", default=["Software Engineer", "AR/VR Developer"])
    parser.add_argument("--relevance-levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--request-delay", type=float, default=0.6)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    target_roles = set(args.roles)
    relevance_levels = sorted(set(int(l) for l in args.relevance_levels if 1 <= int(l) <= 5))

    # --- Step 1: Load existing dataset, partition into keep / remove ---
    logging.info("Loading existing dataset: %s", args.dataset)
    all_rows = [
        json.loads(l)
        for l in args.dataset.read_text("utf-8").splitlines()
        if l.strip()
    ]
    keep_rows = [r for r in all_rows if r.get("role") not in target_roles]
    drop_rows = [r for r in all_rows if r.get("role") in target_roles]
    logging.info(
        "Keeping %d rows, dropping %d rows (%s)",
        len(keep_rows), len(drop_rows), ", ".join(sorted(target_roles)),
    )

    # --- Step 2: Write back kept rows ---
    tmp_path = args.dataset.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        for r in keep_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- Step 3: Regenerate dropped roles ---
    client = get_openai_client(args.api_key)
    existing_ids = load_existing_pair_ids(tmp_path)
    generated = 0

    with tmp_path.open("a", encoding="utf-8") as out_handle:
        for role in sorted(target_roles):
            variants = FIXED_ROLE_VARIANTS.get(role)
            if not variants:
                logging.warning("No fixed variants defined for role %r — skipping.", role)
                continue

            role_slug = role.lower().replace(" ", "_").replace("/", "_")

            for jd_idx, variant in enumerate(variants, start=1):
                jd_pair_id_prefix = f"{role_slug}_jd{jd_idx}"
                all_cv_ids = {f"{jd_pair_id_prefix}_lv{lvl}" for lvl in relevance_levels}
                if all_cv_ids.issubset(existing_ids):
                    logging.info("Skipping %s (all levels done).", jd_pair_id_prefix)
                    continue

                stack = variant["stack"]
                stack_str = "\n".join(f"  - {t}" for t in stack)
                stack_json = json.dumps(stack)

                logging.info(
                    "Generating JD: %s variant %d (%s / %s) ...",
                    role, jd_idx, variant["seniority"], stack[0],
                )
                try:
                    jd_data = call_openai_json(
                        client,
                        model=args.model,
                        system_prompt=JD_SYSTEM_PROMPT,
                        user_prompt=JD_USER_PROMPT_FIXED.format(
                            role=role,
                            desc=variant["desc"],
                            seniority=variant["seniority"],
                            exp=variant["exp"],
                            twist=variant["twist"],
                            stack_str=stack_str,
                            stack_json=stack_json,
                        ),
                        temperature=args.temperature,
                        max_retries=args.max_retries,
                        retry_delay=args.retry_delay,
                    )
                except RuntimeError as exc:
                    logging.error("Failed JD for %s v%d: %s", role, jd_idx, exc)
                    continue
                time.sleep(args.request_delay)

                jd_text = str(jd_data.get("jd_text") or "")
                if len(jd_text) < 100:
                    logging.warning("JD too short for %s v%d — skipping.", role, jd_idx)
                    continue

                # Force key_technologies to our pinned stack
                jd_data["key_technologies"] = stack

                jd_record: dict[str, Any] = {
                    "record_id": f"synthetic_jd_{jd_pair_id_prefix}",
                    "document_type": "jd",
                    "text": jd_text,
                    "role": role,
                    "jd_variant": jd_idx,
                    "required_experience_years": jd_data.get("required_experience_years"),
                    "required_degree": jd_data.get("required_degree"),
                    "required_certifications": jd_data.get("required_certifications") or [],
                    "key_technologies": stack,
                    "seniority": variant["seniority"],
                    "qualification_facts": _build_jd_facts(jd_data),
                }

                for level in relevance_levels:
                    pair_id = f"{jd_pair_id_prefix}_lv{level}"
                    if pair_id in existing_ids:
                        logging.debug("Skipping %s (already done).", pair_id)
                        continue

                    logging.info("  Generating CV level %d for %s v%d ...", level, role, jd_idx)
                    try:
                        cv_data = call_openai_json(
                            client,
                            model=args.model,
                            system_prompt=CV_SYSTEM_PROMPT,
                            user_prompt=CV_USER_PROMPT.format(
                                level=level,
                                jd_text=jd_text[:1500],
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
                        logging.warning("CV too short for %s — skipping.", pair_id)
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
                            "qualification_facts": _build_cv_facts(cv_data),
                        },
                        "jd_record": jd_record,
                    }

                    out_handle.write(json.dumps(pair_row, ensure_ascii=False) + "\n")
                    out_handle.flush()
                    generated += 1
                    existing_ids.add(pair_id)
                    logging.info(
                        "  [+%d] %s  level=%d  stack=%s",
                        generated, pair_id, level, stack[0],
                    )

    # --- Step 4: Replace original file with fixed one ---
    tmp_path.replace(args.dataset)
    logging.info("Done. Regenerated %d pairs. Dataset updated: %s", generated, args.dataset)

    # --- Step 5: Quick diversity check ---
    rows_final = [
        json.loads(l)
        for l in args.dataset.read_text("utf-8").splitlines()
        if l.strip()
    ]
    logging.info("Final dataset size: %d pairs", len(rows_final))
    for role in sorted(target_roles):
        from collections import defaultdict
        tech_sets: set[tuple[str, ...]] = set()
        for r in rows_final:
            if r.get("role") == role:
                techs = tuple(sorted(t.lower() for t in r["jd_record"].get("key_technologies", [])))
                tech_sets.add(techs)
        logging.info("  %s: %d unique tech sets", role, len(tech_sets))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
