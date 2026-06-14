from __future__ import annotations

import argparse
import cgi
import io
import json
import logging
import math
import mimetypes
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote
from urllib import error as urllib_error
from urllib import request as urllib_request

import torch

from prepare_external_benchmark import infer_qualification_facts
from score_candidates import (
    DocumentEmbedder,
    compute_hard_constraint_score,
    compute_hard_penalty_factor,
    compute_role_alignment_score,
    compute_seniority_score,
    compute_tech_skill_features,
)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web"
BUNDLED_PYTHON_PACKAGES = (
    Path.home()
    / ".cache"
    / "codex-runtimes"
    / "codex-primary-runtime"
    / "dependencies"
    / "python"
)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:160]
    return ""


def recommendation(score: float) -> str:
    if score >= 0.78:
        return "Strong recommend"
    if score >= 0.62:
        return "Consider"
    return "Do not recommend"


def decision(score: float) -> str:
    return "Recommend to interview" if score >= 0.62 else "Do not recommend"


def fit_status(score: float, *, high: float = 0.75, mid: float = 0.5) -> str:
    if score >= high:
        return "pass"
    if score >= mid:
        return "partial"
    return "gap"


def build_fit_checklist(
    *,
    semantic_score: float,
    required_coverage: float,
    tech_score: float,
    role_score: float,
    seniority_score: float,
    hard_score: float,
    hard_mandatory: bool,
) -> list[dict[str, str]]:
    return [
        {
            "label": "Semantic fit",
            "status": fit_status(semantic_score, high=0.72, mid=0.62),
            "detail": f"CV/JD contextual similarity: {round(semantic_score * 100)}%.",
        },
        {
            "label": "Required skills",
            "status": fit_status(required_coverage, high=0.75, mid=0.5),
            "detail": f"Required technical skill coverage: {round(required_coverage * 100)}%.",
        },
        {
            "label": "Tech stack overlap",
            "status": fit_status(tech_score, high=0.45, mid=0.2),
            "detail": f"Normalized stack overlap: {round(tech_score * 100)}%.",
        },
        {
            "label": "Role alignment",
            "status": fit_status(role_score, high=0.7, mid=0.5),
            "detail": f"Role-family alignment score: {round(role_score * 100)}%.",
        },
        {
            "label": "Seniority",
            "status": fit_status(seniority_score, high=0.7, mid=0.5),
            "detail": f"Seniority match score: {round(seniority_score * 100)}%.",
        },
        {
            "label": "Hard requirements",
            "status": fit_status(hard_score, high=0.75, mid=0.55) if hard_mandatory else "pass",
            "detail": (
                f"Mandatory requirement satisfaction: {round(hard_score * 100)}%."
                if hard_mandatory
                else "No mandatory hard constraint penalty detected."
            ),
        },
    ]


def build_risk_flags(
    *,
    missing_skills: list[str],
    required_coverage: float,
    role_score: float,
    seniority_score: float,
    hard_score: float,
    hard_mandatory: bool,
) -> list[str]:
    flags: list[str] = []
    if required_coverage < 0.5 and missing_skills:
        flags.append("Major required-skill gap")
    elif missing_skills:
        flags.append("Some required skills are not evidenced")
    if role_score < 0.5:
        flags.append("Possible role-family mismatch")
    if seniority_score < 0.7:
        flags.append("Seniority should be verified")
    if hard_mandatory and hard_score < 0.7:
        flags.append("Mandatory requirement evidence is incomplete")
    return flags


def build_interview_questions(
    *,
    missing_skills: list[str],
    matched_skills: list[str],
    role_score: float,
    seniority_score: float,
    hard_mandatory: bool,
    hard_score: float,
) -> list[str]:
    questions: list[str] = []
    for skill in missing_skills[:3]:
        questions.append(f"Can you describe hands-on experience with {skill} in a production project?")
    if matched_skills:
        questions.append(
            f"Which project best demonstrates your experience with {', '.join(matched_skills[:3])}?"
        )
    if role_score < 0.7:
        questions.append("How does your recent work map to the core responsibilities of this role?")
    if seniority_score < 0.7:
        questions.append("Can you clarify the level of ownership, leadership, and delivery scope in your recent projects?")
    if hard_mandatory and hard_score < 0.8:
        questions.append("Can you provide evidence for the mandatory experience, education, or certification requirements?")
    if len(questions) < 3:
        questions.extend(
            [
                "Please walk through the most relevant project in your CV and explain your exact responsibilities.",
                "What technical decision in that project had the largest impact, and how did you evaluate alternatives?",
                "Describe a production issue or delivery risk you handled and what you changed afterward.",
            ]
        )
    return questions[:5]


def build_fit_summary(
    *,
    score: float,
    matched_skills: list[str],
    missing_skills: list[str],
    risk_flags: list[str],
) -> str:
    if score >= 0.78:
        opening = "Strong candidate for the shortlist"
    elif score >= 0.62:
        opening = "Viable candidate with points to verify"
    else:
        opening = "Weak match for the current JD"
    matched = f"matches {', '.join(matched_skills[:4])}" if matched_skills else "has limited explicit stack overlap"
    missing = f"; gaps: {', '.join(missing_skills[:3])}" if missing_skills else "; no major required-skill gap detected"
    risk = f"; risks: {', '.join(risk_flags[:2])}" if risk_flags else ""
    return f"{opening}: {matched}{missing}{risk}."


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


class DeepSeekQuestionGenerator:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        api_url: str,
        timeout_seconds: float,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        *,
        job_description: str,
        candidate_name: str,
        candidate_text: str,
        fit_summary: str,
        matched_skills: list[str],
        missing_skills: list[str],
        risk_flags: list[str],
        fallback_questions: list[str],
    ) -> list[str]:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 360,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate recruiter interview questions for CV-JD matching. "
                        "Return only compact JSON with key 'questions'. "
                        "Use evidence from the JD/CV and avoid inventing technologies not present in the inputs."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "Generate 3 to 5 concise interview questions.",
                            "candidate_name": candidate_name,
                            "fit_summary": fit_summary,
                            "matched_skills": matched_skills[:8],
                            "missing_skills": missing_skills[:8],
                            "risk_flags": risk_flags[:6],
                            "job_description_excerpt": job_description[:1800],
                            "candidate_cv_excerpt": candidate_text[:2200],
                            "question_style": (
                                "practical, evidence-seeking, recruiter-facing, "
                                "focused on missing skills, matched strengths, seniority, and hard requirements"
                            ),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        request = urllib_request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logging.warning("DeepSeek interview-question generation failed: %s", exc)
            return fallback_questions

        content = stringify(data.get("choices", [{}])[0].get("message", {}).get("content"))
        parsed = extract_json_object(content)
        questions = parsed.get("questions") if parsed else None
        if not isinstance(questions, list):
            logging.warning("DeepSeek response did not contain a questions list.")
            return fallback_questions

        cleaned = [
            stringify(question).strip()
            for question in questions
            if stringify(question).strip()
        ][:5]
        return cleaned or fallback_questions


def make_row(record_id: str, document_type: str, text: str) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "document_type": document_type,
        "source_job_title": first_line(text),
        "text": text,
        "qualification_facts": infer_qualification_facts(text),
        "entities": [],
        "onet_mappings": [],
    }


def ensure_document_packages_path() -> None:
    candidate_paths = [
        BUNDLED_PYTHON_PACKAGES,
        BUNDLED_PYTHON_PACKAGES / "Lib",
        BUNDLED_PYTHON_PACKAGES / "DLLs",
        BUNDLED_PYTHON_PACKAGES / "Lib" / "site-packages",
    ]
    for path in candidate_paths:
        if path.exists():
            package_path = str(path)
            if package_path not in sys.path:
                sys.path.append(package_path)


def compact_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def extract_pdf_text(data: bytes) -> str:
    ensure_document_packages_path()
    errors: list[str] = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = compact_text("\n".join(pages))
        if text:
            return text
        errors.append("pypdf extracted no text")
    except Exception as exc:
        errors.append(f"pypdf failed: {exc}")

    try:
        from pdfminer.high_level import extract_text

        text = compact_text(extract_text(io.BytesIO(data)) or "")
        if text:
            return text
        errors.append("pdfminer extracted no text")
    except ImportError:
        errors.append("pdfminer.six is not installed")
    except Exception as exc:
        errors.append(f"pdfminer failed: {exc}")

    detail = "; ".join(errors)
    raise ValueError(
        "Could not extract readable text from the PDF. "
        "The file may be scanned/image-only, protected, or malformed. "
        f"Details: {detail}"
    )


def extract_docx_text(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml_data = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("Invalid DOCX file.") from exc

    root = ET.fromstring(xml_data)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    lines: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        pieces = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        line = "".join(pieces).strip()
        if line:
            lines.append(line)
    return compact_text("\n".join(lines))


def extract_plain_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return compact_text(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    return compact_text(data.decode("utf-8", errors="replace"))


def extract_text_from_upload(filename: str, content_type: str, data: bytes) -> str:
    if not data:
        raise ValueError(f"{filename or 'uploaded file'} is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise ValueError(f"{filename or 'uploaded file'} is larger than {limit_mb} MB.")

    suffix = Path(filename or "").suffix.lower()
    normalized_type = (content_type or "").split(";", 1)[0].lower()
    if suffix == ".pdf" or normalized_type == "application/pdf":
        text = extract_pdf_text(data)
    elif suffix == ".docx" or normalized_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_docx_text(data)
    elif suffix in {".txt", ".text"} or normalized_type.startswith("text/"):
        text = extract_plain_text(data)
    else:
        raise ValueError(f"Unsupported file type for {filename or 'uploaded file'}. Use PDF, DOCX, or TXT.")

    if not text.strip():
        raise ValueError(f"Could not extract readable text from {filename or 'uploaded file'}.")
    return text


def candidate_name_from_filename(filename: str, fallback: str) -> str:
    stem = Path(filename or "").stem.strip()
    return stem or fallback


def parse_textarea_resumes(raw: str) -> list[dict[str, str]]:
    resumes: list[dict[str, str]] = []
    for index, block in enumerate(re.split(r"^---CV---$", raw, flags=re.IGNORECASE | re.MULTILINE), start=1):
        text = block.strip()
        if not text:
            continue
        lines = text.splitlines()
        first = lines[0].strip() if lines else f"Candidate {index}"
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
        resumes.append(
            {
                "id": f"text_cv_{index}",
                "name": first or f"Candidate {index}",
                "text": body or text,
            }
        )
    return resumes


class DemoRanker:
    def __init__(
        self,
        *,
        embedding_model: str,
        embedding_device: str | None,
        embedding_batch_size: int,
        hard_penalty_strength: float,
        question_generator: DeepSeekQuestionGenerator | None = None,
    ) -> None:
        self.embedder = DocumentEmbedder(
            embedding_model,
            device=embedding_device,
            batch_size=embedding_batch_size,
        )
        self.hard_penalty_strength = hard_penalty_strength
        self.question_generator = question_generator

    def rank(self, job_description: str, resumes: list[dict[str, str]]) -> dict[str, Any]:
        jd_text = job_description.strip()
        if not jd_text:
            raise ValueError("Job description is required.")
        normalized_resumes = [
            {
                "id": stringify(item.get("id") or f"cv_{index + 1}"),
                "name": stringify(item.get("name") or f"Candidate {index + 1}"),
                "text": stringify(item.get("text")).strip(),
            }
            for index, item in enumerate(resumes)
            if stringify(item.get("text")).strip()
        ]
        if not normalized_resumes:
            raise ValueError("At least one CV/resume is required.")

        jd_row = make_row("jd_input", "jd", jd_text)
        cv_rows = [
            make_row(item["id"], "cv", item["text"])
            for item in normalized_resumes
        ]

        texts = [jd_text[:2000]] + [row["text"][:2000] for row in cv_rows]
        embeddings = self.embedder.encode(texts)
        jd_embedding = embeddings[0]

        results: list[dict[str, Any]] = []
        for item, cv_row, cv_embedding in zip(normalized_resumes, cv_rows, embeddings[1:], strict=False):
            semantic = float(torch.dot(cv_embedding, jd_embedding).clamp(-1.0, 1.0))
            semantic_score = clamp01((semantic + 1.0) / 2.0)

            skill_features = compute_tech_skill_features(cv_row, jd_row)
            role_score, cv_roles, jd_roles = compute_role_alignment_score(cv_row, jd_row)
            seniority_score, cv_seniority, jd_seniority = compute_seniority_score(cv_row, jd_row)
            hard_score = compute_hard_constraint_score(cv_row, jd_row)
            hard_penalty, hard_count, hard_mandatory = compute_hard_penalty_factor(
                hard_score=hard_score,
                jd_row=jd_row,
                strength=self.hard_penalty_strength,
            )

            tech_score = float(skill_features["tech_skill_score"])
            required_coverage = float(skill_features["required_skill_coverage"])
            base_score = (
                0.50 * semantic_score
                + 0.22 * required_coverage
                + 0.12 * role_score
                + 0.08 * seniority_score
                + 0.08 * tech_score
            )
            final_score = base_score * hard_penalty if hard_mandatory else base_score
            final_score = clamp01(final_score)
            matched_skills = skill_features["matched_tech_skills"]
            missing_skills = skill_features["missing_required_skills"]
            fit_checklist = build_fit_checklist(
                semantic_score=semantic_score,
                required_coverage=required_coverage,
                tech_score=tech_score,
                role_score=role_score,
                seniority_score=seniority_score,
                hard_score=hard_score,
                hard_mandatory=hard_mandatory,
            )
            risk_flags = build_risk_flags(
                missing_skills=missing_skills,
                required_coverage=required_coverage,
                role_score=role_score,
                seniority_score=seniority_score,
                hard_score=hard_score,
                hard_mandatory=hard_mandatory,
            )
            interview_questions = build_interview_questions(
                missing_skills=missing_skills,
                matched_skills=matched_skills,
                role_score=role_score,
                seniority_score=seniority_score,
                hard_mandatory=hard_mandatory,
                hard_score=hard_score,
            )
            fit_summary = build_fit_summary(
                score=final_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                risk_flags=risk_flags,
            )
            if self.question_generator is not None:
                interview_questions = self.question_generator.generate(
                    job_description=jd_text,
                    candidate_name=item["name"],
                    candidate_text=item["text"],
                    fit_summary=fit_summary,
                    matched_skills=matched_skills,
                    missing_skills=missing_skills,
                    risk_flags=risk_flags,
                    fallback_questions=interview_questions,
                )

            results.append(
                {
                    "candidate_id": item["id"],
                    "candidate_name": item["name"],
                    "score": round(final_score, 6),
                    "recommendation": recommendation(final_score),
                    "decision": decision(final_score),
                    "semantic_score": round(semantic_score, 6),
                    "required_skill_coverage": round(required_coverage, 6),
                    "tech_skill_score": round(tech_score, 6),
                    "role_alignment_score": round(role_score, 6),
                    "seniority_score": round(seniority_score, 6),
                    "hard_constraint_score": round(hard_score, 6),
                    "hard_penalty_factor": round(hard_penalty, 6),
                    "hard_constraints_mandatory": hard_mandatory,
                    "hard_constraint_count": hard_count,
                    "fit_summary": fit_summary,
                    "fit_checklist": fit_checklist,
                    "risk_flags": risk_flags,
                    "interview_questions": interview_questions,
                    "matched_tech_skills": matched_skills,
                    "missing_required_skills": missing_skills,
                    "cv_tech_skill_count": skill_features["cv_tech_skill_count"],
                    "jd_tech_skill_count": skill_features["jd_tech_skill_count"],
                    "cv_roles": sorted(cv_roles),
                    "jd_roles": sorted(jd_roles),
                    "cv_seniority": cv_seniority,
                    "jd_seniority": jd_seniority,
                }
            )

        results.sort(key=lambda row: row["score"], reverse=True)
        for rank, row in enumerate(results, start=1):
            row["rank"] = rank
        return {
            "job_title": first_line(jd_text),
            "num_candidates": len(results),
            "ranking": results,
        }


class DemoRequestHandler(BaseHTTPRequestHandler):
    ranker: DemoRanker

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_multipart_form(self) -> cgi.FieldStorage:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            raise ValueError("Expected multipart/form-data request.")
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_UPLOAD_BYTES * 12:
            raise ValueError("Upload request is too large.")
        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(length),
            },
            keep_blank_values=True,
        )

    @staticmethod
    def form_files(form: cgi.FieldStorage, name: str) -> list[cgi.FieldStorage]:
        value = form[name] if name in form else []
        items = value if isinstance(value, list) else [value]
        return [
            item
            for item in items
            if getattr(item, "filename", None) and getattr(item, "file", None) is not None
        ]

    @staticmethod
    def form_text(form: cgi.FieldStorage, name: str) -> str:
        return stringify(form.getfirst(name, "")).strip()

    def handle_rank_json(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        result = self.ranker.rank(
            stringify(payload.get("job_description")),
            payload.get("resumes") or [],
        )
        self.send_json(result)

    def handle_rank_files(self) -> None:
        form = self.read_multipart_form()
        parsed_files: list[dict[str, Any]] = []

        jd_text = self.form_text(form, "job_description")
        jd_files = self.form_files(form, "jd_file")
        if jd_files:
            jd_file = jd_files[0]
            jd_text = extract_text_from_upload(
                stringify(jd_file.filename),
                stringify(jd_file.type),
                jd_file.file.read(),
            )
            parsed_files.append(
                {
                    "field": "jd_file",
                    "filename": stringify(jd_file.filename),
                    "characters": len(jd_text),
                }
            )

        resumes = []
        cv_text = self.form_text(form, "cv_text")
        if cv_text:
            resumes.extend(parse_textarea_resumes(cv_text))

        for index, cv_file in enumerate(self.form_files(form, "cv_files"), start=1):
            filename = stringify(cv_file.filename)
            text = extract_text_from_upload(filename, stringify(cv_file.type), cv_file.file.read())
            resumes.append(
                {
                    "id": f"uploaded_cv_{index}",
                    "name": candidate_name_from_filename(filename, f"Uploaded CV {index}"),
                    "text": text,
                }
            )
            parsed_files.append(
                {
                    "field": "cv_files",
                    "filename": filename,
                    "characters": len(text),
                }
            )

        result = self.ranker.rank(jd_text, resumes)
        result["parsed_files"] = parsed_files
        self.send_json(result)

    def do_GET(self) -> None:
        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            path = "/index.html"
        file_path = (STATIC_DIR / path.lstrip("/")).resolve()
        try:
            file_path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        if file_path.suffix in {".html", ".css", ".js"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if self.path not in {"/api/rank", "/api/rank-files"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            if self.path == "/api/rank-files":
                self.handle_rank_files()
            else:
                self.handle_rank_json()
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except json.JSONDecodeError as exc:
            self.send_json({"error": f"Invalid JSON request: {exc}"}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - defensive server guard
            logging.exception("Request failed")
            self.send_json({"error": f"Internal server error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CV-JD matching demo web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-device", default=None, help="cpu/cuda. Auto-detected if omitted.")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--hard-penalty-strength", type=float, default=0.35)
    parser.set_defaults(use_llm_questions=True)
    parser.add_argument("--use-llm-questions", dest="use_llm_questions", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-llm-questions",
        dest="use_llm_questions",
        action="store_false",
        help="Disable DeepSeek interview-question generation and use rule-based questions only.",
    )
    parser.add_argument("--deepseek-model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument(
        "--deepseek-api-url",
        default=os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions"),
    )
    parser.add_argument("--llm-question-timeout", type=float, default=20.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    if not STATIC_DIR.exists():
        raise SystemExit(f"Static web directory not found: {STATIC_DIR}")

    question_generator = None
    if args.use_llm_questions:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            logging.warning("DEEPSEEK_API_KEY is not set. Falling back to rule-based interview questions.")
        else:
            question_generator = DeepSeekQuestionGenerator(
                api_key=api_key,
                model=args.deepseek_model,
                api_url=args.deepseek_api_url,
                timeout_seconds=args.llm_question_timeout,
            )
            logging.info("LLM interview-question generation enabled with model=%s", args.deepseek_model)

    DemoRequestHandler.ranker = DemoRanker(
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        hard_penalty_strength=args.hard_penalty_strength,
        question_generator=question_generator,
    )
    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    logging.info("Serving demo app at http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
