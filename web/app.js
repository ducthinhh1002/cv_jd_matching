const form = document.querySelector("#match-form");
const jdInput = document.querySelector("#jd-input");
const cvInput = document.querySelector("#cv-input");
const resultTitle = document.querySelector("#result-title");
const statusPill = document.querySelector("#status-pill");
const summary = document.querySelector("#summary");
const results = document.querySelector("#results");
const rankButton = document.querySelector("#rank-button");
const clearButton = document.querySelector("#clear-button");
const jdPdfInput = document.querySelector("#jd-pdf-input");
const cvPdfInput = document.querySelector("#cv-pdf-input");
const jdPdfName = document.querySelector("#jd-pdf-name");
const cvPdfName = document.querySelector("#cv-pdf-name");
const thresholdInput = document.querySelector("#threshold-input");
const thresholdValue = document.querySelector("#threshold-value");
const shortlistOnly = document.querySelector("#shortlist-only");
const exportCsvButton = document.querySelector("#export-csv-button");
const copyShortlistButton = document.querySelector("#copy-shortlist-button");

let currentPayload = null;
const initialCvText = cvInput.value;

function hasSelectedFiles() {
  return Boolean(jdPdfInput.files?.length || cvPdfInput.files?.length);
}

function parseResumes(raw) {
  return raw
    .split(/^---CV---$/gim)
    .map((block, index) => {
      const text = block.trim();
      if (!text) return null;
      const lines = text.split(/\r?\n/);
      const firstLine = lines[0]?.trim() || `Candidate ${index + 1}`;
      const body = lines.length > 1 ? lines.slice(1).join("\n").trim() : text;
      return {
        id: `candidate_${index + 1}`,
        name: firstLine,
        text: body || text,
      };
    })
    .filter(Boolean);
}

function setStatus(label, state) {
  statusPill.textContent = label;
  statusPill.className = `status-pill ${state}`;
}

function pct(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function badgeClass(label) {
  if (label === "Strong recommend") return "";
  if (label === "Consider") return "warn";
  return "no";
}

function currentThreshold() {
  return Number(thresholdInput.value || 62) / 100;
}

function getVisibleRanking(payload) {
  const ranking = payload?.ranking || [];
  if (!shortlistOnly.checked) return ranking;
  const threshold = currentThreshold();
  return ranking.filter((item) => Number(item.score) >= threshold);
}

function updateThresholdLabel() {
  thresholdValue.textContent = `${thresholdInput.value}%`;
}

function renderSummary(payload) {
  const ranking = payload.ranking || [];
  const visible = getVisibleRanking(payload);
  const top = ranking[0];
  const threshold = currentThreshold();
  const aboveThreshold = ranking.filter((item) => Number(item.score) >= threshold).length;
  const averageScore = ranking.length
    ? ranking.reduce((sum, item) => sum + Number(item.score || 0), 0) / ranking.length
    : 0;
  const parsedFiles = payload.parsed_files?.length || 0;
  summary.innerHTML = `
    <article class="summary-card">
      <span>Candidates</span>
      <strong>${payload.num_candidates}</strong>
    </article>
    <article class="summary-card">
      <span>Top Score</span>
      <strong>${top ? pct(top.score) : "N/A"}</strong>
    </article>
    <article class="summary-card">
      <span>Above Threshold</span>
      <strong>${aboveThreshold}</strong>
    </article>
    <article class="summary-card">
      <span>Average Score</span>
      <strong>${pct(averageScore)}</strong>
    </article>
    <article class="summary-card">
      <span>Visible</span>
      <strong>${visible.length}</strong>
    </article>
    <article class="summary-card">
      <span>Top Decision</span>
      <strong>${top ? top.decision : "N/A"}</strong>
    </article>
    <article class="summary-card">
      <span>Parsed Files</span>
      <strong>${parsedFiles}</strong>
    </article>
  `;
}

function renderResults(payload) {
  const visibleRanking = getVisibleRanking(payload);
  const threshold = currentThreshold();
  if (!payload.ranking?.length) {
    results.className = "results empty-state";
    results.textContent = "Không có candidate hợp lệ.";
    return;
  }
  if (!visibleRanking.length) {
    results.className = "results empty-state";
    results.textContent = "Không có ứng viên nào đạt ngưỡng hiện tại.";
    return;
  }
  results.className = "results";
  results.innerHTML = visibleRanking
    .map((item) => {
      const missing = item.missing_required_skills?.length
        ? item.missing_required_skills.join(", ")
        : "None";
      const matched = item.matched_tech_skills?.length
        ? item.matched_tech_skills.join(", ")
        : "None";
      const risks = item.risk_flags?.length
        ? item.risk_flags.map((risk) => `<span>${escapeHtml(risk)}</span>`).join("")
        : "<span>No major risk flag</span>";
      const checklist = item.fit_checklist?.length
        ? item.fit_checklist.map((factor) => `
            <li class="${escapeHtml(factor.status)}">
              <strong>${escapeHtml(factor.label)}</strong>
              <small>${escapeHtml(factor.detail)}</small>
            </li>
          `).join("")
        : "";
      const questions = item.interview_questions?.length
        ? item.interview_questions.map((question) => `<li>${escapeHtml(question)}</li>`).join("")
        : "<li>No follow-up question generated.</li>";
      const passThreshold = Number(item.score) >= threshold;
      return `
        <article class="candidate">
          <div class="candidate-top">
            <div class="rank">#${item.rank}</div>
            <div>
              <h3>${escapeHtml(item.candidate_name)}</h3>
              <span class="badge ${badgeClass(item.recommendation)}">${item.recommendation}</span>
              <span class="threshold-badge ${passThreshold ? "" : "below"}">
                ${passThreshold ? "Đạt ngưỡng" : "Dưới ngưỡng"}
              </span>
            </div>
            <div class="score">
              <strong>${pct(item.score)}</strong>
              <small>${item.decision}</small>
            </div>
          </div>
          <div class="score-grid">
            ${scoreCell("Semantic", item.semantic_score)}
            ${scoreCell("Skill Coverage", item.required_skill_coverage)}
            ${scoreCell("Tech Match", item.tech_skill_score)}
            ${scoreCell("Role", item.role_alignment_score)}
            ${scoreCell("Seniority", item.seniority_score)}
          </div>
          <div class="skill-list">
            <div><strong>Matched skills:</strong> ${escapeHtml(matched)}</div>
            <div><strong>Missing required skills:</strong> ${escapeHtml(missing)}</div>
            <div><strong>JD roles:</strong> ${escapeHtml((item.jd_roles || []).join(", ") || "N/A")}</div>
          </div>
          <section class="review-pack">
            <div>
              <h4>Recruiter review</h4>
              <p>${escapeHtml(item.fit_summary || "No explanation generated.")}</p>
              <div class="risk-tags">${risks}</div>
            </div>
            <ul class="fit-checklist">${checklist}</ul>
          </section>
          <details class="candidate-details">
            <summary>Scoring details & interview questions</summary>
            <div class="detail-grid">
              <div><span>Hard constraint</span>${pct(item.hard_constraint_score)}</div>
              <div><span>Hard penalty</span>${pct(item.hard_penalty_factor)}</div>
              <div><span>Mandatory constraints</span>${item.hard_constraints_mandatory ? "Yes" : "No"}</div>
              <div><span>Constraint count</span>${item.hard_constraint_count}</div>
              <div><span>CV roles</span>${escapeHtml((item.cv_roles || []).join(", ") || "N/A")}</div>
              <div><span>CV seniority</span>${item.cv_seniority ?? "N/A"}</div>
            </div>
            <div class="question-box">
              <span>Suggested interview questions</span>
              <ol>${questions}</ol>
            </div>
          </details>
        </article>
      `;
    })
    .join("");
}

function scoreCell(label, value) {
  return `
    <div>
      <span>${label}</span>
      ${pct(value)}
      <i style="--value:${Math.round((Number(value) || 0) * 100)}%"></i>
    </div>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderFileNames(input, target, emptyLabel) {
  const files = Array.from(input.files || []);
  if (!files.length) {
    target.textContent = emptyLabel;
    return;
  }
  if (files.length === 1) {
    target.textContent = files[0].name;
    return;
  }
  target.textContent = `${files.length} files selected`;
}

jdPdfInput.addEventListener("change", () => {
  renderFileNames(jdPdfInput, jdPdfName, "Chưa chọn file");
  setStatus("File selected", "idle");
});

cvPdfInput.addEventListener("change", () => {
  renderFileNames(cvPdfInput, cvPdfName, "Chưa chọn file");
  setStatus("File selected", "idle");
});

function refreshCurrentResults() {
  updateThresholdLabel();
  if (!currentPayload) return;
  renderSummary(currentPayload);
  renderResults(currentPayload);
}

function csvValue(value) {
  const text = Array.isArray(value) ? value.join("; ") : String(value ?? "");
  return `"${text.replaceAll('"', '""')}"`;
}

function exportVisibleCsv() {
  if (!currentPayload?.ranking?.length) return;
  const rows = getVisibleRanking(currentPayload);
  const header = [
    "rank",
    "candidate_name",
    "score",
    "recommendation",
    "decision",
    "semantic_score",
    "skill_coverage",
    "tech_match",
    "role_alignment",
    "seniority",
    "matched_skills",
    "missing_required_skills",
  ];
  const csvRows = [
    header.join(","),
    ...rows.map((item) =>
      [
        item.rank,
        item.candidate_name,
        item.score,
        item.recommendation,
        item.decision,
        item.semantic_score,
        item.required_skill_coverage,
        item.tech_skill_score,
        item.role_alignment_score,
        item.seniority_score,
        item.matched_tech_skills,
        item.missing_required_skills,
      ].map(csvValue).join(",")
    ),
  ];
  const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "cv_jd_ranking.csv";
  link.click();
  URL.revokeObjectURL(url);
}

async function copyShortlist() {
  if (!currentPayload?.ranking?.length) return;
  const rows = getVisibleRanking(currentPayload);
  const lines = rows.map((item) => {
    const missing = item.missing_required_skills?.length
      ? `Missing: ${item.missing_required_skills.join(", ")}`
      : "Missing: None";
    return `#${item.rank} ${item.candidate_name} - ${pct(item.score)} - ${item.recommendation}. ${missing}`;
  });
  const text = [`Job: ${currentPayload.job_title || "N/A"}`, ...lines].join("\n");
  await navigator.clipboard.writeText(text);
  setStatus("Shortlist copied", "ready");
}

thresholdInput.addEventListener("input", refreshCurrentResults);
shortlistOnly.addEventListener("change", refreshCurrentResults);
exportCsvButton.addEventListener("click", exportVisibleCsv);
copyShortlistButton.addEventListener("click", () => {
  copyShortlist().catch((error) => {
    setStatus("Copy failed", "error");
    results.className = "results empty-state";
    results.textContent = error.message;
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const useFiles = hasSelectedFiles();
  const resumes = parseResumes(cvInput.value);
  const hasJobInput = Boolean(jdInput.value.trim() || jdPdfInput.files?.length);
  const hasCvInput = Boolean(resumes.length || cvPdfInput.files?.length);
  if (!hasJobInput) {
    setStatus("Error", "error");
    results.className = "results empty-state";
    results.textContent = "Can nhap Job Description hoac upload JD file.";
    return;
  }
  if (!hasCvInput) {
    setStatus("Error", "error");
    results.className = "results empty-state";
    results.textContent = "Can nhap it nhat mot CV hoac upload CV file.";
    return;
  }
  if (!useFiles && !resumes.length) {
    setStatus("Error", "error");
    results.className = "results empty-state";
    results.textContent = "Cần nhập ít nhất một CV hoặc upload CV file.";
    return;
  }
  if (useFiles && !cvPdfInput.files?.length && !resumes.length) {
    setStatus("Error", "error");
    results.className = "results empty-state";
    results.textContent = "Cần nhập CV text hoặc upload ít nhất một CV file.";
    return;
  }

  setStatus("Running", "running");
  rankButton.disabled = true;
  resultTitle.textContent = "Đang tính ranking...";
  results.className = "results empty-state";
  results.textContent = "Backend đang encode JD/CV và tính score.";

  try {
    let response;
    if (useFiles) {
      const formData = new FormData();
      const shouldSendCvText = !cvPdfInput.files?.length || cvInput.value.trim() !== initialCvText.trim();
      formData.append("job_description", jdInput.value);
      if (shouldSendCvText) {
        formData.append("cv_text", cvInput.value);
      }
      if (jdPdfInput.files?.[0]) {
        formData.append("jd_file", jdPdfInput.files[0]);
      }
      for (const file of Array.from(cvPdfInput.files || [])) {
        formData.append("cv_files", file);
      }
      response = await fetch("/api/rank-files", {
        method: "POST",
        body: formData,
      });
    } else {
      response = await fetch("/api/rank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_description: jdInput.value,
          resumes,
        }),
      });
    }
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Request failed");
    }
    currentPayload = payload;
    setStatus("Ready", "ready");
    resultTitle.textContent = payload.job_title || "Ranking result";
    renderSummary(payload);
    renderResults(payload);
  } catch (error) {
    currentPayload = null;
    setStatus("Error", "error");
    resultTitle.textContent = "Lỗi khi chạy matching";
    summary.innerHTML = "";
    results.className = "results empty-state";
    results.textContent = error.message;
  } finally {
    rankButton.disabled = false;
  }
});

clearButton.addEventListener("click", () => {
  currentPayload = null;
  jdPdfInput.value = "";
  cvPdfInput.value = "";
  renderFileNames(jdPdfInput, jdPdfName, "Chưa chọn file");
  renderFileNames(cvPdfInput, cvPdfName, "Chưa chọn file");
  setStatus("Idle", "idle");
  resultTitle.textContent = "Chưa có kết quả";
  summary.innerHTML = "";
  results.className = "results empty-state";
  results.textContent = "Nhập JD/CV rồi bấm Run matching để xem ranking.";
});
