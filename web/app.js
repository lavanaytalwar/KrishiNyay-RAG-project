const state = {
  lastSources: [],
  health: null,
  demoConfig: null,
  conversationId: `web-${Date.now()}-${Math.random().toString(16).slice(2)}`,
  workflowContext: null,
};

const pages = {
  home: document.querySelector("#page-home"),
  chat: document.querySelector("#page-chat"),
  knowledge: document.querySelector("#page-knowledge"),
  ingest: document.querySelector("#page-ingest"),
  system: document.querySelector("#page-system"),
};

const pageTitles = {
  home: "A source-grounded farmer assistant for Indian schemes.",
  chat: "Ask farmer questions and inspect the evidence.",
  knowledge: "Run demo checks against trusted sources.",
  ingest: "Add small official documents during a live demo.",
  system: "Verify backend readiness before presenting.",
};

const navItems = document.querySelectorAll("[data-route]");
const messages = document.querySelector("#messages");
const queryForm = document.querySelector("#queryForm");
const questionInput = document.querySelector("#questionInput");
const sendButton = document.querySelector("#sendButton");
const sourceList = document.querySelector("#sourceList");
const sourceCount = document.querySelector("#sourceCount");
const modeBadge = document.querySelector("#modeBadge");
const statusStrip = document.querySelector("#statusStrip");
const healthGrid = document.querySelector("#healthGrid");
const ingestForm = document.querySelector("#ingestForm");
const ingestResult = document.querySelector("#ingestResult");
const pageTitle = document.querySelector("#pageTitle");
const phaseGrid = document.querySelector("#phaseGrid");
const recentPhaseGrid = document.querySelector("#recentPhaseGrid");
const routeTrace = document.querySelector("#routeTrace");
const validationGrid = document.querySelector("#validationGrid");
const apiKeyGuide = document.querySelector("#apiKeyGuide");
const demoChecklist = document.querySelector("#demoChecklist");
const ingestInputs = ingestForm ? ingestForm.querySelectorAll("input, textarea, button") : [];

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setRoute(route) {
  const selected = pages[route] ? route : "home";
  Object.entries(pages).forEach(([key, page]) => {
    page.classList.toggle("active", key === selected);
  });
  navItems.forEach((item) => {
    item.classList.toggle("active", item.dataset.route === selected);
  });
  pageTitle.textContent = pageTitles[selected] || pageTitles.home;
}

function addMessage(role, text, options = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}${options.loading ? " loading" : ""}`;
  article.innerHTML = `
    <div class="avatar">${role === "user" ? "You" : "AI"}</div>
    <div class="bubble">
      ${options.badge ? `<span class="sticky-badge ${escapeHtml(options.badgeClass || "badge-gray")}">${escapeHtml(options.badge)}</span>` : ""}
      <p>${escapeHtml(text).replaceAll("\n", "<br>")}</p>
    </div>
  `;
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function badgeClass(index) {
  return ["badge-yellow", "badge-blue", "badge-green", "badge-pink", "badge-gray"][index % 5];
}

function buildQueryPayload(question) {
  return {
    question,
    conversation_id: state.conversationId,
    workflow_context: state.workflowContext,
  };
}

function updateWorkflowContext(data) {
  state.workflowContext = data.workflow_context || null;
}

function formatSimilarity(value) {
  if (value === null || value === undefined) return "Live";
  return `${Math.round(Number(value) * 100)}% match`;
}

function renderSources(sources) {
  state.lastSources = sources || [];
  sourceCount.textContent = `${state.lastSources.length} source${state.lastSources.length === 1 ? "" : "s"}`;

  if (!state.lastSources.length) {
    sourceList.innerHTML = `<div class="empty-state">No retrieved chunks for this response.</div>`;
    return;
  }

  sourceList.innerHTML = state.lastSources
    .map((source, index) => {
      const title = source.display || source.source || "Government source";
      const docType = source.doc_type || source.category || "Source";
      const url = source.url
        ? `<a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">Open official source</a>`
        : "";
      return `
        <article class="source-card">
          <div class="source-meta">
            <span class="sticky-badge ${badgeClass(index)}">${escapeHtml(docType)}</span>
            <span class="sticky-badge badge-gray">${escapeHtml(formatSimilarity(source.similarity))}</span>
            <span class="sticky-badge badge-green">${escapeHtml(source.state || "central")}</span>
          </div>
          <h3>${escapeHtml(title)}</h3>
          <p>${escapeHtml(source.text || "No preview text returned for this source.")}</p>
          ${url}
        </article>
      `;
    })
    .join("");
}

function stepClass(status) {
  if (status === "done") return "step-done";
  if (status === "blocked") return "step-blocked";
  if (status === "waiting") return "step-waiting";
  return "step-idle";
}

function renderWorkflowStep(index, title, value, status = "done") {
  return `
    <article class="workflow-step ${stepClass(status)}">
      <span class="step-index">${index}</span>
      <div>
        <strong>${escapeHtml(title)}</strong>
        <p>${escapeHtml(value || "Not available yet")}</p>
      </div>
    </article>
  `;
}

function workflowToolLabel(data) {
  if (data.answer_kind === "clarification") return "Clarification prompt";
  if (data.route === "rag") return "Hybrid RAG: MiniLM + lexical";
  if (data.route === "dynamic_router") return `Allowed live tool: ${data.tool_used || "dynamic router"}`;
  if (data.route === "system_info") return "System metadata tool";
  return data.route || "Workflow";
}

function workflowEvidenceLabel(data) {
  const verifier = data.evidence_verifier || {};
  if (verifier.status) {
    const expected = verifier.expected_category ? ` · expected ${verifier.expected_category}` : "";
    return `${verifier.status}${expected}`;
  }
  if (data.sources?.length) return `${data.sources.length} source(s) returned`;
  return "No evidence needed";
}

function workflowEvidenceStatus(data) {
  if (data.evidence_verified) return "done";
  if (data.evidence_verifier?.status === "needs_more_input") return "waiting";
  if (data.evidence_verifier?.status === "failed") return "blocked";
  return data.sources?.length ? "done" : "waiting";
}

function workflowAnswerLabel(data) {
  if (data.answer_kind === "clarification") return "Asked for missing input";
  if (data.generation_status) return data.generation_status;
  if (data.route === "dynamic_router") return "Router answer";
  return data.answer_kind || "Prepared answer";
}

function fieldPreview(data) {
  if (!data?.answer || data.answer_kind === "clarification") return "";
  const answer = String(data.answer).replace(/\s+/g, " ").trim();
  const shortAnswer = answer.length > 420 ? `${answer.slice(0, 417).trim()}...` : answer;
  const sourceNames = (data.sources || [])
    .slice(0, 2)
    .map((source) => source.display || source.source || source.doc_type)
    .filter(Boolean);
  const sourceText = sourceNames.length ? ` Sources: ${sourceNames.join("; ")}` : "";
  return `
    <div class="field-preview">
      <strong>Field / WhatsApp preview</strong>
      <p>${escapeHtml(shortAnswer + sourceText)}</p>
    </div>
  `;
}

function renderRouteTrace(data) {
  if (!data) {
    routeTrace.innerHTML = `
      <div class="trace-title">
        <span class="sticky-badge badge-gray">No query yet</span>
        <p>Ask a question to see the guarded workflow.</p>
      </div>
      <div class="workflow-steps">
        ${renderWorkflowStep(1, "Language", "Detect English or Hinglish", "idle")}
        ${renderWorkflowStep(2, "Planner", "Classify intent and slots", "idle")}
        ${renderWorkflowStep(3, "Tool / Retrieval", "Use only allowed tools", "idle")}
        ${renderWorkflowStep(4, "Verifier", "Check route, source type, relevance", "idle")}
        ${renderWorkflowStep(5, "Answer", "Synthesize with selected language", "idle")}
      </div>
    `;
    return;
  }

  const isDynamic = data.route === "dynamic_router";
  const isSystem = data.route === "system_info";
  const isClarification = data.answer_kind === "clarification";
  const isWorkflow = data.route === "workflow";
  const routeLabel = isClarification
    ? "Clarification Needed"
    : isWorkflow
      ? "Workflow"
      : isSystem
        ? "System Info"
        : isDynamic
          ? "Dynamic Router"
          : "Static RAG";
  const routeBadgeClass = isClarification || isWorkflow
    ? "badge-blue"
    : isSystem
      ? "badge-blue"
      : isDynamic
        ? "badge-pink"
        : "badge-yellow";
  const reason = data.route_reason || data.normalized_question || "retrieval";
  const provider = data.llm_provider || "unknown";
  const intent = data.intent
    ? `<span class="sticky-badge badge-gray">${escapeHtml(data.intent)}</span>`
    : "";
  const workflowState = data.workflow_state
    ? `<span class="sticky-badge badge-blue">${escapeHtml(data.workflow_state)}</span>`
    : "";
  const toolUsed = data.tool_used
    ? `<span class="sticky-badge badge-green">Tool: ${escapeHtml(data.tool_used)}</span>`
    : "";
  const answerKind = data.answer_kind
    ? `<span class="sticky-badge badge-gray">${escapeHtml(data.answer_kind)}</span>`
    : "";
  const liveStatus = data.live_status
    ? `<span class="sticky-badge badge-green">${escapeHtml(data.live_status)}</span>`
    : "";
  const dataProvider = data.data_provider
    ? `<span class="sticky-badge badge-gray">${escapeHtml(data.data_provider)}</span>`
    : "";
  const generationStatus = data.generation_status
    ? `<span class="sticky-badge badge-green">${escapeHtml(data.generation_status)}</span>`
    : "";
  const answerLanguage = data.answer_language
    ? `<span class="sticky-badge badge-blue">Lang: ${escapeHtml(data.answer_language)}</span>`
    : "";
  const evidenceStatus = data.evidence_verifier?.status
    ? `<span class="sticky-badge ${data.evidence_verified ? "badge-green" : "badge-yellow"}">Evidence: ${escapeHtml(data.evidence_verifier.status)}</span>`
    : "";
  const generatedAt = data.fetched_at
    ? `<span class="sticky-badge badge-blue">${escapeHtml(new Date(data.fetched_at).toLocaleString())}</span>`
    : "";
  const languageLabel = data.answer_language === "hinglish"
    ? "Hinglish / Roman Hindi"
    : data.answer_language === "english"
      ? "English"
      : data.answer_language || "Unknown";
  routeTrace.innerHTML = `
    <div class="trace-title">
      <span class="sticky-badge ${routeBadgeClass}">${routeLabel}</span>
      <span class="sticky-badge badge-blue">${escapeHtml(provider)}</span>
      ${intent}
      ${workflowState}
      ${toolUsed}
      ${answerKind}
      ${answerLanguage}
      ${evidenceStatus}
      ${liveStatus}
      ${dataProvider}
      ${generationStatus}
      ${generatedAt}
    </div>
    <div class="workflow-steps">
      ${renderWorkflowStep(1, "Language", languageLabel, data.answer_language ? "done" : "waiting")}
      ${renderWorkflowStep(2, "Planner", `${data.intent || "unknown"} · ${reason}`, "done")}
      ${renderWorkflowStep(3, "Tool / Retrieval", workflowToolLabel(data), data.answer_kind === "clarification" ? "waiting" : "done")}
      ${renderWorkflowStep(4, "Verifier", workflowEvidenceLabel(data), workflowEvidenceStatus(data))}
      ${renderWorkflowStep(5, "Answer", workflowAnswerLabel(data), data.answer_kind === "clarification" ? "waiting" : "done")}
    </div>
    ${fieldPreview(data)}
  `;
}

function renderStatus(health) {
  const badges = [
    [health.demo_public ? "badge-green" : "badge-gray", health.demo_public ? "Public demo" : "Local mode"],
    ["badge-blue", `${health.total_chunks ?? 0} chunks`],
    ["badge-yellow", health.embedding_backend || "embedding unknown"],
    ["badge-pink", health.llm_provider || "LLM unknown"],
  ];
  statusStrip.innerHTML = badges
    .map(([klass, text]) => `<span class="sticky-badge ${klass}">${escapeHtml(text)}</span>`)
    .join("");
}

function renderHealth(health) {
  const liveData = health.live_data || {};
  const readiness = health.readiness || {};
  const rows = [
    ["Status", health.status],
    ["Chunks", health.total_chunks],
    ["Embeddings", health.embedding_backend],
    ["Dimensions", health.embedding_dim],
    ["Retrieval", health.retrieval_mode],
    ["Lexical chunks", health.lexical_chunks],
    ["Chroma path", health.chroma_path || "chroma_db"],
    ["Chroma runtime", health.chroma_runtime_path || health.chroma_path || "chroma_db"],
    ["Chunks dir", health.chunks_dir || "data/chunks"],
    ["LLM", health.llm_provider],
    ["Collection", health.collection],
    ["Router", health.dynamic_router],
    ["Mandi API", liveData.mandi_api_configured ? "configured" : "needs key"],
    ["Weather API", liveData.weather_provider || "unknown"],
    ["Phase", health.phase],
    ["OCR", readiness.ocr_ready ? readiness.ocr_engine || "ready" : "needs setup"],
    ["Indic OCR", readiness.indic_ocr_ready ? "ready" : "needs language packs"],
    ["Public demo", health.demo_public ? "enabled" : "disabled"],
    ["Public ready", health.public_demo_ready ? "ready" : "not ready"],
    ["Gemini", readiness.gemini_generation_ready ? "active" : "not active"],
    ["Live ingest", readiness.live_ingest_enabled ? "enabled" : "disabled"],
    ["Ingest token", readiness.live_ingest_requires_token ? "required" : "not required"],
    ["Fine-tuning", health.fine_tuning || "not required"],
    ["Demo", health.demo_ready ? "ready" : "not ready"],
  ];

  healthGrid.innerHTML = rows
    .map(
      ([label, value], index) => `
        <div class="health-card">
          <span class="sticky-badge ${badgeClass(index)}">${escapeHtml(label)}</span>
          <strong>${escapeHtml(value ?? "unknown")}</strong>
          <span>Current backend state</span>
        </div>
      `
    )
    .join("");
}

function renderPhaseGrid(config) {
  const phases = config?.phase_status || [];
  if (!phases.length) {
    phaseGrid.innerHTML = `<div class="empty-state">Phase status is unavailable.</div>`;
    return;
  }

  phaseGrid.innerHTML = phases
    .map((phase, index) => `
      <article class="phase-card">
        <span class="sticky-badge ${badgeClass(index)}">${escapeHtml(phase.status)}</span>
        <strong>${escapeHtml(phase.phase)} · ${escapeHtml(phase.title)}</strong>
        <p>${escapeHtml(phase.summary)}</p>
      </article>
    `)
    .join("");
}

function renderRecentCapabilities(config) {
  const capabilities = config?.recent_capabilities || [];
  if (!capabilities.length) {
    recentPhaseGrid.innerHTML = `<div class="empty-state">Recent phase highlights are unavailable.</div>`;
    return;
  }

  recentPhaseGrid.innerHTML = capabilities
    .map((item, index) => `
      <article class="recent-phase-card">
        <span class="sticky-badge ${badgeClass(index)}">${escapeHtml(item.phase)}</span>
        <strong>${escapeHtml(item.title)}</strong>
        <em>${escapeHtml(item.metric)}</em>
        <p>${escapeHtml(item.summary)}</p>
      </article>
    `)
    .join("");
}

function renderValidationGates(config) {
  const gates = config?.validation_gates || [];
  if (!gates.length) {
    validationGrid.innerHTML = `<div class="empty-state">Validation gates are unavailable.</div>`;
    return;
  }

  validationGrid.innerHTML = gates
    .map((gate, index) => `
      <article class="validation-card">
        <span class="sticky-badge ${badgeClass(index)}">Passed</span>
        <strong>${escapeHtml(gate.name)}</strong>
        <code>${escapeHtml(gate.command)}</code>
        <p>${escapeHtml(gate.result)}</p>
      </article>
    `)
    .join("");
}

function renderApiKeyGuide(config, health = state.health) {
  const guide = config?.api_key_guide || {};
  const liveData = health?.live_data || {};
  const isConfigured = Boolean(liveData.mandi_api_configured);
  const steps = guide.steps || [];
  apiKeyGuide.innerHTML = `
    <div class="checklist-block api-guide-block">
      <div class="guide-head">
        <div>
          <h3>${escapeHtml(guide.title || "API key setup")}</h3>
          <p>${escapeHtml(guide.provider || "Live data provider")}</p>
        </div>
        <span class="sticky-badge ${isConfigured ? "badge-green" : "badge-yellow"}">
          ${isConfigured ? "Mandi API configured" : "Mandi API needs key"}
        </span>
      </div>
      <ol>
        ${steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}
      </ol>
      <code>${escapeHtml(guide.env || "DATA_GOV_IN_API_KEY=your_key_here")}</code>
      <p>${escapeHtml(guide.note || "")}</p>
    </div>
  `;
}

function renderDemoChecklist(config) {
  const remaining = config?.remaining_work || [];
  const mediaNote = config?.media_note || "";
  const publicFlow = config?.recommended_demo_flow || [];
  const safetyNotice = config?.public_safety_notice || "";
  demoChecklist.innerHTML = `
    <div class="checklist-block">
      <h3>Public demo readiness</h3>
      ${safetyNotice ? `<p>${escapeHtml(safetyNotice)}</p>` : ""}
      ${publicFlow.length ? `
        <ol>
          ${publicFlow.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
        </ol>
      ` : ""}
      <h3>Hardening focus after Phase 12 baseline</h3>
      <ul>
        ${remaining.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
      <p>${escapeHtml(mediaNote)}</p>
    </div>
  `;
}

function renderDemoQuestions(config) {
  const questions = config?.demo_questions || [];
  if (!questions.length) return;

  document.querySelector("#examples").innerHTML = questions
    .map((item) => `
      <button data-kind="${escapeHtml(item.kind)}" title="${escapeHtml(item.label)}">
        ${escapeHtml(item.question)}
      </button>
    `)
    .join("");
}

function annotateMotionSlots(config) {
  const slots = config?.motion_slots || [];
  const targets = Array.from(document.querySelectorAll("[data-motion-slot]"));
  slots.forEach((slot) => {
    const target = targets.find((element) => element.dataset.motionSlot === slot.slot);
    if (!target) return;
    target.setAttribute("aria-label", slot.label);
    target.dataset.motionStyle = slot.style;
  });
}

function applyPublicDemoLock() {
  const publicMode = Boolean(state.health?.demo_public || state.demoConfig?.public_demo);
  const ingestEnabled = Boolean(state.health?.readiness?.live_ingest_enabled);
  const locked = publicMode || !ingestEnabled;
  ingestInputs.forEach((input) => {
    input.disabled = locked;
  });
  if (publicMode) {
    ingestResult.textContent =
      "Public demo mode is read-only. Document ingest is disabled; run locally for OCR/admin ingestion.";
  } else if (!ingestEnabled) {
    ingestResult.textContent =
      "Live ingest is disabled. Set ENABLE_LIVE_INGEST=true only in a trusted local demo/admin environment.";
  }
}

async function fetchDemoConfig() {
  try {
    const response = await fetch("/demo-config");
    if (!response.ok) throw new Error(await response.text());
    const config = await response.json();
    state.demoConfig = config;
    renderPhaseGrid(config);
    renderRecentCapabilities(config);
    renderValidationGates(config);
    renderApiKeyGuide(config);
    renderDemoChecklist(config);
    renderDemoQuestions(config);
    annotateMotionSlots(config);
    applyPublicDemoLock();
  } catch (error) {
    phaseGrid.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
    recentPhaseGrid.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
    validationGrid.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
    apiKeyGuide.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
    demoChecklist.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

async function fetchHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error(await response.text());
    const health = await response.json();
    state.health = health;
    renderStatus(health);
    renderHealth(health);
    renderApiKeyGuide(state.demoConfig, health);
    applyPublicDemoLock();
  } catch (error) {
    statusStrip.innerHTML = `<span class="sticky-badge badge-pink">Backend unavailable</span>`;
    healthGrid.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

async function askQuestion(question) {
  addMessage("user", question);
  const payload = buildQueryPayload(question);
  const loadingMessage = addMessage("assistant", "Planning route, verifying evidence, and preparing a grounded answer...", {
    loading: true,
    badge: "Working",
    badgeClass: "badge-blue",
  });
  sendButton.disabled = true;
  sendButton.textContent = "Asking";
  modeBadge.textContent = "Retrieving";

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    loadingMessage.remove();
    const isSystem = data.route === "system_info";
    const isDynamic = data.route === "dynamic_router";
    const isClarification = data.answer_kind === "clarification";
    const isWorkflow = data.route === "workflow";
    const responseBadge = isClarification
      ? "Clarification Needed"
      : isWorkflow
        ? "Workflow"
        : isSystem
          ? "System Info"
          : isDynamic
            ? "Dynamic Router"
            : "RAG Answer";
    const responseBadgeClass = isClarification || isWorkflow
      ? "badge-blue"
      : isSystem
        ? "badge-blue"
        : isDynamic
          ? "badge-pink"
          : "badge-yellow";
    addMessage("assistant", data.answer, {
      badge: responseBadge,
      badgeClass: responseBadgeClass,
    });
    renderSources(data.sources || []);
    renderRouteTrace(data);
    updateWorkflowContext(data);
    modeBadge.textContent = isClarification
      ? "Clarification"
      : isSystem
        ? "System Info"
        : isDynamic
          ? "Dynamic Router"
          : "RAG";
    modeBadge.className = `sticky-badge ${responseBadgeClass}`;
  } catch (error) {
    loadingMessage.remove();
    addMessage("assistant", `I could not answer that request. ${error.message}`);
    modeBadge.textContent = "Error";
    modeBadge.className = "sticky-badge badge-pink";
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Ask";
  }
}

queryForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  askQuestion(question);
});

document.querySelector("#examples").addEventListener("click", (event) => {
  if (event.target.tagName !== "BUTTON") return;
  askQuestion(event.target.textContent.trim());
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    window.location.hash = "chat";
    setRoute("chat");
    askQuestion(button.dataset.question);
  });
});

ingestForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.health?.readiness && !state.health.readiness.live_ingest_enabled) {
    ingestResult.textContent =
      state.health.demo_public
        ? "Public demo mode is read-only. Document ingest is disabled."
        : "Live ingest is disabled. Set ENABLE_LIVE_INGEST=true only in a trusted local demo/admin environment.";
    return;
  }
  ingestResult.textContent = "Indexing document...";

  const payload = {
    title: document.querySelector("#ingestTitleInput").value.trim(),
    url: document.querySelector("#ingestUrlInput").value.trim(),
    category: document.querySelector("#ingestCategoryInput").value.trim() || "scheme",
    state: document.querySelector("#ingestStateInput").value.trim() || "central",
    doc_type: document.querySelector("#ingestDocTypeInput").value.trim() || "Guidelines",
    text: document.querySelector("#ingestTextInput").value.trim(),
  };
  const ingestToken = document.querySelector("#ingestTokenInput").value.trim();
  const headers = { "Content-Type": "application/json" };
  if (ingestToken) headers["X-Ingest-Token"] = ingestToken;

  try {
    const response = await fetch("/ingest", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    ingestResult.textContent = `Indexed ${data.chunks_added} chunks as ${data.doc_id}.`;
    ingestForm.reset();
    await fetchHealth();
  } catch (error) {
    ingestResult.textContent = `Ingest failed: ${error.message}`;
  }
});

document.querySelector("#refreshHealth").addEventListener("click", fetchHealth);

window.addEventListener("hashchange", () => setRoute(window.location.hash.replace("#", "")));
setRoute(window.location.hash.replace("#", "") || "home");
renderRouteTrace(null);
fetchDemoConfig();
fetchHealth();
