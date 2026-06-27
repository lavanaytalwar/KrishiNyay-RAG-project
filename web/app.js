const state = {
  lastSources: [],
  health: null,
  demoConfig: null,
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
const routeTrace = document.querySelector("#routeTrace");
const demoChecklist = document.querySelector("#demoChecklist");

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

function renderRouteTrace(data) {
  if (!data) {
    routeTrace.innerHTML = `
      <span class="sticky-badge badge-gray">No query yet</span>
      <p>Ask a question to see whether the backend used static RAG or the dynamic live-data router.</p>
    `;
    return;
  }

  const isDynamic = data.route === "dynamic_router";
  const routeLabel = isDynamic ? "Dynamic Router" : "Static RAG";
  const reason = data.route_reason || data.normalized_question || "retrieval";
  const provider = data.llm_provider || "unknown";
  const liveStatus = data.live_status
    ? `<span class="sticky-badge badge-green">${escapeHtml(data.live_status)}</span>`
    : "";
  const dataProvider = data.data_provider
    ? `<span class="sticky-badge badge-gray">${escapeHtml(data.data_provider)}</span>`
    : "";
  routeTrace.innerHTML = `
    <span class="sticky-badge ${isDynamic ? "badge-pink" : "badge-yellow"}">${routeLabel}</span>
    <span class="sticky-badge badge-blue">${escapeHtml(provider)}</span>
    ${liveStatus}
    ${dataProvider}
    <p>${escapeHtml(reason)}</p>
  `;
}

function renderStatus(health) {
  const badges = [
    ["badge-blue", `${health.total_chunks ?? 0} chunks`],
    ["badge-yellow", health.embedding_backend || "embedding unknown"],
  ];
  statusStrip.innerHTML = badges
    .map(([klass, text]) => `<span class="sticky-badge ${klass}">${escapeHtml(text)}</span>`)
    .join("");
}

function renderHealth(health) {
  const liveData = health.live_data || {};
  const rows = [
    ["Status", health.status],
    ["Chunks", health.total_chunks],
    ["Embeddings", health.embedding_backend],
    ["Dimensions", health.embedding_dim],
    ["Retrieval", health.retrieval_mode],
    ["Lexical chunks", health.lexical_chunks],
    ["LLM", health.llm_provider],
    ["Collection", health.collection],
    ["Router", health.dynamic_router],
    ["Mandi API", liveData.mandi_api_configured ? "configured" : "needs key"],
    ["Weather API", liveData.weather_provider || "unknown"],
    ["Phase", health.phase],
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

function renderDemoChecklist(config) {
  const remaining = config?.remaining_work || [];
  const mediaNote = config?.media_note || "";
  demoChecklist.innerHTML = `
    <div class="checklist-block">
      <h3>What remains after Phase 3</h3>
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

async function fetchDemoConfig() {
  try {
    const response = await fetch("/demo-config");
    if (!response.ok) throw new Error(await response.text());
    const config = await response.json();
    state.demoConfig = config;
    renderPhaseGrid(config);
    renderDemoChecklist(config);
    renderDemoQuestions(config);
    annotateMotionSlots(config);
  } catch (error) {
    phaseGrid.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
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
  } catch (error) {
    statusStrip.innerHTML = `<span class="sticky-badge badge-pink">Backend unavailable</span>`;
    healthGrid.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

async function askQuestion(question) {
  addMessage("user", question);
  const loadingMessage = addMessage("assistant", "Retrieving trusted sources and checking live-data routing...", {
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
      body: JSON.stringify({ question }),
    });
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    loadingMessage.remove();
    addMessage("assistant", data.answer, {
      badge: data.route === "dynamic_router" ? "Dynamic Router" : "RAG Answer",
      badgeClass: data.route === "dynamic_router" ? "badge-pink" : "badge-yellow",
    });
    renderSources(data.sources || []);
    renderRouteTrace(data);
    modeBadge.textContent = data.mode === "dynamic_router" ? "Dynamic Router" : "RAG";
    modeBadge.className = `sticky-badge ${data.mode === "dynamic_router" ? "badge-pink" : "badge-yellow"}`;
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
  ingestResult.textContent = "Indexing document...";

  const payload = {
    title: document.querySelector("#ingestTitleInput").value.trim(),
    url: document.querySelector("#ingestUrlInput").value.trim(),
    category: document.querySelector("#ingestCategoryInput").value.trim() || "scheme",
    state: document.querySelector("#ingestStateInput").value.trim() || "central",
    doc_type: document.querySelector("#ingestDocTypeInput").value.trim() || "Guidelines",
    text: document.querySelector("#ingestTextInput").value.trim(),
  };

  try {
    const response = await fetch("/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
