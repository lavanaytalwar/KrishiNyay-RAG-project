const state = {
  lastSources: [],
  health: null,
};

const pages = {
  chat: document.querySelector("#page-chat"),
  knowledge: document.querySelector("#page-knowledge"),
  ingest: document.querySelector("#page-ingest"),
  system: document.querySelector("#page-system"),
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

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setRoute(route) {
  const selected = pages[route] ? route : "chat";
  Object.entries(pages).forEach(([key, page]) => {
    page.classList.toggle("active", key === selected);
  });
  navItems.forEach((item) => {
    item.classList.toggle("active", item.dataset.route === selected);
  });
}

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.innerHTML = `
    <div class="avatar">${role === "user" ? "You" : "AI"}</div>
    <div class="bubble"><p>${escapeHtml(text).replaceAll("\n", "<br>")}</p></div>
  `;
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
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
  sourceCount.textContent = `${state.lastSources.length} chunks`;

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

function renderStatus(health) {
  const badges = [
    ["badge-green", health.status || "unknown"],
    ["badge-blue", `${health.total_chunks ?? 0} chunks`],
    ["badge-yellow", health.embedding_backend || "embedding unknown"],
    ["badge-pink", health.llm_provider || "llm unknown"],
  ];
  statusStrip.innerHTML = badges
    .map(([klass, text]) => `<span class="sticky-badge ${klass}">${escapeHtml(text)}</span>`)
    .join("");
}

function renderHealth(health) {
  const rows = [
    ["Status", health.status],
    ["Chunks", health.total_chunks],
    ["Embeddings", health.embedding_backend],
    ["LLM", health.llm_provider],
    ["Collection", health.collection],
    ["Router", health.dynamic_router],
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
    addMessage("assistant", data.answer);
    renderSources(data.sources || []);
    modeBadge.textContent = data.mode === "dynamic_router" ? "Dynamic Router" : "RAG";
    modeBadge.className = `sticky-badge ${data.mode === "dynamic_router" ? "badge-pink" : "badge-yellow"}`;
  } catch (error) {
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
setRoute(window.location.hash.replace("#", "") || "chat");
fetchHealth();
