const input = document.getElementById("query-input");
const suggestionsEl = document.getElementById("suggestions");
const debugEl = document.getElementById("debug");

let debounceTimer = null;
let requestSeq = 0;
const DEBOUNCE_MS = 150;

input.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => fetchSuggestions(input.value), DEBOUNCE_MS);
});

input.addEventListener("blur", () => {
  setTimeout(() => hideSuggestions(), 150); // allow click events on items to fire first
});

async function fetchSuggestions(text) {
  if (!text.trim()) {
    hideSuggestions();
    debugEl.textContent = "";
    return;
  }

  const seq = ++requestSeq;
  try {
    const res = await fetch(`/api/suggest?text=${encodeURIComponent(text)}`);
    const data = await res.json();
    if (seq !== requestSeq) return; // a newer keystroke superseded this request

    renderSuggestions(data);
    renderDebug(data);
  } catch (e) {
    console.error(e);
  }
}

function renderSuggestions(data) {
  suggestionsEl.innerHTML = "";

  if (!data.query || data.entity_types.length === 0) {
    hideSuggestions();
    return;
  }

  if (data.suggestions.length === 0) {
    const empty = document.createElement("div");
    empty.className = "suggestions-empty";
    empty.textContent = `No matches for "${data.query}" in [${data.entity_types.join(", ")}]`;
    suggestionsEl.appendChild(empty);
    suggestionsEl.classList.remove("hidden");
    return;
  }

  for (const s of data.suggestions) {
    const item = document.createElement("div");
    item.className = "suggestion-item";

    const value = document.createElement("span");
    value.className = "suggestion-value";
    value.textContent = s.value;

    const type = document.createElement("span");
    type.className = `suggestion-type type-${s.type}`;
    type.textContent = s.type;

    item.appendChild(value);
    item.appendChild(type);
    item.addEventListener("mousedown", () => selectSuggestion(s));

    suggestionsEl.appendChild(item);
  }

  suggestionsEl.classList.remove("hidden");
}

function selectSuggestion(s) {
  const text = input.value;
  const idx = text.lastIndexOf(" ");
  const head = idx >= 0 ? text.slice(0, idx + 1) : "";
  input.value = head + s.value + " ";
  hideSuggestions();
  input.focus();
}

function hideSuggestions() {
  suggestionsEl.classList.add("hidden");
  suggestionsEl.innerHTML = "";
}

function renderDebug(data) {
  const scores = Object.entries(data.scores)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k}=${v.toFixed(2)}`)
    .join("  ");
  debugEl.textContent =
    `context: "${data.context}"\n` +
    `query:   "${data.query}"\n` +
    `routed types: [${data.entity_types.join(", ")}]  (source: ${data.classifier_source})\n` +
    `scores: ${scores}`;
}
