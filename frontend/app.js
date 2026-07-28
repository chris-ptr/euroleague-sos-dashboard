const state = {
  view: "info",
  round: null,
  latestRound: null,
  nNext: 5,
};

const els = {
  navButtons: document.querySelectorAll(".nav-btn"),
  roundDec: document.getElementById("round-dec"),
  roundInc: document.getElementById("round-inc"),
  roundValue: document.getElementById("round-value"),
  nDec: document.getElementById("n-dec"),
  nInc: document.getElementById("n-inc"),
  nValue: document.getElementById("n-value"),
  nControl: document.getElementById("n-next-control"),
  controls: document.getElementById("controls"),
  lastUpdated: document.getElementById("last-updated"),
  views: {
    info: document.getElementById("view-info"),
    nextn: document.getElementById("view-nextn"),
    scatter: document.getElementById("view-scatter"),
    season: document.getElementById("view-season"),
  },
};

// Per-round chart specs never change once a round is published, so let the
// browser cache them — otherwise stepping back and forth re-downloads the same JSON.
// Only latest.json is refetched, since it's the pointer that does move.
// Artifacts are served from the same origin as this page (see vercel.json).
async function fetchJSON(path, { revalidate = false } = {}) {
  const res = await fetch(`data/${path}`, {
    cache: revalidate ? "no-store" : "default",
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${path}: ${res.status}`);
  }
  return res.json();
}

function showStatus(container, message) {
  container.innerHTML = `<div class="status">${message}</div>`;
}

// --- Auto-fit chart rendering -------------------------------------------
//
// Charts are precomputed at a single fixed pixel size (see sos/presets.py).
// Instead of shipping separate desktop/mobile variants, we scale the
// rendered chart down to fit whatever space is actually available, and
// re-scale on resize — this is what replaces the old "Mobile layout" toggle.

const scaledCharts = [];

function availableWidth() {
  const main = document.querySelector("main");
  return Math.max(240, main.clientWidth - 40);
}

function applyScale(entry) {
  const scale = Math.min(1, availableWidth() / entry.naturalWidth);
  entry.wrapper.style.transformOrigin = "top left";
  entry.wrapper.style.transform = `scale(${scale})`;
  entry.container.style.width = `${entry.naturalWidth * scale}px`;
  entry.container.style.height = `${entry.naturalHeight * scale}px`;
}

async function embedResponsive(container, spec) {
  container.innerHTML = "";
  container.style.width = "";
  container.style.height = "";
  const wrapper = document.createElement("div");
  container.appendChild(wrapper);

  const result = await vegaEmbed(wrapper, spec, { actions: false });
  const el = wrapper.querySelector("svg, canvas");
  if (!el) return result;

  const rect = el.getBoundingClientRect();
  const entry = { container, wrapper, naturalWidth: rect.width, naturalHeight: rect.height };
  scaledCharts.push(entry);
  applyScale(entry);
  return result;
}

window.addEventListener("resize", () => {
  scaledCharts.forEach(applyScale);
});

// --- View rendering -------------------------------------------------------

async function renderNextN() {
  const container = document.getElementById("chart-nextn");
  showStatus(container, "Loading...");
  try {
    const spec = await fetchJSON(`rounds/${state.round}/next-n/${state.nNext}.json`);
    await embedResponsive(container, spec);
  } catch (err) {
    showStatus(container, `Could not load this view (${err.message}).`);
  }
}

async function renderScatter() {
  const mainEl = document.getElementById("chart-scatter-main");
  const tableEl = document.getElementById("chart-scatter-table");
  showStatus(mainEl, "Loading...");
  tableEl.innerHTML = "";
  try {
    const { main, table } = await fetchJSON(`rounds/${state.round}/scatter.json`);
    await embedResponsive(mainEl, main);
    await embedResponsive(tableEl, table);
  } catch (err) {
    showStatus(mainEl, `Could not load this view (${err.message}).`);
  }
}

async function renderSeason() {
  const container = document.getElementById("chart-season");
  showStatus(container, "Loading...");
  try {
    const spec = await fetchJSON(`rounds/${state.round}/season-table.json`);
    await embedResponsive(container, spec);
  } catch (err) {
    showStatus(container, `Could not load this view (${err.message}).`);
  }
}

const renderers = { nextn: renderNextN, scatter: renderScatter, season: renderSeason };

function renderActiveView() {
  if (state.view === "info" || !state.round) return;
  renderers[state.view]();
}

function setView(view) {
  state.view = view;
  els.navButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.view === view));
  Object.entries(els.views).forEach(([name, el]) => el.classList.toggle("active", name === view));
  // Info / About is static text — neither stepper applies, so drop the whole
  // controls block rather than leaving an empty container.
  els.controls.style.display = view === "info" ? "none" : "";
  els.nControl.style.display = view === "nextn" ? "" : "none";
  renderActiveView();
}

// --- Steppers -------------------------------------------------------------
//
// Round and Next-N move one step at a time rather than by drag: each value is
// a separate precomputed JSON file, so every intermediate value a slider swept
// through was a fetch nobody asked to see.

function initStepper({ decEl, incEl, valueEl, min, max, get, set }) {
  const sync = () => {
    const value = get();
    valueEl.textContent = value;
    decEl.disabled = value <= min;
    incEl.disabled = value >= max;
  };

  const step = (delta) => {
    const next = Math.min(max, Math.max(min, get() + delta));
    if (next === get()) return;
    set(next);
    sync();
    renderActiveView();
  };

  decEl.addEventListener("click", () => step(-1));
  incEl.addEventListener("click", () => step(1));
  sync();
}

function initSteppers(latestRound) {
  initStepper({
    decEl: els.roundDec,
    incEl: els.roundInc,
    valueEl: els.roundValue,
    min: 1,
    max: latestRound,
    get: () => state.round,
    set: (v) => { state.round = v; },
  });

  initStepper({
    decEl: els.nDec,
    incEl: els.nInc,
    valueEl: els.nValue,
    min: 1,
    max: 10,
    get: () => state.nNext,
    set: (v) => { state.nNext = v; },
  });
}

async function init() {
  els.navButtons.forEach((btn) => {
    btn.addEventListener("click", () => setView(btn.dataset.view));
  });

  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
    });
  }

  setView("info");

  try {
    const latest = await fetchJSON("latest.json", { revalidate: true });
    state.latestRound = latest.round;
    state.round = latest.round;
    initSteppers(latest.round);
    if (latest.updated_at) {
      els.lastUpdated.textContent = `Updated ${new Date(latest.updated_at).toLocaleDateString()}`;
    }
  } catch (err) {
    console.error("Failed to load latest.json — has scripts/publish_all.py been run and the output committed?", err);
  }
}

init();
