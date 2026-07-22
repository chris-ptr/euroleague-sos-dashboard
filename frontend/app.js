// Fill in your Supabase project URL after creating the project (see setup docs).
// This is NOT a secret — it's the public base URL for the read-only "sos-public" bucket.
const CONFIG = {
  supabaseUrl: "https://ncvzkxidwkrpygszgsvh.supabase.co",
  publicBucket: "sos-public",
};

const state = {
  view: "info",
  round: null,
  latestRound: null,
  nNext: 5,
  mobile: window.matchMedia("(max-width: 700px)").matches,
};

const els = {
  navButtons: document.querySelectorAll(".nav-btn"),
  roundSelect: document.getElementById("round-select"),
  nSelect: document.getElementById("n-select"),
  nControl: document.getElementById("n-next-control"),
  mobileToggle: document.getElementById("mobile-toggle"),
  views: {
    info: document.getElementById("view-info"),
    nextn: document.getElementById("view-nextn"),
    scatter: document.getElementById("view-scatter"),
    season: document.getElementById("view-season"),
  },
};

function publicBase() {
  return `${CONFIG.supabaseUrl}/storage/v1/object/public/${CONFIG.publicBucket}`;
}

async function fetchJSON(path) {
  const res = await fetch(`${publicBase()}/${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${path}: ${res.status}`);
  }
  return res.json();
}

function variant() {
  return state.mobile ? "mobile" : "desktop";
}

function showStatus(container, message) {
  container.innerHTML = `<div class="status">${message}</div>`;
}

async function renderNextN() {
  const container = document.getElementById("chart-nextn");
  showStatus(container, "Loading...");
  try {
    const spec = await fetchJSON(`rounds/${state.round}/next-n/${state.nNext}/${variant()}.json`);
    container.innerHTML = "";
    await vegaEmbed(container, spec, { actions: false });
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
    const { main, table } = await fetchJSON(`rounds/${state.round}/scatter/${variant()}.json`);
    mainEl.innerHTML = "";
    await vegaEmbed(mainEl, main, { actions: false });
    await vegaEmbed(tableEl, table, { actions: false });
  } catch (err) {
    showStatus(mainEl, `Could not load this view (${err.message}).`);
  }
}

async function renderSeason() {
  const container = document.getElementById("chart-season");
  showStatus(container, "Loading...");
  try {
    const spec = await fetchJSON(`rounds/${state.round}/season-table/${variant()}.json`);
    container.innerHTML = "";
    await vegaEmbed(container, spec, { actions: false });
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
  els.nControl.style.display = view === "nextn" ? "" : "none";
  renderActiveView();
}

function populateRoundSelect(latestRound) {
  els.roundSelect.innerHTML = "";
  for (let r = latestRound; r >= 1; r--) {
    const opt = document.createElement("option");
    opt.value = r;
    opt.textContent = `Round ${r}`;
    els.roundSelect.appendChild(opt);
  }
  els.roundSelect.value = latestRound;
}

async function init() {
  els.navButtons.forEach((btn) => {
    btn.addEventListener("click", () => setView(btn.dataset.view));
  });
  els.roundSelect.addEventListener("change", () => {
    state.round = parseInt(els.roundSelect.value, 10);
    renderActiveView();
  });
  els.nSelect.addEventListener("change", () => {
    state.nNext = parseInt(els.nSelect.value, 10);
    renderActiveView();
  });
  els.mobileToggle.checked = state.mobile;
  els.mobileToggle.addEventListener("change", () => {
    state.mobile = els.mobileToggle.checked;
    renderActiveView();
  });

  setView("info");

  try {
    const latest = await fetchJSON("latest.json");
    state.latestRound = latest.round;
    state.round = latest.round;
    populateRoundSelect(latest.round);
  } catch (err) {
    console.error("Failed to load latest.json — has the data been published to Supabase yet?", err);
  }
}

init();
