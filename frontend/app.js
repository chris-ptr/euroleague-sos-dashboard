const state = {
  view: "info",
  round: null,
  latestRound: null,
  nNext: 5,
  // Publish timestamp from latest.json, appended to every spec request so a
  // recompute invalidates hard-cached artifacts. See fetchJSON.
  version: null,
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

// Per-round chart specs are cached hard by the browser and the CDN (see
// vercel.json) so that stepping back and forth doesn't re-download the same
// JSON. That is only safe while the URL changes whenever the content does —
// and a full recompute (scripts/recompute_all.py) rewrites every spec in place
// under the same path. So each request carries the publish timestamp from
// latest.json, which is the one file always revalidated, as a version token.
//
// Without it a browser that cached the pre-recompute spec would keep serving it
// for a year: the Cache-Control is `immutable`, so it never revalidates and
// never even learns the file changed. Only latest.json moving is not enough —
// that just tells the app which round to ask for, not that the round's contents
// were rewritten.
async function fetchJSON(path, { revalidate = false } = {}) {
  const url = state.version && !revalidate
    ? `data/${path}?v=${encodeURIComponent(state.version)}`
    : `data/${path}`;
  const res = await fetch(url, {
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
//
// Below MIN_SCALE the chart is still legible but the labels are getting small,
// so shrinking stops there and the leftover width becomes a horizontal swipe
// instead. A phone showing a ~1000px-wide table at 0.35 is technically "fitted"
// and completely unreadable; scroll is the better trade.
const MIN_SCALE = 0.62;

// Keyed by container so re-rendering a view (round/N change) replaces its entry
// rather than leaving a stale one that keeps writing to the same element.
const scaledCharts = new Map();

function horizontalInsets(el) {
  const cs = getComputedStyle(el);
  return (
    parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight) +
    parseFloat(cs.borderLeftWidth) + parseFloat(cs.borderRightWidth)
  );
}

// The space a chart can actually occupy: <main>'s content box, less the padding
// and borders of everything between it and the chart (the panel card, mainly).
// Measured off <main> rather than off the container's own parent because the
// panel is shrink-to-fit — its width is a consequence of the chart's, so asking
// it how much room there is would just echo back the last size we set.
function availableWidth(container) {
  const main = document.querySelector("main");
  let width = main.clientWidth - horizontalInsets(main);
  for (let el = container.parentElement; el && el !== main; el = el.parentElement) {
    width -= horizontalInsets(el);
  }
  return Math.max(200, width);
}

function applyScale(entry) {
  const available = availableWidth(entry.container);
  const scale = Math.min(1, Math.max(MIN_SCALE, available / entry.naturalWidth));
  const width = Math.ceil(entry.naturalWidth * scale);
  const height = Math.ceil(entry.naturalHeight * scale);

  entry.wrapper.style.transform = `scale(${scale})`;
  // A transform doesn't change layout, so the untransformed wrapper would still
  // reserve its full natural box. The sizer carries the scaled dimensions and
  // is what the panel and the scroll port measure against.
  entry.sizer.style.width = `${width}px`;
  entry.sizer.style.height = `${height}px`;
  // The container is the scroll port: never wider than the room we have, so a
  // clamped chart can't push the page sideways — it scrolls inside the card.
  entry.container.style.width = `${Math.min(width, available)}px`;
  entry.container.classList.toggle("is-scrollable", width > available);

  // A classic, space-taking scrollbar is laid out inside the port, over the
  // bottom of the chart — and overflow-y: hidden would then clip what it
  // covers. Hand back exactly the height it claims. Overlay scrollbars (touch,
  // macOS) measure zero, so this is a no-op on the devices that have them.
  entry.container.style.paddingBottom = "";
  if (width > available) {
    const bar = entry.container.offsetHeight - entry.container.clientHeight;
    if (bar > 0) entry.container.style.paddingBottom = `${bar}px`;
  }
}

// Touch scrollbars are overlays that only appear once you are already
// scrolling, so a clamped chart would just look cropped. The hint says
// otherwise, and retires itself the moment it has been acted on.
function onChartScroll(event) {
  event.currentTarget.classList.add("scrolled");
}

async function embedResponsive(container, spec) {
  scaledCharts.delete(container);
  container.classList.remove("is-scrollable", "scrolled");
  container.innerHTML = "";
  container.style.width = "";
  // Same handler reference every time, so repeated renders of one container
  // don't stack listeners.
  container.addEventListener("scroll", onChartScroll, { passive: true });

  const sizer = document.createElement("div");
  // Clips nothing — the scaled chart is exactly the size the sizer is given —
  // but it stops the wrapper's untransformed layout box (still the chart's full
  // natural width) from counting toward the scroll port's scrollable area,
  // which would otherwise put a scrollbar and a stretch of dead space under
  // every chart that had been scaled down at all.
  sizer.style.overflow = "hidden";
  const wrapper = document.createElement("div");
  wrapper.style.transformOrigin = "top left";
  sizer.appendChild(wrapper);
  container.appendChild(sizer);

  const result = await vegaEmbed(wrapper, spec, { actions: false });
  const el = wrapper.querySelector("svg, canvas");
  if (!el) return result;

  const entry = { container, sizer, wrapper, el, naturalWidth: 0, naturalHeight: 0 };
  scaledCharts.set(container, entry);
  measure(entry);
  applyScale(entry);
  return result;
}

// Natural size has to be read with the transform off, or a re-measure would
// scale down what is already scaled.
function measure(entry) {
  entry.wrapper.style.transform = "none";
  const rect = entry.el.getBoundingClientRect();
  if (rect.width) {
    entry.naturalWidth = rect.width;
    entry.naturalHeight = rect.height;
  }
}

// Coalesced so a drag-resize or a phone rotation doesn't re-scale per event.
let rescaleFrame = null;
function rescaleAll() {
  if (rescaleFrame !== null) return;
  rescaleFrame = requestAnimationFrame(() => {
    rescaleFrame = null;
    scaledCharts.forEach(applyScale);
  });
}

window.addEventListener("resize", rescaleAll);
window.addEventListener("orientationchange", rescaleAll);
// Catches the layout changes `resize` doesn't report — the sidebar collapsing
// to a top bar at the breakpoint, on-screen keyboards, desktop zoom.
if (window.ResizeObserver) {
  new ResizeObserver(rescaleAll).observe(document.querySelector("main"));
}
// A chart is measured with whatever font is available at render time; if
// Geomini swaps in afterwards the text metrics — and so the chart's natural
// width — change, so take the measurement again once the swap has happened.
if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(() => {
    scaledCharts.forEach((entry) => {
      measure(entry);
      applyScale(entry);
    });
  });
}

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
  // The Round value is an <input> so it can be typed into directly; Next-N is
  // still a plain <span>. One code path drives both.
  const editable = valueEl.tagName === "INPUT";

  const sync = () => {
    const value = get();
    if (editable) valueEl.value = value;
    else valueEl.textContent = value;
    decEl.disabled = value <= min;
    incEl.disabled = value >= max;
  };

  // Single funnel for arrows and typing alike, so a typed value can never
  // reach a round that was never published — it is clamped on the way in.
  const apply = (next) => {
    const clamped = Math.min(max, Math.max(min, next));
    if (clamped === get()) {
      // Still re-sync: an out-of-range or malformed entry has to snap back to
      // the value actually in effect rather than sit there looking accepted.
      sync();
      return;
    }
    set(clamped);
    sync();
    renderActiveView();
  };

  const step = (delta) => apply(get() + delta);

  decEl.addEventListener("click", () => step(-1));
  incEl.addEventListener("click", () => step(1));

  if (editable) {
    // 38 rounds means two digits; derived rather than hardcoded so a longer
    // season doesn't silently truncate what can be typed.
    valueEl.maxLength = String(max).length;

    // Commit on Enter or on leaving the field — never per keystroke, or typing
    // "1" on the way to "12" would fetch round 1 and re-render for nothing.
    const commit = () => {
      const parsed = parseInt(valueEl.value, 10);
      apply(Number.isNaN(parsed) ? get() : parsed);
    };

    valueEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        commit();
        valueEl.blur();
      } else if (event.key === "Escape") {
        sync();
        valueEl.blur();
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        step(1);
      } else if (event.key === "ArrowDown") {
        event.preventDefault();
        step(-1);
      }
    });

    valueEl.addEventListener("blur", commit);
    valueEl.addEventListener("focus", () => valueEl.select());
  }

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
    max: 5,
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
    // Set before anything else can fetch a spec: initSteppers only syncs the
    // DOM, and renderActiveView runs on interaction, so no artifact request
    // can slip through unversioned.
    state.version = latest.updated_at || null;
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
