import { drawEvolutionWorld, getWorldCoords } from "./evo_renderer.js";

const canvas = document.getElementById("evoCanvas");
const ctx = canvas.getContext("2d");

let state = null;
let selectedUid = null;
let autoplayTimer = null;
let stepsPerCall = 5;

const el = {
  startBtn:     document.getElementById("evoStartBtn"),
  resetBtn:     document.getElementById("evoResetBtn"),
  speedSlider:  document.getElementById("evoSpeed"),
  speedLabel:   document.getElementById("evoSpeedLabel"),
  seedInput:    document.getElementById("evoSeed"),
  popInput:     document.getElementById("evoPopSize"),
  tickCounter:  document.getElementById("evoTick"),
  aliveCount:   document.getElementById("evoAlive"),
  eggCount:     document.getElementById("evoEggs"),
  genMax:       document.getElementById("evoGenMax"),
  matings:      document.getElementById("evoMatings"),
  born:         document.getElementById("evoBorn"),
  died:         document.getElementById("evoDied"),
  populationTable: document.getElementById("evoPopTable"),
  selectedInfo:    document.getElementById("evoSelectedInfo"),
  traitSpeed:      document.getElementById("traitSpeed"),
  traitSight:      document.getElementById("traitSight"),
  traitSize:       document.getElementById("traitSize"),
};

async function api(path, options = {}) {
  const body = options.body ? JSON.stringify(options.body) : undefined;
  const res = await fetch(path, {
    method: options.method || "GET",
    headers: body ? { "Content-Type": "application/json" } : {},
    body,
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

async function fetchState() {
  state = await api("/api/evolution/state");
  render();
  updatePanels();
}

async function stepEvolution() {
  try {
    state = await api("/api/evolution/step", { method: "POST", body: { steps: stepsPerCall } });
    render();
    updatePanels();
  } catch (e) {
    console.error(e);
  }
}

function render() {
  if (!state) return;
  resizeCanvas();
  drawEvolutionWorld(ctx, state, { width: canvas.width, height: canvas.height }, selectedUid);
}

function resizeCanvas() {
  const container = canvas.parentElement;
  const w = container.clientWidth;
  const h = container.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function updatePanels() {
  if (!state) return;

  if (el.tickCounter) el.tickCounter.textContent = state.tick.toLocaleString();
  if (el.aliveCount) el.aliveCount.textContent = state.alive_count;
  if (el.eggCount) el.eggCount.textContent = state.egg_count;
  if (el.genMax) el.genMax.textContent = state.generation_max;
  if (el.matings) el.matings.textContent = state.mating_events;
  if (el.born) el.born.textContent = state.total_born;
  if (el.died) el.died.textContent = state.total_died;

  updatePopulationTable();
  updateTraitBars();
  updateSelectedInfo();
}

function updatePopulationTable() {
  if (!el.populationTable || !state) return;
  const tbody = el.populationTable.querySelector("tbody");
  if (!tbody) return;
  const sorted = [...state.organisms].sort((a, b) => b.food_eaten - a.food_eaten);
  tbody.innerHTML = sorted.slice(0, 30).map(org => {
    const hsl = `hsl(${org.visual.color_h}, ${Math.round(org.visual.color_s * 100)}%, ${Math.round(org.visual.color_l * 100)}%)`;
    const isSelected = org.uid === selectedUid ? ' class="selected-row"' : "";
    return `<tr${isSelected} data-uid="${org.uid}" style="cursor:pointer">
      <td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${hsl};margin-right:4px;"></span>g${org.generation}</td>
      <td>${org.food_eaten}</td>
      <td>${org.age}</td>
      <td>${org.physical.move_speed.toFixed(3)}</td>
      <td>${org.physical.food_visible_range.toFixed(2)}</td>
      <td>${org.visual.body_size.toFixed(2)}</td>
      <td>${org.visual.shape[0]}</td>
    </tr>`;
  }).join("");

  tbody.querySelectorAll("tr").forEach(row => {
    row.addEventListener("click", () => {
      const uid = parseInt(row.dataset.uid, 10);
      selectedUid = selectedUid === uid ? null : uid;
      render();
      updatePanels();
    });
  });
}

function updateTraitBars() {
  if (!state || !state.trait_stats) return;
  const ts = state.trait_stats;
  _setTraitBar(el.traitSpeed, ts.mean_speed, ts.std_speed, 0.02, 0.12, "Speed");
  _setTraitBar(el.traitSight, ts.mean_sight, ts.std_sight, 0.20, 1.00, "Sight");
  _setTraitBar(el.traitSize, ts.mean_size, ts.std_size, 0.60, 1.80, "Size");
}

function _setTraitBar(container, mean, std, min, max, label) {
  if (!container) return;
  const pct = val => Math.round(((val - min) / (max - min)) * 100);
  const meanPct = Math.max(0, Math.min(100, pct(mean)));
  const lo = Math.max(0, pct(mean - (std || 0)));
  const hi = Math.min(100, pct(mean + (std || 0)));
  container.innerHTML = `
    <div style="font-size:11px;color:#666;margin-bottom:2px">${label}: ${mean.toFixed(3)} ±${(std||0).toFixed(3)}</div>
    <div style="position:relative;height:8px;background:#ddd;border-radius:4px">
      <div style="position:absolute;left:${lo}%;width:${hi-lo}%;height:100%;background:rgba(60,140,80,0.4);border-radius:4px"></div>
      <div style="position:absolute;left:${meanPct}%;width:2px;height:100%;background:#2a7a3a"></div>
    </div>`;
}

function updateSelectedInfo() {
  if (!el.selectedInfo || !state) return;
  const org = selectedUid != null ? state.organisms.find(o => o.uid === selectedUid) : null;
  if (!org) { el.selectedInfo.textContent = "Click an organism to inspect"; return; }
  const p = org.physical; const v = org.visual;
  el.selectedInfo.innerHTML = `
    <b>uid ${org.uid}</b> · gen ${org.generation} · lineage ${org.lineage}<br>
    age ${org.age} · food ${org.food_eaten}<br>
    ♥ ${(org.energy*100).toFixed(0)}%&nbsp; ✕ ${(org.damage*100).toFixed(0)}%&nbsp; ⚡ ${(org.fatigue*100).toFixed(0)}%<br>
    <i>${org.last_action}</i><br>
    <hr style="margin:4px 0">
    speed ${p.move_speed.toFixed(3)} · turn ${p.turn_angle.toFixed(2)}<br>
    sight ${p.food_visible_range.toFixed(2)} · sens ${p.sensor_range.toFixed(2)}<br>
    fov ${p.fov_half_angle.toFixed(2)} · metab ${p.energy_decay.toFixed(4)}<br>
    eat_r ${p.eat_radius.toFixed(2)} · haz_s ${p.hazard_sensitivity.toFixed(2)}<br>
    <hr style="margin:4px 0">
    shape <b>${v.shape}</b> · pattern <b>${v.pattern}</b> · size ${v.body_size.toFixed(2)}`;
}

// Controls
el.startBtn?.addEventListener("click", () => {
  if (autoplayTimer) {
    clearInterval(autoplayTimer);
    autoplayTimer = null;
    el.startBtn.textContent = "▶ Start";
  } else {
    autoplayTimer = setInterval(stepEvolution, 120);
    el.startBtn.textContent = "⏸ Pause";
  }
});

el.resetBtn?.addEventListener("click", async () => {
  const seed = parseInt(el.seedInput?.value || "42", 10);
  const pop = parseInt(el.popInput?.value || "12", 10);
  state = await api("/api/evolution/reset", { method: "POST", body: { seed, population_size: pop } });
  selectedUid = null;
  render();
  updatePanels();
});

el.speedSlider?.addEventListener("input", () => {
  stepsPerCall = parseInt(el.speedSlider.value, 10);
  if (el.speedLabel) el.speedLabel.textContent = stepsPerCall;
});

// Canvas click → select organism
canvas.addEventListener("click", (e) => {
  if (!state) return;
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const [wx, wy] = getWorldCoords(cx, cy, { width: canvas.width, height: canvas.height });
  let best = null; let bestDist = 0.04;
  state.organisms.forEach(org => {
    const dx = org.position[0] - wx;
    const dy = org.position[1] - wy;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < bestDist) { bestDist = dist; best = org.uid; }
  });
  selectedUid = (best != null && best !== selectedUid) ? best : null;
  render();
  updatePanels();
});

window.addEventListener("resize", render);

// Init
fetchState();
