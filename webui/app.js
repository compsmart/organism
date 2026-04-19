const canvas = document.getElementById("worldCanvas");
const canvasFrame = document.getElementById("canvasFrame");
const ctx = canvas.getContext("2d");

const elements = {
  checkpointSelect: document.getElementById("checkpointSelect"),
  seedInput: document.getElementById("seedInput"),
  applySeedBtn: document.getElementById("applySeedBtn"),
  randomSeedBtn: document.getElementById("randomSeedBtn"),
  resetBtn: document.getElementById("resetBtn"),
  deterministicToggle: document.getElementById("deterministicToggle"),
  sensorsToggle: document.getElementById("sensorsToggle"),
  policyStepBtn: document.getElementById("policyStepBtn"),
  autoplayBtn: document.getElementById("autoplayBtn"),
  statusText: document.getElementById("statusText"),
  policyMode: document.getElementById("policyMode"),
  runChip: document.getElementById("runChip"),
  seedChip: document.getElementById("seedChip"),
  stepChip: document.getElementById("stepChip"),
  energyValue: document.getElementById("energyValue"),
  damageValue: document.getElementById("damageValue"),
  fatigueValue: document.getElementById("fatigueValue"),
  stressValue: document.getElementById("stressValue"),
  returnValue: document.getElementById("returnValue"),
  rewardValue: document.getElementById("rewardValue"),
  actionValue: document.getElementById("actionValue"),
  reflexValue: document.getElementById("reflexValue"),
  energyBar: document.getElementById("energyBar"),
  damageBar: document.getElementById("damageBar"),
  fatigueBar: document.getElementById("fatigueBar"),
  stressBar: document.getElementById("stressBar"),
  foodSensor: document.getElementById("foodSensor"),
  hazardSensor: document.getElementById("hazardSensor"),
  wallSensor: document.getElementById("wallSensor"),
  shelterSensor: document.getElementById("shelterSensor"),
  internalSensor: document.getElementById("internalSensor"),
};

let appState = null;
let autoplayTimer = null;
let canvasSize = { width: 920, height: 680 };
const autoplayDelay = 130;

function formatTriplet(triplet) {
  return `L ${triplet.left.toFixed(2)} · C ${triplet.center.toFixed(2)} · R ${triplet.right.toFixed(2)}`;
}

function setBar(element, value) {
  element.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    method: options.method || "GET",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status}`);
  }
  return response.json();
}

function updateCheckpointSelect(checkpoints, currentCheckpoint) {
  const previous = elements.checkpointSelect.value;
  elements.checkpointSelect.innerHTML = "";

  const manualOption = document.createElement("option");
  manualOption.value = "";
  manualOption.textContent = "Manual mode (no checkpoint)";
  elements.checkpointSelect.appendChild(manualOption);

  checkpoints.forEach((checkpoint) => {
    const option = document.createElement("option");
    option.value = checkpoint.path;
    option.textContent = checkpoint.name;
    elements.checkpointSelect.appendChild(option);
  });

  elements.checkpointSelect.value = currentCheckpoint || previous || "";
}

function updatePanels(state) {
  const { session, checkpoints } = state;
  const { episode, sensors } = session;

  appState = state;
  updateCheckpointSelect(checkpoints, session.checkpoint);

  elements.deterministicToggle.checked = session.deterministic;
  elements.seedInput.value = session.seed;
  elements.policyMode.textContent = `Policy: ${session.deterministic ? "deterministic" : "stochastic"}`;
  elements.runChip.textContent = `Checkpoint: ${session.checkpoint_name || "manual"}`;
  elements.seedChip.textContent = `Seed ${session.seed}`;
  elements.stepChip.textContent = `Step ${episode.steps}`;

  if (episode.done) {
    const reason = episode.death_reason ? `Episode ended: ${episode.death_reason}.` : "Episode reached max steps.";
    elements.statusText.textContent = reason;
  } else {
    elements.statusText.textContent = `Last action ${episode.action_name}, reward ${episode.reward.toFixed(3)}.`;
  }

  elements.energyValue.textContent = episode.energy.toFixed(2);
  elements.damageValue.textContent = episode.damage.toFixed(2);
  elements.fatigueValue.textContent = episode.fatigue.toFixed(2);
  elements.stressValue.textContent = episode.stress.toFixed(2);
  elements.returnValue.textContent = episode.return.toFixed(3);
  elements.rewardValue.textContent = episode.reward.toFixed(3);
  elements.actionValue.textContent = episode.action_name;
  elements.reflexValue.textContent = episode.reflex_override ? "yes" : "no";

  setBar(elements.energyBar, episode.energy);
  setBar(elements.damageBar, episode.damage);
  setBar(elements.fatigueBar, episode.fatigue);
  setBar(elements.stressBar, Math.min(1, episode.stress));

  elements.foodSensor.textContent = formatTriplet(sensors.food);
  elements.hazardSensor.textContent = formatTriplet(sensors.hazard);
  elements.wallSensor.textContent = formatTriplet(sensors.wall);
  elements.shelterSensor.textContent =
    `align ${sensors.shelter.alignment.toFixed(2)} · prox ${sensors.shelter.proximity.toFixed(2)} · contact ${sensors.shelter.contact.toFixed(0)}`;
  elements.internalSensor.textContent =
    `energy ${sensors.internal.energy.toFixed(2)} · damage ${sensors.internal.damage.toFixed(2)} · fatigue ${sensors.internal.fatigue.toFixed(2)} · novelty ${sensors.internal.novelty.toFixed(2)}`;

  drawWorld(session);
}

function resizeCanvas() {
  const rect = canvasFrame.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  canvasSize = { width: rect.width, height: rect.height };
  if (appState) {
    drawWorld(appState.session);
  }
}

function getSceneRect(width, height) {
  const outerPad = 18;
  const usableWidth = Math.max(120, width - outerPad * 2);
  const usableHeight = Math.max(120, height - outerPad * 2);
  const side = Math.max(120, Math.min(usableWidth, usableHeight));
  const left = outerPad + (usableWidth - side) / 2;
  const top = outerPad + (usableHeight - side) / 2;
  return { left, top, size: side };
}

function worldToCanvas(point, scene) {
  return {
    x: scene.left + point[0] * scene.size,
    y: scene.top + (1 - point[1]) * scene.size,
  };
}

function drawWorld(session) {
  const { world, episode, sensors } = session;
  const width = canvasSize.width;
  const height = canvasSize.height;
  const scene = getSceneRect(width, height);

  ctx.clearRect(0, 0, width, height);

  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, "#eee7d7");
  bg.addColorStop(1, "#d9d0bb");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = "rgba(255,255,255,0.24)";
  ctx.fillRect(scene.left, scene.top, scene.size, scene.size);

  drawVisitation(world, scene);
  drawGrid(scene);
  drawShelter(world, scene);
  drawFood(world, scene);
  drawHazards(world, scene);
  drawTrail(world, scene);
  if (elements.sensorsToggle.checked) {
    drawSensorRays(world, sensors, scene);
  }
  drawAgent(world, episode, scene);
  drawSceneBorder(scene);
  drawHud(session, width, height, scene);
}

function drawVisitation(world, scene) {
  const visitRows = world.visitation.length;
  const visitCols = world.visitation[0].length;
  const visitWidth = scene.size / visitCols;
  const visitHeight = scene.size / visitRows;
  const maxVisit = Math.max(1, ...world.visitation.flat());

  for (let y = 0; y < visitRows; y += 1) {
    for (let x = 0; x < visitCols; x += 1) {
      const visits = world.visitation[y][x];
      if (!visits) continue;
      const alpha = Math.min(1, visits / maxVisit) * 0.5;
      ctx.fillStyle = `rgba(82, 118, 124, ${alpha})`;
      ctx.fillRect(
        scene.left + x * visitWidth,
        scene.top + scene.size - (y + 1) * visitHeight,
        visitWidth,
        visitHeight
      );
    }
  }
}

function drawGrid(scene) {
  ctx.strokeStyle = "rgba(53, 66, 54, 0.08)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i += 1) {
    const offset = (scene.size / 10) * i;
    ctx.beginPath();
    ctx.moveTo(scene.left + offset, scene.top);
    ctx.lineTo(scene.left + offset, scene.top + scene.size);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(scene.left, scene.top + offset);
    ctx.lineTo(scene.left + scene.size, scene.top + offset);
    ctx.stroke();
  }
}

function drawShelter(world, scene) {
  const point = worldToCanvas(world.shelter, scene);
  const radius = world.shelter_radius * scene.size;
  ctx.fillStyle = "rgba(140, 188, 155, 0.42)";
  ctx.strokeStyle = "#4f6f59";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#254536";
  ctx.font = "700 16px Georgia";
  ctx.textAlign = "center";
  ctx.fillText("S", point.x, point.y + 5);
}

function drawFood(world, scene) {
  world.food.forEach((food) => {
    const point = worldToCanvas(food, scene);
    const radius = Math.max(5, world.eat_radius * scene.size * 0.8);
    ctx.fillStyle = "#67a35c";
    ctx.strokeStyle = "#2e6b2e";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
}

function drawHazards(world, scene) {
  world.hazards.forEach((hazard) => {
    const point = worldToCanvas(hazard, scene);
    const radius = world.hazard_radius * scene.size;
    ctx.fillStyle = "rgba(204, 110, 88, 0.48)";
    ctx.strokeStyle = "#913a2e";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#7d241c";
    ctx.font = "700 14px Georgia";
    ctx.textAlign = "center";
    ctx.fillText("!", point.x, point.y + 5);
  });
}

function drawTrail(world, scene) {
  if (world.trail.length < 2) return;
  ctx.strokeStyle = "rgba(85, 127, 160, 0.84)";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  world.trail.forEach((point, index) => {
    const canvasPoint = worldToCanvas(point, scene);
    if (index === 0) {
      ctx.moveTo(canvasPoint.x, canvasPoint.y);
    } else {
      ctx.lineTo(canvasPoint.x, canvasPoint.y);
    }
  });
  ctx.stroke();
}

function drawSensorRays(world, sensors, scene) {
  const origin = worldToCanvas(world.agent, scene);
  const sets = [
    { values: sensors.food, color: "rgba(87, 150, 86, 0.65)" },
    { values: sensors.hazard, color: "rgba(181, 84, 66, 0.65)" },
    { values: sensors.wall, color: "rgba(92, 92, 92, 0.48)" },
  ];

  sets.forEach((set) => {
    world.sector_offsets.forEach((offset, index) => {
      const value = index === 0 ? set.values.left : index === 1 ? set.values.center : set.values.right;
      const angle = world.heading + offset;
      const reach = world.sensor_range * scene.size * value;
      const endX = origin.x + Math.cos(angle) * reach;
      const endY = origin.y - Math.sin(angle) * reach;
      ctx.save();
      ctx.strokeStyle = set.color;
      ctx.lineWidth = 2;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      ctx.moveTo(origin.x, origin.y);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      ctx.restore();
    });
  });
}

function drawAgent(world, episode, scene) {
  const point = worldToCanvas(world.agent, scene);
  const radius = Math.max(8, scene.size * 0.018);

  ctx.fillStyle = episode.done ? "#6c7278" : "#294f68";
  ctx.strokeStyle = "#0f2330";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  const arrowLength = Math.max(20, scene.size * 0.05);
  const endX = point.x + Math.cos(world.heading) * arrowLength;
  const endY = point.y - Math.sin(world.heading) * arrowLength;
  ctx.strokeStyle = "#0d1923";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(point.x, point.y);
  ctx.lineTo(endX, endY);
  ctx.stroke();

  const headSize = Math.max(5, scene.size * 0.012);
  const left = {
    x: endX - Math.cos(world.heading - 0.45) * headSize,
    y: endY + Math.sin(world.heading - 0.45) * headSize,
  };
  const right = {
    x: endX - Math.cos(world.heading + 0.45) * headSize,
    y: endY + Math.sin(world.heading + 0.45) * headSize,
  };
  ctx.fillStyle = "#0d1923";
  ctx.beginPath();
  ctx.moveTo(endX, endY);
  ctx.lineTo(left.x, left.y);
  ctx.lineTo(right.x, right.y);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "#11212c";
  ctx.font = `600 ${Math.max(11, scene.size * 0.022)}px Aptos`;
  ctx.textAlign = "center";
  const label = episode.reflex_override ? `${episode.action_name} · reflex` : episode.action_name;
  ctx.fillText(label, point.x, point.y - radius - 12);
}

function drawSceneBorder(scene) {
  ctx.strokeStyle = "rgba(48, 59, 47, 0.28)";
  ctx.lineWidth = 2;
  ctx.strokeRect(scene.left, scene.top, scene.size, scene.size);
}

function drawHud(session, width, height, scene) {
  const { episode } = session;
  const boxWidth = Math.min(250, width * 0.34);
  const boxHeight = 72;
  const x = Math.min(width - boxWidth - 14, scene.left + scene.size - boxWidth - 10);
  const y = scene.top + 10;

  ctx.fillStyle = "rgba(22, 28, 22, 0.72)";
  roundRect(ctx, x, y, boxWidth, boxHeight, 16);
  ctx.fill();

  ctx.fillStyle = "#f5efe4";
  ctx.textAlign = "left";
  ctx.font = "700 15px Georgia";
  ctx.fillText(session.checkpoint_name || "manual", x + 14, y + 24);
  ctx.font = "13px Aptos";
  ctx.fillText(`return ${episode.return.toFixed(3)} · reward ${episode.reward.toFixed(3)}`, x + 14, y + 45);
  ctx.fillText(`energy ${episode.energy.toFixed(2)} · fatigue ${episode.fatigue.toFixed(2)}`, x + 14, y + 62);

  if (episode.done) {
    const bannerWidth = Math.min(300, width - 28);
    ctx.fillStyle = "rgba(145, 43, 31, 0.9)";
    roundRect(ctx, 14, height - 58, bannerWidth, 42, 14);
    ctx.fill();
    ctx.fillStyle = "#fff1eb";
    ctx.font = "700 13px Aptos";
    const text = episode.death_reason ? `Episode ended: ${episode.death_reason}` : "Episode reached max steps";
    ctx.fillText(text, 28, height - 31);
  }
}

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.arcTo(x + width, y, x + width, y + height, radius);
  context.arcTo(x + width, y + height, x, y + height, radius);
  context.arcTo(x, y + height, x, y, radius);
  context.arcTo(x, y, x + width, y, radius);
  context.closePath();
}

async function refreshState() {
  const state = await api("/api/state");
  updatePanels(state);
}

async function stepPolicy() {
  if (!appState?.session.has_policy) {
    elements.statusText.textContent = "Load a checkpoint to use policy stepping.";
    return;
  }
  const state = await api("/api/step/policy", { method: "POST" });
  updatePanels({ checkpoints: appState.checkpoints, session: state });
}

async function manualStep(action) {
  const state = await api("/api/step/manual", {
    method: "POST",
    body: { action },
  });
  updatePanels({ checkpoints: appState.checkpoints, session: state });
}

async function resetEpisode(seed = null) {
  const state = await api("/api/reset", {
    method: "POST",
    body: { seed },
  });
  updatePanels({ checkpoints: appState.checkpoints, session: state });
}

function setAutoplay(enabled) {
  if (autoplayTimer) {
    window.clearInterval(autoplayTimer);
    autoplayTimer = null;
  }
  if (!enabled) {
    elements.autoplayBtn.textContent = "Autoplay";
    return;
  }
  elements.autoplayBtn.textContent = "Pause";
  autoplayTimer = window.setInterval(async () => {
    if (appState?.session.episode.done) {
      setAutoplay(false);
      return;
    }
    try {
      await stepPolicy();
    } catch (error) {
      setAutoplay(false);
      elements.statusText.textContent = error.message;
    }
  }, autoplayDelay);
}

async function loadCheckpoint(path) {
  const state = await api("/api/load-checkpoint", {
    method: "POST",
    body: { checkpoint: path || null },
  });
  updatePanels({ checkpoints: appState.checkpoints, session: state });
}

async function setDeterministic(deterministic) {
  const state = await api("/api/options", {
    method: "POST",
    body: { deterministic },
  });
  updatePanels({ checkpoints: appState.checkpoints, session: state });
}

function bindEvents() {
  elements.policyStepBtn.addEventListener("click", () => stepPolicy().catch(showError));
  elements.autoplayBtn.addEventListener("click", () => {
    const active = autoplayTimer !== null;
    setAutoplay(!active);
  });
  elements.applySeedBtn.addEventListener("click", () => {
    const value = Number.parseInt(elements.seedInput.value, 10);
    resetEpisode(Number.isNaN(value) ? null : value).catch(showError);
  });
  elements.randomSeedBtn.addEventListener("click", () => {
    const randomSeed = Math.floor(Math.random() * 1_000_000);
    elements.seedInput.value = randomSeed;
    resetEpisode(randomSeed).catch(showError);
  });
  elements.resetBtn.addEventListener("click", () => {
    const current = Number.parseInt(elements.seedInput.value, 10);
    resetEpisode(Number.isNaN(current) ? null : current).catch(showError);
  });
  elements.deterministicToggle.addEventListener("change", (event) => {
    setDeterministic(event.target.checked).catch(showError);
  });
  elements.checkpointSelect.addEventListener("change", (event) => {
    loadCheckpoint(event.target.value).catch(showError);
  });
  document.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => manualStep(button.dataset.action).catch(showError));
  });
  window.addEventListener("keydown", (event) => {
    if (event.target.tagName === "INPUT" || event.target.tagName === "SELECT") return;
    if (event.code === "Space") {
      event.preventDefault();
      elements.autoplayBtn.click();
      return;
    }
    const map = {
      ArrowUp: "forward",
      ArrowLeft: "turn_left",
      ArrowRight: "turn_right",
      KeyE: "eat",
      KeyR: "rest",
      Enter: "policy",
    };
    const action = map[event.code];
    if (!action) return;
    event.preventDefault();
    if (action === "policy") {
      elements.policyStepBtn.click();
    } else {
      manualStep(action).catch(showError);
    }
  });
}

function showError(error) {
  console.error(error);
  elements.statusText.textContent = error.message || "Request failed.";
}

function attachResize() {
  const observer = new ResizeObserver(() => resizeCanvas());
  observer.observe(canvasFrame);
  window.addEventListener("resize", resizeCanvas);
}

async function init() {
  attachResize();
  bindEvents();
  resizeCanvas();
  await refreshState();
}

init().catch(showError);
