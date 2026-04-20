import { drawWorld } from "./renderer.js";
import {
  updateCheckpointSelect,
  updateStatus,
  updateHomeostasis,
  updateSensors,
  updateCognition,
} from "./panels.js";

const canvas = document.getElementById("worldCanvas");
const canvasFrame = document.getElementById("canvasFrame");
const ctx = canvas.getContext("2d");

const el = {
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
  // Homeostasis
  energyValue: document.getElementById("energyValue"),
  damageValue: document.getElementById("damageValue"),
  fatigueValue: document.getElementById("fatigueValue"),
  stressValue: document.getElementById("stressValue"),
  surpriseValue: document.getElementById("surpriseValue"),
  confidenceValue: document.getElementById("confidenceValue"),
  returnValue: document.getElementById("returnValue"),
  rewardValue: document.getElementById("rewardValue"),
  actionValue: document.getElementById("actionValue"),
  reflexValue: document.getElementById("reflexValue"),
  energyBar: document.getElementById("energyBar"),
  damageBar: document.getElementById("damageBar"),
  fatigueBar: document.getElementById("fatigueBar"),
  stressBar: document.getElementById("stressBar"),
  surpriseBar: document.getElementById("surpriseBar"),
  confidenceBar: document.getElementById("confidenceBar"),
  // Sensors
  foodSensor: document.getElementById("foodSensor"),
  hazardSensor: document.getElementById("hazardSensor"),
  wallSensor: document.getElementById("wallSensor"),
  shelterSensor: document.getElementById("shelterSensor"),
  internalSensor: document.getElementById("internalSensor"),
  // Cognition
  workspaceDisplay: document.getElementById("workspaceDisplay"),
  ws_food: document.getElementById("wsFood"),
  ws_danger: document.getElementById("wsDanger"),
  ws_shelter: document.getElementById("wsShelter"),
  ws_homeostasis: document.getElementById("wsHomeostasis"),
  memoryDisplay: document.getElementById("memoryDisplay"),
  memoryBar: document.getElementById("memoryBar"),
  ownershipDisplay: document.getElementById("ownershipDisplay"),
};

let appState = null;
let autoplayTimer = null;
let canvasSize = { width: 920, height: 680 };
const autoplayDelay = 130;

// --- API ---

async function api(path, options = {}) {
  const response = await fetch(path, {
    method: options.method || "GET",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status}`);
  }
  return response.json();
}

// --- State Updates ---

function updatePanels(state) {
  const { session, checkpoints } = state;
  appState = state;

  updateCheckpointSelect(el.checkpointSelect, checkpoints, session.checkpoint);
  updateStatus(el, session);
  updateHomeostasis(el, session.episode);
  updateSensors(el, session.sensors);
  updateCognition(el, session.cognition);

  drawWorld(ctx, session, canvasSize, el.sensorsToggle.checked);
}

// --- Canvas ---

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
    drawWorld(ctx, appState.session, canvasSize, el.sensorsToggle.checked);
  }
}

// --- Actions ---

async function refreshState() {
  const state = await api("/api/state");
  updatePanels(state);
}

async function stepPolicy() {
  if (!appState?.session.has_policy) {
    el.statusText.textContent = "Load a checkpoint to use policy stepping.";
    return;
  }
  const state = await api("/api/step/policy", { method: "POST" });
  updatePanels({ checkpoints: appState.checkpoints, session: state });
}

async function manualStep(action) {
  const state = await api("/api/step/manual", { method: "POST", body: { action } });
  updatePanels({ checkpoints: appState.checkpoints, session: state });
}

async function resetEpisode(seed = null) {
  const state = await api("/api/reset", { method: "POST", body: { seed } });
  updatePanels({ checkpoints: appState.checkpoints, session: state });
}

function setAutoplay(enabled) {
  if (autoplayTimer) {
    window.clearInterval(autoplayTimer);
    autoplayTimer = null;
  }
  if (!enabled) {
    el.autoplayBtn.textContent = "Autoplay";
    return;
  }
  el.autoplayBtn.textContent = "Pause";
  autoplayTimer = window.setInterval(async () => {
    if (appState?.session.episode.done) {
      setAutoplay(false);
      return;
    }
    try {
      await stepPolicy();
    } catch (error) {
      setAutoplay(false);
      el.statusText.textContent = error.message;
    }
  }, autoplayDelay);
}

async function loadCheckpoint(path) {
  const state = await api("/api/load-checkpoint", { method: "POST", body: { checkpoint: path || null } });
  updatePanels({ checkpoints: appState.checkpoints, session: state });
}

async function setDeterministic(deterministic) {
  const state = await api("/api/options", { method: "POST", body: { deterministic } });
  updatePanels({ checkpoints: appState.checkpoints, session: state });
}

// --- Events ---

function showError(error) {
  console.error(error);
  el.statusText.textContent = error.message || "Request failed.";
}

function bindEvents() {
  el.policyStepBtn.addEventListener("click", () => stepPolicy().catch(showError));
  el.autoplayBtn.addEventListener("click", () => setAutoplay(autoplayTimer === null));
  el.applySeedBtn.addEventListener("click", () => {
    const value = Number.parseInt(el.seedInput.value, 10);
    resetEpisode(Number.isNaN(value) ? null : value).catch(showError);
  });
  el.randomSeedBtn.addEventListener("click", () => {
    const randomSeed = Math.floor(Math.random() * 1_000_000);
    el.seedInput.value = randomSeed;
    resetEpisode(randomSeed).catch(showError);
  });
  el.resetBtn.addEventListener("click", () => {
    const current = Number.parseInt(el.seedInput.value, 10);
    resetEpisode(Number.isNaN(current) ? null : current).catch(showError);
  });
  el.deterministicToggle.addEventListener("change", (e) => setDeterministic(e.target.checked).catch(showError));
  el.checkpointSelect.addEventListener("change", (e) => loadCheckpoint(e.target.value).catch(showError));
  document.querySelectorAll("[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => manualStep(btn.dataset.action).catch(showError));
  });
  window.addEventListener("keydown", (event) => {
    if (event.target.tagName === "INPUT" || event.target.tagName === "SELECT") return;
    if (event.code === "Space") { event.preventDefault(); el.autoplayBtn.click(); return; }
    const map = { ArrowUp: "forward", ArrowLeft: "turn_left", ArrowRight: "turn_right", KeyE: "eat", KeyR: "rest", Enter: "policy" };
    const action = map[event.code];
    if (!action) return;
    event.preventDefault();
    if (action === "policy") el.policyStepBtn.click();
    else manualStep(action).catch(showError);
  });
}

// --- Init ---

const observer = new ResizeObserver(() => resizeCanvas());
observer.observe(canvasFrame);
window.addEventListener("resize", resizeCanvas);
bindEvents();
resizeCanvas();
refreshState().catch(showError);
