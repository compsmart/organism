/* Panel update logic for meters, sensors, and cognition displays. */

function setBar(element, value) {
  element.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
}

function formatTriplet(triplet) {
  return `L ${triplet.left.toFixed(2)} · C ${triplet.center.toFixed(2)} · R ${triplet.right.toFixed(2)}`;
}

export function updateCheckpointSelect(selectEl, checkpoints, currentCheckpoint) {
  const previous = selectEl.value;
  selectEl.innerHTML = "";

  const manualOption = document.createElement("option");
  manualOption.value = "";
  manualOption.textContent = "Manual mode (no checkpoint)";
  selectEl.appendChild(manualOption);

  checkpoints.forEach((checkpoint) => {
    const option = document.createElement("option");
    option.value = checkpoint.path;
    option.textContent = checkpoint.name;
    selectEl.appendChild(option);
  });

  selectEl.value = currentCheckpoint || previous || "";
}

export function updateStatus(el, session) {
  const { episode } = session;
  el.deterministicToggle.checked = session.deterministic;
  el.seedInput.value = session.seed;
  el.policyMode.textContent = `Policy: ${session.deterministic ? "deterministic" : "stochastic"}`;
  el.runChip.textContent = `Checkpoint: ${session.checkpoint_name || "manual"}`;
  el.seedChip.textContent = `Seed ${session.seed}`;
  el.stepChip.textContent = `Step ${episode.steps}`;

  if (episode.done) {
    const reason = episode.death_reason ? `Episode ended: ${episode.death_reason}.` : "Episode reached max steps.";
    el.statusText.textContent = reason;
  } else {
    el.statusText.textContent = `Last action ${episode.action_name}, reward ${episode.reward.toFixed(3)}.`;
  }
}

export function updateHomeostasis(el, episode) {
  el.energyValue.textContent = episode.energy.toFixed(2);
  el.damageValue.textContent = episode.damage.toFixed(2);
  el.fatigueValue.textContent = episode.fatigue.toFixed(2);
  el.stressValue.textContent = episode.stress.toFixed(2);

  const surprise = episode.surprise || 0;
  el.surpriseValue.textContent = surprise.toFixed(3);

  const confidence = episode.confidence || 0;
  el.confidenceValue.textContent = confidence.toFixed(2);

  el.returnValue.textContent = episode.return.toFixed(3);
  el.rewardValue.textContent = episode.reward.toFixed(3);
  el.actionValue.textContent = episode.action_name;
  el.reflexValue.textContent = episode.reflex_override ? "yes" : "no";

  if (el.foodEatenValue) {
    el.foodEatenValue.textContent = episode.food_eaten || 0;
  }
  if (el.ateFoodValue) {
    el.ateFoodValue.textContent = episode.ate_food ? "YES" : "no";
    el.ateFoodValue.style.color = episode.ate_food ? "#2e6b2e" : "";
    el.ateFoodValue.style.fontWeight = episode.ate_food ? "800" : "";
  }

  setBar(el.energyBar, episode.energy);
  setBar(el.damageBar, episode.damage);
  setBar(el.fatigueBar, episode.fatigue);
  setBar(el.stressBar, Math.min(1, episode.stress));
  setBar(el.surpriseBar, Math.min(1, surprise * 10));
  setBar(el.confidenceBar, confidence);
}

export function updateSensors(el, sensors) {
  el.foodSensor.textContent = formatTriplet(sensors.food);
  el.hazardSensor.textContent = formatTriplet(sensors.hazard);
  el.wallSensor.textContent = formatTriplet(sensors.wall);
  el.shelterSensor.textContent =
    `align ${sensors.shelter.alignment.toFixed(2)} · prox ${sensors.shelter.proximity.toFixed(2)} · contact ${sensors.shelter.contact.toFixed(0)}`;
  el.internalSensor.textContent =
    `energy ${sensors.internal.energy.toFixed(2)} · damage ${sensors.internal.damage.toFixed(2)} · fatigue ${sensors.internal.fatigue.toFixed(2)} · novelty ${sensors.internal.novelty.toFixed(2)}`;
}

export function updateCognition(el, cognition) {
  if (!cognition) return;

  // Workspace attention
  const weights = cognition.workspace_weights || [];
  const channels = cognition.workspace_channels || [];
  if (weights.length > 0 && el.workspaceDisplay) {
    const parts = channels.map((ch, i) => {
      const w = weights[i] || 0;
      return `${ch} ${(w * 100).toFixed(0)}%`;
    });
    el.workspaceDisplay.textContent = parts.join(" · ");

    // Update workspace bars
    channels.forEach((ch, i) => {
      const bar = el[`ws_${ch}`];
      if (bar) setBar(bar, weights[i] || 0);
    });
  }

  // Memory utilization
  if (el.memoryDisplay) {
    const used = cognition.memory_slots_used || 0;
    const total = cognition.memory_slots_total || 0;
    el.memoryDisplay.textContent = `${used} / ${total} slots`;
    if (el.memoryBar && total > 0) {
      setBar(el.memoryBar, used / total);
    }
  }

  // Ownership (self-model)
  const own = cognition.ownership || {};
  if (el.ownershipDisplay) {
    const e = (own.ownership_energy || 0).toFixed(4);
    const d = (own.ownership_damage || 0).toFixed(4);
    const f = (own.ownership_fatigue || 0).toFixed(4);
    el.ownershipDisplay.textContent = `energy ${e} · damage ${d} · fatigue ${f}`;
  }

  // Narration (introspective report)
  const narr = cognition.narration || {};
  if (el.narrationFocus) {
    el.narrationFocus.textContent = narr.focus || "--";
  }
  if (el.narrationIntent) {
    el.narrationIntent.textContent = narr.intent || "--";
  }
}
