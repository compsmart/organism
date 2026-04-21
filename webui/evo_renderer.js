/* Canvas rendering for the evolution world view. */

const BASE_ORGANISM_RADIUS = 10;

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

function drawGrid(ctx, scene) {
  ctx.strokeStyle = "rgba(53, 66, 54, 0.06)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i++) {
    const offset = (scene.size / 10) * i;
    ctx.beginPath(); ctx.moveTo(scene.left + offset, scene.top); ctx.lineTo(scene.left + offset, scene.top + scene.size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(scene.left, scene.top + offset); ctx.lineTo(scene.left + scene.size, scene.top + offset); ctx.stroke();
  }
}

function drawShelter(ctx, world, scene) {
  const p = worldToCanvas(world.shelter, scene);
  const r = world.shelter_radius * scene.size;
  ctx.fillStyle = "rgba(140, 188, 155, 0.30)";
  ctx.strokeStyle = "#4f6f59";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#254536"; ctx.font = "700 14px Georgia"; ctx.textAlign = "center";
  ctx.fillText("S", p.x, p.y + 5);
}

function drawFood(ctx, world, scene) {
  const values = world.food_values || [];
  const quantities = world.food_quantities || [];
  const maxQtys = world.food_max_quantities || [];
  world.food.forEach((pos, i) => {
    const p = worldToCanvas(pos, scene);
    const v = values[i] ?? 1.0;
    const qty = quantities[i] ?? 1.0;
    const maxQty = maxQtys[i] ?? qty;
    const fraction = Math.max(0.12, qty / Math.max(maxQty, 0.01));
    const r = Math.max(3, world.eat_radius * scene.size * 0.9 * Math.sqrt(fraction) * Math.sqrt(v));
    ctx.fillStyle = v >= 2.0 ? "#3c9048" : v >= 1.4 ? "#56a756" : v >= 0.8 ? "#82b96c" : "#b8cc8c";
    ctx.strokeStyle = "#2e6b2e"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    if (v >= 2.0 && fraction > 0.3) {
      ctx.fillStyle = "#ffe88a"; ctx.font = `700 ${Math.max(7, r * 0.9)}px Georgia`;
      ctx.textAlign = "center"; ctx.fillText("★", p.x, p.y + r * 0.35);
    }
  });
}

function drawHazards(ctx, world, scene) {
  const values = world.hazard_values || [];
  const radii = world.hazard_radii || [];
  world.hazards.forEach((pos, i) => {
    const p = worldToCanvas(pos, scene);
    const v = values[i] ?? 1.0;
    const hazardRadius = radii[i] ?? world.hazard_radius;
    const r = hazardRadius * scene.size * Math.sqrt(v);
    ctx.fillStyle = `rgba(204, 110, 88, ${Math.min(0.85, 0.3 + v * 0.25)})`;
    ctx.strokeStyle = v >= 2.0 ? "#5b0e05" : v >= 1.4 ? "#7a1b10" : "#913a2e";
    ctx.lineWidth = 1.5 + (v >= 1.4 ? 1 : 0);
    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#7d241c";
    ctx.font = `700 ${Math.max(10, 9 + v * 2.5)}px Georgia`; ctx.textAlign = "center";
    ctx.fillText(v >= 2.0 ? "☠" : v >= 1.4 ? "‼" : "!", p.x, p.y + 5);
  });
}

function drawOrganism(ctx, org, scene, isSelected) {
  const p = worldToCanvas(org.position, scene);
  const v = org.visual;
  const radius = (BASE_ORGANISM_RADIUS * v.body_size * (0.5 + 0.5 * org.energy)) * (scene.size / 600);
  const r = Math.max(5, radius);

  // Blend damage into lightness so healthy=bright, damaged=dark — no extra draw call
  const l = Math.round(v.color_l * 100 * (1.0 - org.damage * 0.5));
  const baseColor = `hsl(${v.color_h}, ${Math.round(v.color_s * 100)}%, ${l}%)`;

  ctx.save();
  ctx.translate(p.x, p.y);

  // Shape — single fill+stroke (no pattern overlay for perf)
  ctx.fillStyle = baseColor;
  ctx.strokeStyle = isSelected ? "rgba(255,230,80,0.9)" : "rgba(0,0,0,0.4)";
  ctx.lineWidth = isSelected ? 2.5 : 1.2;
  _drawShape(ctx, v.shape, r);
  ctx.fill(); ctx.stroke();

  // Heading arrow — simple line only
  const arrowLen = r * 1.5;
  ctx.strokeStyle = "rgba(0,0,0,0.55)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(Math.cos(org.heading) * arrowLen, -Math.sin(org.heading) * arrowLen);
  ctx.stroke();

  // Health bar — thin flat bar instead of arc (cheaper)
  const barW = r * 2;
  const barH = 3;
  const barY = -r - 6;
  ctx.fillStyle = "rgba(0,0,0,0.25)";
  ctx.fillRect(-barW / 2, barY, barW, barH);
  ctx.fillStyle = `hsl(${Math.round(120 * org.energy)}, 70%, 45%)`;
  ctx.fillRect(-barW / 2, barY, barW * org.energy, barH);

  // Gen label (only for selected or gen>0)
  if (isSelected || org.generation > 0) {
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.font = `bold ${Math.max(7, r * 0.5)}px monospace`;
    ctx.textAlign = "center";
    ctx.fillText(`g${org.generation}`, 0, r + 9);
  }

  // Mate-ready pulsing ring
  if (org.mate_ready) {
    const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 400 + org.uid);
    ctx.strokeStyle = `rgba(255, 140, 200, ${0.5 + 0.5 * pulse})`;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.arc(0, 0, r + 5 + pulse * 3, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Selection ring + sensor circles
  if (isSelected) {
    ctx.strokeStyle = "rgba(255, 230, 80, 0.9)"; ctx.lineWidth = 2.5;
    ctx.setLineDash([5, 3]);
    ctx.beginPath(); ctx.arc(0, 0, org.physical.sensor_range * scene.size, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = "rgba(80, 180, 80, 0.7)"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(0, 0, org.physical.food_visible_range * scene.size, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.restore();
}

function _drawShape(ctx, shape, r) {
  if (shape === "circle") {
    ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2);
  } else if (shape === "triangle") {
    ctx.beginPath();
    ctx.moveTo(0, -r);
    ctx.lineTo(r * 0.87, r * 0.5);
    ctx.lineTo(-r * 0.87, r * 0.5);
    ctx.closePath();
  } else { // diamond
    ctx.beginPath();
    ctx.moveTo(0, -r);
    ctx.lineTo(r * 0.75, 0);
    ctx.lineTo(0, r);
    ctx.lineTo(-r * 0.75, 0);
    ctx.closePath();
  }
}

function _drawPattern(ctx, pattern, hue, r) {
  if (pattern === "solid") return;
  const contrastColor = `hsl(${(hue + 180) % 360}, 60%, 35%)`;
  ctx.strokeStyle = contrastColor; ctx.lineWidth = 1.5;
  if (pattern === "stripe") {
    const savedClip = ctx.save();
    ctx.beginPath(); ctx.arc(0, 0, r * 0.9, 0, Math.PI * 2); ctx.clip();
    for (let offset = -r; offset < r; offset += r * 0.45) {
      ctx.beginPath(); ctx.moveTo(offset, -r); ctx.lineTo(offset + r * 0.3, r); ctx.stroke();
    }
    ctx.restore();
  } else if (pattern === "spot") {
    ctx.fillStyle = contrastColor;
    ctx.beginPath(); ctx.arc(r * 0.3, -r * 0.2, r * 0.22, 0, Math.PI * 2); ctx.fill();
  } else if (pattern === "ring") {
    ctx.beginPath(); ctx.arc(0, 0, r * 0.55, 0, Math.PI * 2); ctx.stroke();
  }
}

function drawEggs(ctx, eggs, world, scene) {
  eggs.forEach(egg => {
    const p = worldToCanvas(egg.position, scene);
    const pulse = 0.8 + 0.2 * Math.sin(Date.now() / 300 + egg.uid);
    const r = 7 * pulse;

    // Egg body
    ctx.save(); ctx.translate(p.x, p.y);
    ctx.fillStyle = "rgba(250, 240, 200, 0.9)";
    ctx.strokeStyle = "rgba(160, 130, 50, 0.8)"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.ellipse(0, 0, r * 0.7, r, 0, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();

    // Parent color dots
    const h1 = _lineageHue(egg.lineage_a, world._lineageCount || 12);
    const h2 = _lineageHue(egg.lineage_b, world._lineageCount || 12);
    ctx.fillStyle = `hsl(${h1}, 70%, 55%)`;
    ctx.beginPath(); ctx.arc(-r * 0.3, 0, 2.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = `hsl(${h2}, 70%, 55%)`;
    ctx.beginPath(); ctx.arc(r * 0.3, 0, 2.5, 0, Math.PI * 2); ctx.fill();

    // Countdown
    if (egg.hatch_countdown <= 10) {
      ctx.fillStyle = "rgba(200,100,0,0.9)"; ctx.font = "bold 8px monospace"; ctx.textAlign = "center";
      ctx.fillText(egg.hatch_countdown, 0, r + 9);
    }
    ctx.restore();
  });
}

function _lineageHue(lineage, total) {
  return (lineage / Math.max(total, 1)) * 360;
}

function drawEvolutionHud(ctx, state, canvasSize, scene) {
  const { width } = canvasSize;
  const padding = 12;
  ctx.fillStyle = "rgba(18, 24, 18, 0.76)";
  ctx.beginPath();
  ctx.roundRect(scene.left, scene.top, scene.size, 30, 6);
  ctx.fill();

  ctx.fillStyle = "#d8e8c8"; ctx.font = "600 12px monospace"; ctx.textAlign = "left";
  const parts = [
    `tick ${state.tick}`,
    `alive ${state.alive_count}`,
    `eggs ${state.egg_count}`,
    `gen ${state.generation_max}`,
    `matings ${state.mating_events}`,
    `born ${state.total_born}`,
    `died ${state.total_died}`,
  ];
  let x = scene.left + 10;
  parts.forEach(part => {
    ctx.fillText(part, x, scene.top + 20);
    x += ctx.measureText(part).width + 18;
  });
}

export function drawEvolutionWorld(ctx, state, canvasSize, selectedUid) {
  const { width, height } = canvasSize;
  const scene = getSceneRect(width, height);

  ctx.clearRect(0, 0, width, height);
  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, "#eee7d7"); bg.addColorStop(1, "#d9d0bb");
  ctx.fillStyle = bg; ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "rgba(255,255,255,0.22)";
  ctx.fillRect(scene.left, scene.top, scene.size, scene.size);

  drawGrid(ctx, scene);
  drawShelter(ctx, state.world, scene);
  drawFood(ctx, state.world, scene);
  drawHazards(ctx, state.world, scene);
  drawEggs(ctx, state.eggs, state.world, scene);

  // Draw non-selected first, selected on top
  state.organisms.forEach(org => {
    if (org.uid !== selectedUid) drawOrganism(ctx, org, scene, false);
  });
  state.organisms.forEach(org => {
    if (org.uid === selectedUid) drawOrganism(ctx, org, scene, true);
  });

  drawEvolutionHud(ctx, state, canvasSize, scene);

  ctx.strokeStyle = "rgba(48, 59, 47, 0.28)"; ctx.lineWidth = 2;
  ctx.strokeRect(scene.left, scene.top, scene.size, scene.size);
}

export function getWorldCoords(canvasX, canvasY, canvasSize) {
  const scene = getSceneRect(canvasSize.width, canvasSize.height);
  return [
    (canvasX - scene.left) / scene.size,
    1.0 - (canvasY - scene.top) / scene.size,
  ];
}
