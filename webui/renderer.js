/* Canvas rendering for the organism world view. */

export function getSceneRect(width, height) {
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

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.arcTo(x + width, y, x + width, y + height, radius);
  context.arcTo(x + width, y + height, x, y + height, radius);
  context.arcTo(x, y + height, x, y, radius);
  context.arcTo(x, y, x + width, y, radius);
  context.closePath();
}

function drawVisitation(ctx, world, scene) {
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

function drawGrid(ctx, scene) {
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

function drawShelter(ctx, world, scene) {
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

function entityPos(entity) {
  return Array.isArray(entity) ? entity : entity.pos;
}

function entityValue(entity) {
  return Array.isArray(entity) ? 1.0 : (entity.value ?? 1.0);
}

function drawFood(ctx, world, scene) {
  world.food.forEach((food) => {
    const point = worldToCanvas(entityPos(food), scene);
    const value = entityValue(food);
    const baseRadius = Math.max(5, world.eat_radius * scene.size * 0.8);
    const radius = baseRadius * Math.sqrt(value);
    const saturation = Math.min(1, 0.5 + value * 0.25);
    ctx.fillStyle = value >= 2.0 ? "#3c9048" : value >= 1.4 ? "#56a756" : value >= 0.8 ? "#82b96c" : "#b8cc8c";
    ctx.strokeStyle = "#2e6b2e";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    if (value >= 2.0) {
      ctx.fillStyle = "#ffe88a";
      ctx.font = `700 ${Math.max(9, radius * 0.9)}px Georgia`;
      ctx.textAlign = "center";
      ctx.fillText("★", point.x, point.y + radius * 0.35);
    }
  });
}

function drawHazards(ctx, world, scene) {
  world.hazards.forEach((hazard) => {
    const point = worldToCanvas(entityPos(hazard), scene);
    const value = entityValue(hazard);
    const radius = world.hazard_radius * scene.size * Math.sqrt(value);
    const alpha = Math.min(0.85, 0.3 + value * 0.25);
    ctx.fillStyle = `rgba(204, 110, 88, ${alpha})`;
    ctx.strokeStyle = value >= 2.0 ? "#5b0e05" : value >= 1.4 ? "#7a1b10" : "#913a2e";
    ctx.lineWidth = 2 + (value >= 1.4 ? 1 : 0);
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#7d241c";
    ctx.font = `700 ${Math.max(11, 10 + value * 3)}px Georgia`;
    ctx.textAlign = "center";
    const mark = value >= 2.0 ? "☠" : value >= 1.4 ? "‼" : "!";
    ctx.fillText(mark, point.x, point.y + 5);
  });
}

function drawTrail(ctx, world, scene) {
  if (world.trail.length < 2) return;
  ctx.strokeStyle = "rgba(85, 127, 160, 0.84)";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  world.trail.forEach((point, index) => {
    const p = worldToCanvas(point, scene);
    if (index === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
}

function drawSensorRays(ctx, world, sensors, scene) {
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

  // Draw food visible range circle
  if (world.food_visible_range && world.food_visible_range < world.sensor_range) {
    const fvr = world.food_visible_range * scene.size;
    ctx.save();
    ctx.strokeStyle = "rgba(87, 150, 86, 0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(origin.x, origin.y, fvr, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
}

function drawAgent(ctx, world, episode, scene) {
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

function drawSceneBorder(ctx, scene) {
  ctx.strokeStyle = "rgba(48, 59, 47, 0.28)";
  ctx.lineWidth = 2;
  ctx.strokeRect(scene.left, scene.top, scene.size, scene.size);
}

function drawHud(ctx, session, width, height, scene) {
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

export function drawWorld(ctx, session, canvasSize, showSensors) {
  const { world, episode, sensors } = session;
  const { width, height } = canvasSize;
  const scene = getSceneRect(width, height);

  ctx.clearRect(0, 0, width, height);

  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, "#eee7d7");
  bg.addColorStop(1, "#d9d0bb");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = "rgba(255,255,255,0.24)";
  ctx.fillRect(scene.left, scene.top, scene.size, scene.size);

  drawVisitation(ctx, world, scene);
  drawGrid(ctx, scene);
  drawShelter(ctx, world, scene);
  drawFood(ctx, world, scene);
  drawHazards(ctx, world, scene);
  drawTrail(ctx, world, scene);
  if (showSensors) {
    drawSensorRays(ctx, world, sensors, scene);
  }
  drawAgent(ctx, world, episode, scene);
  drawSceneBorder(ctx, scene);
  drawHud(ctx, session, width, height, scene);
}
