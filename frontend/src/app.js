const SETTINGS_KEY = "biomechanics-motion-ui";
const RAW_API_BASE_URL =
  typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_BASE_URL
    ? import.meta.env.VITE_API_BASE_URL
    : "";

function normalizeBaseUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

const API_BASE_URL = normalizeBaseUrl(RAW_API_BASE_URL);

function backendOrigin() {
  return API_BASE_URL ? new URL(API_BASE_URL).origin : window.location.origin;
}

function apiUrl(path) {
  if (/^https?:\/\//i.test(path)) return path;
  if (!API_BASE_URL) return path;
  return new URL(path, `${API_BASE_URL}/`).toString();
}

function backendAssetUrl(path) {
  if (!path) return path;
  if (/^(https?:)?\/\//i.test(path) || path.startsWith("data:")) return path;
  return new URL(path, `${backendOrigin()}/`).toString();
}

function withBackendUrls(value) {
  if (Array.isArray(value)) return value.map(withBackendUrls);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, withBackendUrls(item)]));
  }
  if (typeof value === "string" && value.startsWith("/files/")) {
    return backendAssetUrl(value);
  }
  return value;
}

const DEFAULT_CONFIG = {
  shoulderView: "front",
  posePlane: "frontal",
  modelSize: "medium",
  skipFrames: 2,
};

const METRIC_LABELS = {
  shoulder_tilt_deg: "Shoulder Tilt",
  trunk_lean_angle: "Trunk Lean",
  kyphosis_angle: "Kyphosis",
  lordosis_angle: "Lordosis",
  shoulder_tilt_mean: "Shoulder Tilt",
  kyphosis_mean: "Kyphosis",
  lordosis_mean: "Lordosis",
  trunk_lean_mean: "Trunk Lean",
};

const state = {
  uploadFile: null,
  researchFile: null,
  researchJobId: "",
  researchPoller: null,
  researchRows: [],
  ui: {
    theme: "midnight",
    accent: "cyan",
  },
  live: {
    stream: null,
    socket: null,
    socketOpen: false,
    connectionState: "idle",
    rafId: 0,
    inFlight: false,
    lastSentStamp: 0,
    lastSentAt: 0,
    lastPayloadAt: 0,
    points: [],
    chartMetric: "shoulder_tilt_deg",
    metrics: {},
    routing: null,
    performance: {
      latencyMs: null,
      rateHz: null,
    },
  },
};

function cloneConfig() {
  return JSON.parse(JSON.stringify(DEFAULT_CONFIG));
}

const uploadConfig = cloneConfig();
const liveConfig = cloneConfig();
const researchConfig = cloneConfig();

function $(id) {
  return document.getElementById(id);
}

function cssVar(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

function rgbaVar(name, alpha) {
  return `rgba(${cssVar(name)}, ${alpha})`;
}

function setActiveTab(tabId) {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.hidden = panel.id !== `${tabId}-panel`;
  });
}

function humanizePlane(value) {
  if (!value) return "Pending";
  return String(value)
    .replace(/_/g, " ")
    .replace(/^\w/, (char) => char.toUpperCase());
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${Math.round(Number(value) * 100)}%`;
}

function setHtml(id, html) {
  $(id).innerHTML = html;
}

function setShellFrameState(id, hasFrame) {
  const node = $(id);
  if (!node) return;
  node.classList.toggle("has-frame", Boolean(hasFrame));
}

function readConfig(prefix, target) {
  const shoulderView = $(`${prefix}-shoulder-view`);
  const posePlane = $(`${prefix}-pose-plane`);
  const skipFrames = $(`${prefix}-skip-frames`);
  target.shoulderView = shoulderView ? shoulderView.value : DEFAULT_CONFIG.shoulderView;
  target.posePlane = posePlane ? posePlane.value : DEFAULT_CONFIG.posePlane;
  target.modelSize = DEFAULT_CONFIG.modelSize;
  target.skipFrames = Number(skipFrames?.value ?? DEFAULT_CONFIG.skipFrames);
}

function clearErrors(prefix) {
  document.querySelectorAll(`[data-error-prefix="${prefix}"]`).forEach((node) => {
    node.textContent = "";
  });
  const server = $(`${prefix}-server-error`);
  if (server) server.textContent = "";
}

function showErrors(prefix, errors) {
  Object.entries(errors).forEach(([key, message]) => {
    const node = document.querySelector(`[data-error-prefix="${prefix}"][data-error-key="${key}"]`);
    if (node) node.textContent = message;
  });
}

function validateConfig(config, needsFile, fileLabel) {
  const errors = {};
  if (needsFile) errors.file = `Please add a ${fileLabel}.`;
  if (!config.shoulderView) errors.shoulderView = "Shoulder camera view is required.";
  if (!config.posePlane) errors.posePlane = "Pose plane is required.";
  if (!Number.isInteger(config.skipFrames) || config.skipFrames < 1 || config.skipFrames > 30) {
    errors.skipFrames = "Use a whole number from 1 to 30.";
  }
  return errors;
}

function formDataFromConfig(fileKey, file, config) {
  const data = new FormData();
  data.append(fileKey, file);
  data.append("shoulder_view", config.shoulderView);
  data.append("pose_plane", config.posePlane);
  data.append("model_size", DEFAULT_CONFIG.modelSize);
  data.append("skip_frames", String(config.skipFrames));
  return data;
}

function renderMetricCards(containerId, cards) {
  setHtml(
    containerId,
    cards
      .map(
        (card) => `
          <div class="metric-card">
            <span>${card.label}</span>
            <strong>${formatNumber(card.value)}${card.unit ? ` ${card.unit}` : ""}</strong>
          </div>
        `,
      )
      .join(""),
  );
}

function renderWarnings(containerId, warnings) {
  if (!warnings || !warnings.length) {
    setHtml(containerId, "");
    return;
  }
  setHtml(
    containerId,
    warnings
      .map(
        (warning) => `
          <div class="warning-item">
            <span class="warning-mark">!</span>
            <span>${warning}</span>
          </div>
        `,
      )
      .join(""),
  );
}

function routeHeading(route) {
  if (!route) return "Analysis route pending";
  if (route.fallback_active) return "Oblique detected, sagittal fallback active";
  return `${humanizePlane(route.effective_pose_plane)} workflow active`;
}

function routePills(route, detectedLabel) {
  const pills = [
    `Detected: <strong>${humanizePlane(detectedLabel || route?.detected_pose_plane || "not detected")}</strong>`,
    `Models: <strong>${(route?.active_models || []).join(", ") || "none"}</strong>`,
  ];
  if (route?.fallback_active) {
    pills.push(`Fallback: <strong>${humanizePlane(route.fallback_plane)}</strong>`);
  }
  return pills
    .map((pill) => `<span class="route-pill">${pill}</span>`)
    .join("");
}

function renderUploadResult(result) {
  result = withBackendUrls(result);
  $("upload-result").hidden = false;
  const route = result.routing || {};
  const shoulderSurface = result.shoulder.active
    ? `<video src="${result.shoulder.assets.annotated_video}" controls preload="metadata"></video>`
    : `<div class="media-surface">
         <div class="media-state">
           <strong>Shoulder workflow inactive for this capture</strong>
           <p>${result.shoulder.guidance}</p>
         </div>
       </div>`;
  const spinalSurface = result.spinal.active
    ? (result.spinal.assets.annotated_video
        ? `<video src="${result.spinal.assets.annotated_video}" controls preload="metadata"></video>`
        : `<div class="media-surface">
             <div class="media-state">
               <strong>Video export unavailable</strong>
               <p>The spinal metrics and plots were created, but no annotated spinal video was returned for this run.</p>
             </div>
           </div>`)
    : `<div class="media-surface">
         <div class="media-state">
           <strong>Spinal workflow inactive for this capture</strong>
           <p>${result.spinal.guidance}</p>
         </div>
       </div>`;

  setHtml(
    "upload-routing",
    `
      <div class="route-copy">
        <span class="eyebrow">Analysis Route</span>
        <h3>${routeHeading(route)}</h3>
        <p>${route.guidance || ""}</p>
        <div class="route-pill-row">
          ${routePills(route, result.detected_pose_plane?.label)}
        </div>
      </div>
    `,
  );

  renderMetricCards("upload-metrics", [
    { label: "Shoulder Tilt Mean", value: result.shoulder.summary?.shoulder_tilt_mean, unit: "deg" },
    { label: "Trunk Lean Mean", value: result.spinal.summary?.trunk_lean?.mean, unit: "deg" },
    { label: "Kyphosis Mean", value: result.spinal.summary?.kyphosis?.mean, unit: "deg" },
    { label: "Valid Frames", value: result.spinal.summary?.valid_frames, unit: "" },
  ]);

  renderWarnings("upload-warnings", result.warnings || []);

  setHtml(
    "upload-media",
    `
      <div class="media-panel">
        <h3>Shoulder Outputs</h3>
        ${shoulderSurface}
        ${
          result.shoulder.active
            ? `<div class="asset-row">
                <a class="asset-link" href="${result.shoulder.assets.annotated_video}" target="_blank" rel="noreferrer">Open annotated video</a>
                <a class="asset-link" href="${result.shoulder.assets.csv}" target="_blank" rel="noreferrer">Open metrics CSV</a>
                <a class="asset-link" href="${result.shoulder.assets.angle_plot}" target="_blank" rel="noreferrer">Angle plot</a>
                <a class="asset-link" href="${result.shoulder.assets.symmetry_bar_chart}" target="_blank" rel="noreferrer">Symmetry chart</a>
              </div>`
            : ""
        }
      </div>
      <div class="media-panel">
        <h3>Spinal Outputs</h3>
        ${spinalSurface}
        ${
          result.spinal.active
            ? `<div class="asset-row">
                ${result.spinal.assets.annotated_video ? `<a class="asset-link" href="${result.spinal.assets.annotated_video}" target="_blank" rel="noreferrer">Open annotated video</a>` : ""}
                <a class="asset-link" href="${result.spinal.assets.time_series}" target="_blank" rel="noreferrer">Time series</a>
                <a class="asset-link" href="${result.spinal.assets.standard_comparison}" target="_blank" rel="noreferrer">Standard comparison</a>
              </div>`
            : ""
        }
      </div>
    `,
  );

  $("upload-shoulder-summary").textContent = result.shoulder.summary_text || result.shoulder.guidance || "No text summary generated.";
  $("upload-spinal-summary").textContent = result.spinal.active ? JSON.stringify(result.spinal.clinical_report, null, 2) : result.spinal.guidance;
}

async function handleUploadSubmit() {
  readConfig("upload", uploadConfig);
  clearErrors("upload");
  const errors = validateConfig(uploadConfig, !state.uploadFile, "video file");
  if (Object.keys(errors).length) {
    showErrors("upload", errors);
    return;
  }

  const button = $("upload-submit");
  button.disabled = true;
  button.querySelector(".button-label").textContent = "Running analysis...";
  try {
    const response = await fetch(apiUrl("/api/analyze/video"), {
      method: "POST",
      body: formDataFromConfig("video", state.uploadFile, uploadConfig),
    });
    const payload = await response.json();
    if (!response.ok) {
      showErrors("upload", payload.errors || {});
      $("upload-server-error").textContent = payload.detail || "Analysis failed.";
      return;
    }
    renderUploadResult(payload);
  } catch (error) {
    $("upload-server-error").textContent = error.message || "Analysis failed.";
  } finally {
    button.disabled = false;
    button.querySelector(".button-label").textContent = "Analyze video";
  }
}

function thresholdText(stateItem) {
  if (!stateItem?.threshold) return "Awaiting threshold";
  if (stateItem.state === "inactive") return "Inactive for this route";
  if (stateItem.threshold.type === "upper_abs") return `Normal <= ${stateItem.threshold.upper}`;
  return `Normal ${stateItem.threshold.lower} to ${stateItem.threshold.upper}`;
}

function renderLiveStatus(states) {
  const items = [
    ["Shoulder Tilt", states?.shoulder_tilt],
    ["Trunk Lean", states?.trunk_lean],
    ["Kyphosis", states?.kyphosis],
    ["Lordosis", states?.lordosis],
  ];
  setHtml(
    "live-status-grid",
    items
      .map(([label, stateItem]) => {
        const tone = stateItem?.state === "alert" ? "bad" : stateItem?.state === "normal" ? "good" : "";
        const stateText =
          stateItem?.state === "alert"
            ? "Above threshold"
            : stateItem?.state === "normal"
              ? "Within threshold"
              : stateItem?.state === "inactive"
                ? "Standby"
                : "No reading";
        return `
          <div class="status-card ${tone}">
            <span>${label}</span>
            <strong>${stateItem?.value === null || stateItem?.value === undefined ? "--" : formatNumber(stateItem.value)}</strong>
            <small>${stateText}</small>
            <em>${thresholdText(stateItem)}</em>
          </div>
        `;
      })
      .join(""),
  );
}

function renderLiveRouting(routing) {
  if (!routing) return;
  $("live-route-selected").textContent = humanizePlane(routing.selected_pose_plane);
  $("live-route-detected").textContent = humanizePlane(routing.detected_pose_plane || "pending");
  $("live-route-models").textContent = (routing.active_models || []).join(", ") || "none";
  $("live-route-active").textContent = routeHeading(routing);
  $("live-route-guidance").textContent = routing.guidance || "";
}

function renderLiveInsights(routing) {
  if (!routing) {
    setHtml(
      "live-insight-cards",
      `
        <div class="insight-card"><span>Detection confidence</span><strong>--</strong></div>
        <div class="insight-card"><span>Dominant plane</span><strong>Pending</strong></div>
        <div class="insight-card"><span>Route source</span><strong>Waiting</strong></div>
      `,
    );
    return;
  }
  const routeSource = routing.fallback_active ? "Oblique fallback" : routing.evaluation_source || "selected";
  setHtml(
    "live-insight-cards",
    `
      <div class="insight-card">
        <span>Detection confidence</span>
        <strong>${formatPercent(routing.detection_confidence)}</strong>
      </div>
      <div class="insight-card">
        <span>Dominant plane</span>
        <strong>${humanizePlane(routing.dominant_pose_plane || routing.effective_pose_plane)}</strong>
      </div>
      <div class="insight-card">
        <span>Route source</span>
        <strong>${humanizePlane(routeSource)}</strong>
      </div>
    `,
  );
}

function updateLiveSignalCards() {
  const transportLabel =
    state.live.connectionState === "open"
      ? "WebSocket"
      : state.live.connectionState === "connecting"
        ? "Connecting"
        : state.live.connectionState === "closed"
          ? "Disconnected"
          : "Idle";
  $("live-transport").textContent = transportLabel;
  $("live-latency").textContent =
    state.live.performance.latencyMs === null ? "--" : `${Math.round(state.live.performance.latencyMs)} ms`;
  $("live-rate").textContent =
    state.live.performance.rateHz === null ? "--" : `${state.live.performance.rateHz.toFixed(1)} fps`;
}

function setActiveMetricButton(metric, selector) {
  document.querySelectorAll(selector).forEach((button) => {
    button.classList.toggle("active", button.dataset.metric === metric);
  });
}

function syncLiveChartLabel() {
  $("live-chart-label").textContent = `${METRIC_LABELS[state.live.chartMetric] || "Live Metric"} Trend`;
  $("live-chart-meta").textContent = `Last ${Math.max(state.live.points.length, 40)} readings`;
}

function ensureLiveMetricForRoute(routing) {
  if (!routing?.active_metric_keys?.length) return;
  if (!routing.active_metric_keys.includes(state.live.chartMetric)) {
    state.live.chartMetric = routing.default_chart_metric || routing.active_metric_keys[0];
    setActiveMetricButton(state.live.chartMetric, ".live-metric-toggle");
  }
  syncLiveChartLabel();
}

function setupCanvas(canvas) {
  const context = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { context, width, height };
}

function drawLineChart(canvasId, points, metric) {
  const canvas = $(canvasId);
  const { context: ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  ctx.strokeStyle = rgbaVar("--accent-rgb", 0.12);
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = 22 + ((height - 44) / 4) * i;
    ctx.beginPath();
    ctx.moveTo(44, y);
    ctx.lineTo(width - 20, y);
    ctx.stroke();
  }

  const values = points.map((point) => Number(point[metric])).filter((value) => Number.isFinite(value));
  if (!values.length) {
    ctx.fillStyle = cssVar("--muted");
    ctx.font = "14px Space Grotesk";
    ctx.fillText("Live chart will appear here as the stream starts.", 44, height / 2);
    return;
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const left = 44;
  const top = 22;
  const chartWidth = width - 66;
  const chartHeight = height - 44;

  const gradient = ctx.createLinearGradient(0, top, 0, top + chartHeight);
  gradient.addColorStop(0, rgbaVar("--accent-rgb", 0.38));
  gradient.addColorStop(1, rgbaVar("--accent-rgb", 0.02));

  ctx.beginPath();
  values.forEach((value, index) => {
    const x = left + (chartWidth * index) / Math.max(values.length - 1, 1);
    const y = top + chartHeight - ((value - min) / range) * chartHeight;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(left + chartWidth, top + chartHeight);
  ctx.lineTo(left, top + chartHeight);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.strokeStyle = cssVar("--accent");
  ctx.lineWidth = 2.8;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = left + (chartWidth * index) / Math.max(values.length - 1, 1);
    const y = top + chartHeight - ((value - min) / range) * chartHeight;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  const lastValue = values[values.length - 1];
  const lastX = left + chartWidth;
  const lastY = top + chartHeight - ((lastValue - min) / range) * chartHeight;
  ctx.fillStyle = cssVar("--accent");
  ctx.beginPath();
  ctx.arc(lastX, lastY, 4.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = cssVar("--muted");
  ctx.font = "12px Space Grotesk";
  ctx.fillText(max.toFixed(1), 10, top + 8);
  ctx.fillText(min.toFixed(1), 10, top + chartHeight);
}

function drawBarChart(canvasId, rows, metric) {
  const canvas = $(canvasId);
  const { context: ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const valid = rows.filter((row) => Number.isFinite(Number(row[metric])));
  if (!valid.length) {
    ctx.fillStyle = cssVar("--muted");
    ctx.font = "14px Space Grotesk";
    ctx.fillText("Batch chart will appear here.", 44, height / 2);
    return;
  }

  const max = Math.max(...valid.map((row) => Number(row[metric]))) || 1;
  const left = 48;
  const bottom = height - 42;
  const chartHeight = height - 70;
  const barWidth = Math.max(18, (width - left - 28) / valid.length - 12);

  ctx.strokeStyle = rgbaVar("--accent-rgb", 0.12);
  for (let i = 0; i < 4; i += 1) {
    const y = 18 + (chartHeight / 3) * i;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(width - 18, y);
    ctx.stroke();
  }

  valid.forEach((row, index) => {
    const value = Number(row[metric]);
    const x = left + index * (barWidth + 12);
    const barHeight = (value / max) * chartHeight;
    const y = bottom - barHeight;
    ctx.fillStyle = cssVar("--accent");
    ctx.fillRect(x, y, barWidth, barHeight);
    ctx.fillStyle = cssVar("--muted");
    ctx.font = "10px Space Grotesk";
    ctx.save();
    ctx.translate(x + 4, bottom + 10);
    ctx.rotate(-0.4);
    ctx.fillText(`${row.person_id}-${row.state}`, 0, 0);
    ctx.restore();
  });
}

function drawAreaChart(canvasId, rows, metric) {
  const canvas = $(canvasId);
  const { context: ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const valid = rows.filter((row) => Number.isFinite(Number(row[metric])));
  if (!valid.length) {
    ctx.fillStyle = cssVar("--muted");
    ctx.font = "14px Space Grotesk";
    ctx.fillText("State comparison area will appear here.", 44, height / 2);
    return;
  }

  const values = valid.map((row) => Number(row[metric]));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const left = 40;
  const top = 22;
  const chartWidth = width - 58;
  const chartHeight = height - 60;

  const gradient = ctx.createLinearGradient(0, top, 0, top + chartHeight);
  gradient.addColorStop(0, rgbaVar("--accent-rgb", 0.34));
  gradient.addColorStop(1, rgbaVar("--accent-rgb", 0.03));

  ctx.beginPath();
  valid.forEach((row, index) => {
    const value = Number(row[metric]);
    const x = left + (chartWidth * index) / Math.max(valid.length - 1, 1);
    const y = top + chartHeight - ((value - min) / range) * chartHeight;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(left + chartWidth, top + chartHeight);
  ctx.lineTo(left, top + chartHeight);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.strokeStyle = cssVar("--accent");
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  valid.forEach((row, index) => {
    const value = Number(row[metric]);
    const x = left + (chartWidth * index) / Math.max(valid.length - 1, 1);
    const y = top + chartHeight - ((value - min) / range) * chartHeight;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawLimb(ctx, a, b, color, width = 3) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function drawJoint(ctx, point, radius, fill) {
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function skeletonPose(width, height, options = {}) {
  const scale = options.scale || Math.min(width, height) * 0.18;
  const centerX = options.centerX || width * 0.5;
  const centerY = options.centerY || height * 0.66;
  const lean = ((options.trunkLean || 0) * Math.PI) / 180;
  const shoulderTilt = ((options.shoulderTilt || 0) * Math.PI) / 180;
  const phase = options.phase || 0;
  const swing = Math.sin(phase) * scale * 0.42;
  const kneeLift = Math.cos(phase) * scale * 0.16;

  const hip = { x: centerX, y: centerY };
  const neck = {
    x: centerX + Math.sin(lean) * scale * 0.55,
    y: centerY - scale * 1.14,
  };
  const head = {
    x: neck.x + Math.sin(lean) * scale * 0.1,
    y: neck.y - scale * 0.34,
  };
  const shoulderSpan = scale * 0.42;
  const hipSpan = scale * 0.28;

  const leftShoulder = {
    x: neck.x - shoulderSpan,
    y: neck.y + Math.sin(shoulderTilt) * scale * 0.12,
  };
  const rightShoulder = {
    x: neck.x + shoulderSpan,
    y: neck.y - Math.sin(shoulderTilt) * scale * 0.12,
  };

  const leftHip = { x: hip.x - hipSpan, y: hip.y };
  const rightHip = { x: hip.x + hipSpan, y: hip.y };

  const leftElbow = { x: leftShoulder.x - swing * 0.54, y: leftShoulder.y + scale * 0.48 };
  const rightElbow = { x: rightShoulder.x + swing * 0.54, y: rightShoulder.y + scale * 0.48 };
  const leftHand = { x: leftElbow.x - swing * 0.36, y: leftElbow.y + scale * 0.42 };
  const rightHand = { x: rightElbow.x + swing * 0.36, y: rightElbow.y + scale * 0.42 };

  const leftKnee = { x: leftHip.x + swing * 0.4, y: leftHip.y + scale * 0.62 - kneeLift };
  const rightKnee = { x: rightHip.x - swing * 0.4, y: rightHip.y + scale * 0.62 + kneeLift };
  const leftAnkle = { x: leftKnee.x + swing * 0.16, y: leftKnee.y + scale * 0.62 };
  const rightAnkle = { x: rightKnee.x - swing * 0.16, y: rightKnee.y + scale * 0.62 };
  const leftFoot = { x: leftAnkle.x + scale * 0.24, y: leftAnkle.y + scale * 0.06 };
  const rightFoot = { x: rightAnkle.x + scale * 0.24, y: rightAnkle.y + scale * 0.06 };

  return {
    head,
    neck,
    hip,
    leftShoulder,
    rightShoulder,
    leftHip,
    rightHip,
    leftElbow,
    rightElbow,
    leftHand,
    rightHand,
    leftKnee,
    rightKnee,
    leftAnkle,
    rightAnkle,
    leftFoot,
    rightFoot,
  };
}

function drawSkeletonFigure(ctx, width, height, options = {}) {
  const alpha = options.alpha ?? 1;
  const accent = rgbaVar("--accent-rgb", Math.max(0.3, alpha));
  const accentAlt = rgbaVar("--accent-alt-rgb", Math.max(0.24, alpha * 0.9));
  const pose = skeletonPose(width, height, options);

  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  drawLimb(ctx, pose.neck, pose.hip, accent, 4);
  drawLimb(ctx, pose.leftShoulder, pose.rightShoulder, accentAlt, 3);
  drawLimb(ctx, pose.leftHip, pose.rightHip, accentAlt, 3);
  drawLimb(ctx, pose.leftShoulder, pose.leftElbow, accent, 3);
  drawLimb(ctx, pose.leftElbow, pose.leftHand, accent, 3);
  drawLimb(ctx, pose.rightShoulder, pose.rightElbow, accent, 3);
  drawLimb(ctx, pose.rightElbow, pose.rightHand, accent, 3);
  drawLimb(ctx, pose.leftHip, pose.leftKnee, accent, 3);
  drawLimb(ctx, pose.leftKnee, pose.leftAnkle, accent, 3);
  drawLimb(ctx, pose.leftAnkle, pose.leftFoot, accentAlt, 3);
  drawLimb(ctx, pose.rightHip, pose.rightKnee, accent, 3);
  drawLimb(ctx, pose.rightKnee, pose.rightAnkle, accent, 3);
  drawLimb(ctx, pose.rightAnkle, pose.rightFoot, accentAlt, 3);

  [pose.head, pose.neck, pose.leftShoulder, pose.rightShoulder, pose.leftElbow, pose.rightElbow, pose.leftHip, pose.rightHip, pose.leftKnee, pose.rightKnee, pose.leftAnkle, pose.rightAnkle]
    .forEach((joint) => drawJoint(ctx, joint, joint === pose.head ? 7 : 4.5, accentAlt));
  ctx.restore();
}

function sceneProfile() {
  switch (state.ui.theme) {
    case "aurora":
      return { horizon: 0.75, figures: 6, spacing: 0.145, drift: 38, scale: 88 };
    case "xray":
      return { horizon: 0.78, figures: 5, spacing: 0.16, drift: 24, scale: 82 };
    case "ember":
      return { horizon: 0.8, figures: 4, spacing: 0.19, drift: 26, scale: 92 };
    case "ivory":
      return { horizon: 0.77, figures: 5, spacing: 0.16, drift: 18, scale: 78 };
    default:
      return { horizon: 0.77, figures: 5, spacing: 0.17, drift: 30, scale: 84 };
  }
}

function initBackgroundScene() {
  const particleCanvas = $("background-particles");
  const motionCanvas = $("background-motion");
  if (!particleCanvas || !motionCanvas) return;

  const particles = Array.from({ length: 90 }, () => ({
    x: Math.random(),
    y: Math.random(),
    size: Math.random() * 2.5 + 0.6,
    drift: Math.random() * 0.22 + 0.12,
    speed: Math.random() * 0.0008 + 0.0003,
  }));

  const resize = () => {
    [particleCanvas, motionCanvas].forEach((canvas) => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    });
  };

  const draw = (time) => {
    const particleCtx = particleCanvas.getContext("2d");
    const motionCtx = motionCanvas.getContext("2d");
    const profile = sceneProfile();
    particleCtx.clearRect(0, 0, particleCanvas.width, particleCanvas.height);
    motionCtx.clearRect(0, 0, motionCanvas.width, motionCanvas.height);

    particles.forEach((particle, index) => {
      particle.y += particle.speed;
      particle.x += Math.sin(time * 0.0002 + index) * 0.00006;
      if (particle.y > 1.05) particle.y = -0.05;
      if (particle.x > 1.02) particle.x = -0.02;
      if (particle.x < -0.02) particle.x = 1.02;

      const x = particle.x * particleCanvas.width;
      const y = particle.y * particleCanvas.height;
      const glow = particleCtx.createRadialGradient(x, y, 0, x, y, particle.size * 14);
      glow.addColorStop(0, rgbaVar("--accent-rgb", 0.9));
      glow.addColorStop(1, rgbaVar("--accent-rgb", 0));
      particleCtx.fillStyle = glow;
      particleCtx.beginPath();
      particleCtx.arc(x, y, particle.size * 12, 0, Math.PI * 2);
      particleCtx.fill();

      if (index < particles.length - 1) {
        const next = particles[index + 1];
        const dx = next.x - particle.x;
        const dy = next.y - particle.y;
        if (Math.abs(dx) < 0.08 && Math.abs(dy) < 0.08) {
          particleCtx.strokeStyle = rgbaVar("--accent-rgb", 0.12);
          particleCtx.lineWidth = 1;
          particleCtx.beginPath();
          particleCtx.moveTo(x, y);
          particleCtx.lineTo(next.x * particleCanvas.width, next.y * particleCanvas.height);
          particleCtx.stroke();
        }
      }
    });

    const horizon = motionCanvas.height * profile.horizon;
    motionCtx.strokeStyle = rgbaVar("--accent-rgb", 0.2);
    motionCtx.lineWidth = 1.2;
    motionCtx.beginPath();
    motionCtx.moveTo(0, horizon);
    motionCtx.lineTo(motionCanvas.width, horizon);
    motionCtx.stroke();

    for (let index = 0; index < profile.figures; index += 1) {
      const phase = time * 0.003 + index * 0.9;
      const x = motionCanvas.width * (0.12 + index * profile.spacing) + Math.sin(phase * 0.4) * profile.drift;
      const y = horizon - index * 18;
      const scale = profile.scale + index * 10;
      drawSkeletonFigure(motionCtx, motionCanvas.width, motionCanvas.height, {
        centerX: x,
        centerY: y,
        scale,
        trunkLean: Math.sin(phase * 0.7) * 10,
        shoulderTilt: Math.cos(phase * 0.8) * 8,
        phase,
        alpha: 0.18 + index * 0.08,
      });
    }

    window.requestAnimationFrame(draw);
  };

  resize();
  window.addEventListener("resize", resize);
  window.requestAnimationFrame(draw);
}

function initLiveWireframe() {
  const canvas = $("live-wireframe");
  if (!canvas) return;

  const draw = (time) => {
    const { context: ctx, width, height } = setupCanvas(canvas);
    ctx.clearRect(0, 0, width, height);

    const floorY = height * 0.82;
    ctx.strokeStyle = rgbaVar("--accent-rgb", 0.18);
    ctx.lineWidth = 1;
    for (let index = 0; index < 7; index += 1) {
      const y = floorY - index * 34;
      ctx.beginPath();
      ctx.moveTo(width * 0.08, y);
      ctx.lineTo(width * 0.92, y);
      ctx.stroke();
    }

    const metrics = state.live.metrics || {};
    const lean = Number(metrics.trunk_lean_angle) || 0;
    const shoulder = Number(metrics.shoulder_tilt_deg) || 0;
    const kyphosis = Number(metrics.kyphosis_angle) || 0;
    const phase = time * 0.004 + kyphosis * 0.015;

    drawSkeletonFigure(ctx, width, height, {
      centerX: width * 0.5,
      centerY: height * 0.68,
      scale: Math.min(width, height) * 0.18,
      trunkLean: Math.max(-18, Math.min(18, lean)),
      shoulderTilt: Math.max(-16, Math.min(16, shoulder * 1.8)),
      phase,
      alpha: 0.95,
    });

    ctx.fillStyle = cssVar("--muted");
    ctx.font = "13px Space Grotesk";
    ctx.fillText(`Lean ${formatNumber(lean)} deg`, 20, 96);
    ctx.fillText(`Shoulder ${formatNumber(shoulder)} deg`, 20, 118);
    ctx.fillText(`Kyphosis ${formatNumber(kyphosis)} deg`, 20, 140);

    window.requestAnimationFrame(draw);
  };

  window.requestAnimationFrame(draw);
}

function optimisticRoutingFromConfig() {
  const dominant = liveConfig.posePlane;
  const effective = dominant === "oblique" ? "sagittal" : dominant;
  return {
    selected_pose_plane: liveConfig.posePlane,
    detected_pose_plane: state.live.stream ? "detecting" : "pending",
    dominant_pose_plane: dominant,
    effective_pose_plane: effective,
    fallback_plane: dominant === "oblique" ? "sagittal" : null,
    fallback_active: dominant === "oblique",
    active_models: effective === "frontal" ? ["shoulder"] : ["spinal"],
    active_metric_keys:
      effective === "frontal"
        ? ["shoulder_tilt_deg", "clavicle_tilt_deg", "shoulder_imbalance", "trunk_tilt_deg", "lateral_shift_pct"]
        : ["trunk_lean_angle", "kyphosis_angle", "lordosis_angle", "keypoint_confidence"],
    default_chart_metric: effective === "frontal" ? "shoulder_tilt_deg" : "trunk_lean_angle",
    evaluation_source: "selected",
    guidance:
      dominant === "oblique"
        ? "Oblique posture will stay flagged while the sagittal spinal model is used as the fallback analysis."
        : "Live analysis uses automatic routing, built-in clinical thresholds, and the default model profile without manual model selection.",
  };
}

function pushLiveConfig() {
  readConfig("live", liveConfig);
  const optimistic = optimisticRoutingFromConfig();
  renderLiveRouting(optimistic);
  renderLiveInsights(optimistic);
  ensureLiveMetricForRoute(optimistic);
  drawLineChart("live-chart", state.live.points, state.live.chartMetric);
  if (state.live.socketOpen && state.live.socket) {
    state.live.socket.send(
      JSON.stringify({
        type: "config",
        payload: {
          shoulder_view: liveConfig.shoulderView,
          pose_plane: liveConfig.posePlane,
          model_size: DEFAULT_CONFIG.modelSize,
          skip_frames: Number(liveConfig.skipFrames),
        },
      }),
    );
  }
}

function handleLiveAnalysisPayload(payload) {
  const now = performance.now();
  state.live.performance.latencyMs = state.live.lastSentAt ? now - state.live.lastSentAt : null;
  state.live.performance.rateHz = state.live.lastPayloadAt ? 1000 / Math.max(now - state.live.lastPayloadAt, 1) : null;
  state.live.lastPayloadAt = now;
  state.live.inFlight = false;
  state.live.metrics = payload.metrics || {};
  state.live.routing = payload.routing || optimisticRoutingFromConfig();
  $("live-annotated").src = payload.annotated_frame || "";
  setShellFrameState("live-overlay-shell", Boolean(payload.annotated_frame));
  renderLiveStatus(payload.threshold_states);
  renderLiveRouting(state.live.routing);
  renderLiveInsights(state.live.routing);
  ensureLiveMetricForRoute(state.live.routing);
  updateLiveSignalCards();

  state.live.points.push({
    shoulder_tilt_deg: payload.metrics?.shoulder_tilt_deg,
    trunk_lean_angle: payload.metrics?.trunk_lean_angle,
    kyphosis_angle: payload.metrics?.kyphosis_angle,
    lordosis_angle: payload.metrics?.lordosis_angle,
  });
  state.live.points = state.live.points.slice(-40);
  drawLineChart("live-chart", state.live.points, state.live.chartMetric);
}

function firstErrorMessage(payload) {
  const errors = payload?.payload?.errors || payload?.errors || {};
  const first = Object.values(errors)[0];
  return typeof first === "string" ? first : payload?.detail || "Live analysis error.";
}

function buildLiveSocketUrl() {
  const base = new URL(API_BASE_URL || window.location.origin);
  const protocol = base.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${base.host}/ws/live-analysis`;
}

function closeLiveSocket() {
  if (state.live.socket) {
    state.live.socket.close();
    state.live.socket = null;
  }
  state.live.socketOpen = false;
  state.live.connectionState = "idle";
  updateLiveSignalCards();
}

function connectLiveSocket() {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(buildLiveSocketUrl());
    let settled = false;

    socket.addEventListener("open", () => {
      state.live.socket = socket;
      state.live.socketOpen = true;
      state.live.connectionState = "open";
      updateLiveSignalCards();
      settled = true;
      resolve(socket);
    });

    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "config-ack") return;
      if (message.type === "analysis") {
        $("live-server-error").textContent = "";
        handleLiveAnalysisPayload(message.payload);
        return;
      }
      if (message.type === "error") {
        state.live.inFlight = false;
        $("live-server-error").textContent = firstErrorMessage(message);
      }
    });

    socket.addEventListener("close", () => {
      state.live.socketOpen = false;
      state.live.inFlight = false;
      state.live.connectionState = state.live.stream ? "closed" : "idle";
      updateLiveSignalCards();
    });

    socket.addEventListener("error", () => {
      state.live.connectionState = "closed";
      if (!settled) reject(new Error("Could not open the live analysis stream."));
    });
  });
}

function submitLiveFrame(video, stamp) {
  if (state.live.inFlight || !state.live.socketOpen || !state.live.socket || video.readyState < 2) return;

  const captureCanvas = $("live-capture-canvas");
  const context = captureCanvas.getContext("2d");
  const targetWidth = Math.min(640, video.videoWidth || 640);
  const targetHeight = Math.max(1, Math.round(((video.videoHeight || 360) / (video.videoWidth || 640)) * targetWidth));
  captureCanvas.width = targetWidth;
  captureCanvas.height = targetHeight;
  context.drawImage(video, 0, 0, targetWidth, targetHeight);

  state.live.inFlight = true;
  state.live.lastSentStamp = stamp;
  state.live.lastSentAt = performance.now();
  state.live.socket.send(
    JSON.stringify({
      type: "frame",
      payload: {
        image: captureCanvas.toDataURL("image/jpeg", 0.68),
      },
    }),
  );
}

function startLiveLoop(video) {
  const step = (stamp) => {
    if (!state.live.stream) return;
    const intervalMs = Math.max(120, Number(liveConfig.skipFrames || 2) * 90);
    if (state.live.socketOpen && !state.live.inFlight && stamp - state.live.lastSentStamp >= intervalMs) {
      submitLiveFrame(video, stamp);
    }
    state.live.rafId = window.requestAnimationFrame(step);
  };

  if (state.live.rafId) window.cancelAnimationFrame(state.live.rafId);
  state.live.rafId = window.requestAnimationFrame(step);
}

function resetLiveSurface() {
  state.live.points = [];
  state.live.metrics = {};
  state.live.routing = null;
  state.live.performance = { latencyMs: null, rateHz: null };
  $("live-annotated").removeAttribute("src");
  setShellFrameState("live-camera-shell", Boolean(state.live.stream));
  setShellFrameState("live-overlay-shell", false);
  renderLiveStatus({});
  renderLiveInsights(null);
  renderLiveRouting(optimisticRoutingFromConfig());
  updateLiveSignalCards();
  syncLiveChartLabel();
  drawLineChart("live-chart", [], state.live.chartMetric);
}

function stopLiveMode() {
  if (state.live.rafId) {
    window.cancelAnimationFrame(state.live.rafId);
    state.live.rafId = 0;
  }
  state.live.inFlight = false;
  closeLiveSocket();
  if (state.live.stream) {
    state.live.stream.getTracks().forEach((track) => track.stop());
    state.live.stream = null;
  }
  const video = $("live-video");
  if (video) video.srcObject = null;
  $("live-start").hidden = false;
  $("live-stop").hidden = true;
  resetLiveSurface();
}

async function startLiveMode() {
  readConfig("live", liveConfig);
  clearErrors("live");
  const errors = validateConfig(liveConfig, false, "");
  delete errors.file;
  if (Object.keys(errors).length) {
    showErrors("live", errors);
    return;
  }

  try {
    state.live.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    const video = $("live-video");
    video.srcObject = state.live.stream;
    await video.play();
    setShellFrameState("live-camera-shell", true);
    state.live.connectionState = "connecting";
    updateLiveSignalCards();
    await connectLiveSocket();
    $("live-start").hidden = true;
    $("live-stop").hidden = false;
    $("live-server-error").textContent = "";
    resetLiveSurface();
    pushLiveConfig();
    startLiveLoop(video);
  } catch (error) {
    $("live-server-error").textContent = error.message || "Could not access the camera.";
    stopLiveMode();
  }
}

function renderResearchTables(rows, stateRows) {
  setHtml(
    "research-results-table",
    `
      <table>
        <thead>
          <tr>
            <th>Person</th>
            <th>State</th>
            <th>Label</th>
            <th>Shoulder</th>
            <th>Kyphosis</th>
            <th>Lordosis</th>
            <th>Trunk Lean</th>
          </tr>
        </thead>
        <tbody>
          ${rows
            .map(
              (row) => `
                <tr>
                  <td>${row.person_id}</td>
                  <td>${row.state}</td>
                  <td>${row.label}</td>
                  <td>${formatNumber(row.shoulder_tilt_mean)}</td>
                  <td>${formatNumber(row.kyphosis_mean)}</td>
                  <td>${formatNumber(row.lordosis_mean)}</td>
                  <td>${formatNumber(row.trunk_lean_mean)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    `,
  );

  setHtml(
    "research-state-table",
    `
      <table>
        <thead>
          <tr>
            <th>State</th>
            <th>Shoulder</th>
            <th>Kyphosis</th>
            <th>Lordosis</th>
            <th>Trunk Lean</th>
          </tr>
        </thead>
        <tbody>
          ${stateRows
            .map(
              (row) => `
                <tr>
                  <td>${row.state}</td>
                  <td>${formatNumber(row.shoulder_tilt_mean)}</td>
                  <td>${formatNumber(row.kyphosis_mean)}</td>
                  <td>${formatNumber(row.lordosis_mean)}</td>
                  <td>${formatNumber(row.trunk_lean_mean)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    `,
  );
}

function renderResearchCharts(rows) {
  state.researchRows = rows;
  const metric = $("research-metric").value;
  drawBarChart("research-bar-chart", rows, metric);
  drawAreaChart("research-area-chart", rows, metric);
}

function renderResearchWarnings(skipped) {
  renderWarnings("research-skipped", skipped);
}

async function pollResearchJob() {
  if (!state.researchJobId) return;
  const response = await fetch(apiUrl(`/api/research/jobs/${state.researchJobId}`));
  const payload = await response.json();
  $("research-status").textContent = payload.status || "idle";
  $("research-progress").textContent = `${payload.progress || 0}%`;
  $("research-progress-bar").style.width = `${payload.progress || 0}%`;
  $("research-message").textContent = payload.message || "Waiting for a CSV upload.";

  if (payload.status === "completed") {
    clearInterval(state.researchPoller);
    state.researchPoller = null;
    $("research-results").hidden = false;
    if (payload.result.export_csv) {
      $("research-export-link").href = backendAssetUrl(payload.result.export_csv);
      $("research-export-link").hidden = false;
    }
    renderResearchWarnings(payload.result.skipped || []);
    renderResearchTables(payload.result.rows || [], payload.result.state_comparison || []);
    renderResearchCharts(payload.result.rows || []);
  }

  if (payload.status === "failed") {
    clearInterval(state.researchPoller);
    state.researchPoller = null;
    $("research-server-error").textContent =
      Object.values(payload.result?.errors || {})[0] || payload.message || "Researcher mode failed.";
  }
}

async function handleResearchSubmit() {
  readConfig("research", researchConfig);
  clearErrors("research");
  const errors = validateConfig(researchConfig, !state.researchFile, "CSV dataset");
  if (Object.keys(errors).length) {
    showErrors("research", errors);
    return;
  }

  try {
    const response = await fetch(apiUrl("/api/research/jobs"), {
      method: "POST",
      body: formDataFromConfig("csv_file", state.researchFile, researchConfig),
    });
    const payload = await response.json();
    if (!response.ok) {
      showErrors("research", payload.errors || {});
      $("research-server-error").textContent = payload.detail || "Could not start researcher mode.";
      return;
    }

    state.researchJobId = payload.job_id;
    $("research-results").hidden = true;
    $("research-export-link").hidden = true;
    if (state.researchPoller) clearInterval(state.researchPoller);
    state.researchPoller = setInterval(pollResearchJob, 2000);
    pollResearchJob();
  } catch (error) {
    $("research-server-error").textContent = error.message || "Could not start researcher mode.";
  }
}

function bindConfigSync(prefix, callback) {
  document.querySelectorAll(`[data-config-prefix="${prefix}"]`).forEach((node) => {
    node.addEventListener("change", callback);
    node.addEventListener("input", callback);
  });
}

function saveAppearance() {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.ui));
}

function syncAppearanceButtons() {
  document.querySelectorAll(".theme-chip").forEach((button) => {
    button.classList.toggle("active", button.dataset.theme === state.ui.theme);
  });
  document.querySelectorAll(".accent-chip").forEach((button) => {
    button.classList.toggle("active", button.dataset.accent === state.ui.accent);
  });
}

function applyAppearance() {
  document.body.dataset.theme = state.ui.theme;
  document.body.dataset.accent = state.ui.accent;
  syncAppearanceButtons();
  drawLineChart("live-chart", state.live.points, state.live.chartMetric);
  if ($("research-results") && !$("research-results").hidden && state.researchRows.length) {
    renderResearchCharts(state.researchRows);
  }
}

function initAppearanceControls() {
  const allowedThemes = new Set(["midnight", "aurora", "xray", "ember", "ivory"]);
  const allowedAccents = new Set(["cyan", "coral", "lime", "violet"]);
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
    state.ui.theme = allowedThemes.has(saved.theme) ? saved.theme : state.ui.theme;
    state.ui.accent = allowedAccents.has(saved.accent) ? saved.accent : state.ui.accent;
  } catch {
    state.ui.theme = "midnight";
    state.ui.accent = "cyan";
  }

  applyAppearance();

  $("settings-toggle").addEventListener("click", () => {
    $("settings-panel").hidden = !$("settings-panel").hidden;
  });
  $("settings-close").addEventListener("click", () => {
    $("settings-panel").hidden = true;
  });

  document.querySelectorAll(".theme-chip").forEach((button) => {
    button.addEventListener("click", () => {
      state.ui.theme = button.dataset.theme;
      saveAppearance();
      applyAppearance();
    });
  });

  document.querySelectorAll(".accent-chip").forEach((button) => {
    button.addEventListener("click", () => {
      state.ui.accent = button.dataset.accent;
      saveAppearance();
      applyAppearance();
    });
  });
}

function init() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => setActiveTab(button.dataset.tab));
  });
  setActiveTab("upload");

  $("upload-file").addEventListener("change", (event) => {
    state.uploadFile = event.target.files?.[0] || null;
  });
  $("research-file").addEventListener("change", (event) => {
    state.researchFile = event.target.files?.[0] || null;
  });

  $("upload-submit").addEventListener("click", handleUploadSubmit);
  $("research-submit").addEventListener("click", handleResearchSubmit);
  $("live-start").addEventListener("click", startLiveMode);
  $("live-stop").addEventListener("click", stopLiveMode);

  bindConfigSync("live", pushLiveConfig);

  document.querySelectorAll(".live-metric-toggle").forEach((button) => {
    button.addEventListener("click", () => {
      state.live.chartMetric = button.dataset.metric;
      setActiveMetricButton(state.live.chartMetric, ".live-metric-toggle");
      syncLiveChartLabel();
      drawLineChart("live-chart", state.live.points, state.live.chartMetric);
    });
  });

  document.querySelectorAll(".research-metric-toggle").forEach((button) => {
    button.addEventListener("click", async () => {
      document.querySelectorAll(".research-metric-toggle").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      $("research-metric").value = button.dataset.metric;
      if (!state.researchJobId) {
        renderResearchCharts([]);
        return;
      }
      const response = await fetch(apiUrl(`/api/research/jobs/${state.researchJobId}`));
      const payload = await response.json();
      renderResearchCharts(payload.result?.rows || []);
    });
  });

  window.addEventListener("beforeunload", stopLiveMode);

  initAppearanceControls();
  initBackgroundScene();
  initLiveWireframe();
  resetLiveSurface();
  drawBarChart("research-bar-chart", [], "shoulder_tilt_mean");
  drawAreaChart("research-area-chart", [], "shoulder_tilt_mean");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap);
} else {
  bootstrap();
}

async function ensureAppShell() {
  if (document.querySelector(".app-shell")) return;
  const root = $("root");
  if (!root) return;
  const response = await fetch("/app.html");
  if (!response.ok) {
    throw new Error(`Could not load the app template (${response.status}).`);
  }
  root.innerHTML = await response.text();
}

async function bootstrap() {
  try {
    await ensureAppShell();
    init();
  } catch (error) {
    const root = $("root");
    if (root) {
      root.innerHTML = `
        <div style="padding:24px;color:#f4f7fb;font-family:Space Grotesk, sans-serif;">
          <h2 style="margin:0 0 12px;">Frontend failed to load</h2>
          <p style="margin:0;color:#97a8bb;">${String(error)}</p>
        </div>
      `;
    }
  }
}
