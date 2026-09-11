const ARM_LABELS = [
  "L_SHOULDER_PITCH", "L_SHOULDER_ROLL", "L_SHOULDER_YAW", "L_ELBOW_PITCH",
  "L_WRIST_YAW", "L_WRIST_ROLL", "L_WRIST_PITCH", "R_SHOULDER_PITCH",
  "R_SHOULDER_ROLL", "R_SHOULDER_YAW", "R_ELBOW_PITCH", "R_WRIST_YAW",
  "R_WRIST_ROLL", "R_WRIST_PITCH",
];
const LEG_LABELS = [
  "L_HIP_PITCH", "L_HIP_ROLL", "L_HIP_YAW", "L_KNEE_PITCH", "L_ANKLE_PITCH",
  "L_ANKLE_ROLL", "R_HIP_PITCH", "R_HIP_ROLL", "R_HIP_YAW", "R_KNEE_PITCH",
  "R_ANKLE_PITCH", "R_ANKLE_ROLL",
];
const CHART_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#be123c", "#65a30d", "#7c3aed"];
const TELEMETRY_ACT_COLOR = "#f97316";
const TELEMETRY_OBS_COLOR = "#2563eb";
const SELECT_LABELS = {
  unity_hybrid: "unity(hybrid)",
  always_1: "always 1",
};

const state = {
  status: null,
  optionsReady: false,
  hybridSettingsReady: false,
  experimentTasks: [],
  modeWorkerNames: [],
  tasks: [],
  fileDialog: null,
  walkingPolicyDefaults: {},
  activePage: "teleop",
  viserUrl: "",
  viserLoaded: false,
  cameraStreamTimers: [],
  cameraStreamsActive: true,
  statusInFlight: false,
  telemetryInFlight: false,
  simStereoBaselineDirty: false,
  history: {arm: [], leg: []},
  charts: {},
  logPaused: false,
  datasetViewer: {
    ready: false,
    datasets: [],
    selectedDataset: null,
    data: null,
    frameIdx: 0,
    playing: false,
    playTimer: null,
    loadInFlight: false,
    imageLoading: false,
    pendingImageFrame: null,
    loadedImageFrame: null,
    imageRequestToken: 0,
    selectedSegment: -1,
    segmentDirty: false,
    classDirty: false,
    savingLabels: false,
  },
};

const $ = (id) => document.getElementById(id);

async function api(path, payload) {
  const options = payload === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  };
  const res = await fetch(path, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function setMessage(text, isError = false) {
  const el = $("message");
  el.textContent = text || "";
  el.style.color = isError ? "#b91c1c" : "#666f7a";
}

const LOG_LEVEL_SEVERITY = {
  DEBUG: 10,
  INFO: 20,
  OUTPUT: 20,
  STDOUT: 20,
  WARNING: 30,
  WARN: 30,
  STDERR: 30,
  ERROR: 40,
  EXCEPTION: 40,
  CRITICAL: 50,
  FATAL: 50,
};

function normalizeLogLevel(value) {
  const level = String(value || "OUTPUT").toUpperCase();
  return level === "WARN" ? "WARNING" : level;
}

function logSeverity(level) {
  return LOG_LEVEL_SEVERITY[normalizeLogLevel(level)] ?? 20;
}

function simplifyLogSource(value) {
  const raw = String(value || "").trim();
  if (!raw) return "runtime";
  if (raw.includes("web_ui")) return "web_ui";

  let source = raw.split("/")[0] || raw;
  const parts = source.split(".");
  source = parts[parts.length - 1] || source;
  source = source.replace(/^worker_/, "");
  source = source.replace(/^igris_/, "");
  if (source === "__main__") return "main";
  return source || "runtime";
}

function parseLogLine(value) {
  const raw = String(value ?? "");
  let text = raw;
  let streamSource = "";
  const streamMatch = text.match(/^\[([^\]]+)\]\s*(.*)$/);
  if (streamMatch && streamMatch[1].includes("/")) {
    streamSource = streamMatch[1];
    text = streamMatch[2];
  }

  const recordMatch = text.match(
    /^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d{3})?)\s+([A-Z]+)\s+\[([^\]]+)\]\s*(.*)$/
  );
  if (recordMatch) {
    const level = normalizeLogLevel(recordMatch[2]);
    const source = simplifyLogSource(streamSource || recordMatch[3]);
    return {
      raw,
      time: recordMatch[1].slice(11, 19),
      level,
      source,
      message: recordMatch[4],
      severity: logSeverity(level),
      searchText: `${raw} ${source} ${level}`.toLowerCase(),
    };
  }

  const source = streamSource
    ? simplifyLogSource(streamSource)
    : simplifyLogSource((text.match(/^\[([A-Za-z0-9_.:-]+)\]/) || [])[1]);
  let level = streamSource.endsWith("/stderr") ? "STDERR" : "OUTPUT";
  const levelMatch = text.match(/\b(CRITICAL|FATAL|ERROR|EXCEPTION|WARNING|WARN|INFO|DEBUG)\b/i);
  if (levelMatch) level = normalizeLogLevel(levelMatch[1]);

  return {
    raw,
    time: "",
    level,
    source,
    message: text,
    severity: logSeverity(level),
    searchText: `${raw} ${source} ${level}`.toLowerCase(),
  };
}

function updateLogSourceFilter(items) {
  const select = $("logSourceFilter");
  if (!select) return;

  const previous = select.value || "all";
  const sources = [...new Set(items.map((item) => item.source).filter(Boolean))].sort();
  select.innerHTML = "";

  const allOption = document.createElement("option");
  allOption.value = "all";
  allOption.textContent = "All sources";
  select.appendChild(allOption);

  sources.forEach((source) => {
    const option = document.createElement("option");
    option.value = source;
    option.textContent = source;
    select.appendChild(option);
  });
  select.value = sources.includes(previous) ? previous : "all";
}

function logMatchesFilters(item) {
  const levelFilter = $("logLevelFilter")?.value || "all";
  if (levelFilter === "warn" && item.severity < 30) return false;
  if (levelFilter === "error" && item.severity < 40) return false;

  const sourceFilter = $("logSourceFilter")?.value || "all";
  if (sourceFilter !== "all" && item.source !== sourceFilter) return false;

  const query = String($("logSearchInput")?.value || "").trim().toLowerCase();
  if (query && !item.searchText.includes(query)) return false;
  return true;
}

function appendLogCell(row, className, text) {
  const cell = document.createElement("span");
  cell.className = className;
  cell.textContent = text;
  row.appendChild(cell);
}

function renderLogs(lines) {
  const box = $("logs");
  if (!box) return;

  const parsed = (lines || []).map(parseLogLine);
  updateLogSourceFilter(parsed);
  const filtered = parsed.filter(logMatchesFilters);
  const previousScrollTop = box.scrollTop;
  const shouldAutoscroll = box.scrollTop >= box.scrollHeight - box.clientHeight - 8;
  const fragment = document.createDocumentFragment();

  if (!filtered.length) {
    const row = document.createElement("div");
    row.className = "log-row empty";
    row.textContent = parsed.length ? "No logs match the current filters." : "No logs yet.";
    fragment.appendChild(row);
  } else {
    filtered.forEach((item) => {
      const row = document.createElement("div");
      row.className = `log-row level-${item.level.toLowerCase()}`;
      appendLogCell(row, "log-time", item.time || "--:--:--");
      appendLogCell(row, "log-level", item.level);
      appendLogCell(row, "log-source", item.source);
      appendLogCell(row, "log-message", item.message);
      fragment.appendChild(row);
    });
  }

  box.replaceChildren(fragment);
  box.scrollTop = shouldAutoscroll ? box.scrollHeight : previousScrollTop;

  const count = $("logCount");
  if (count) count.textContent = `${filtered.length} / ${parsed.length}`;
}

function formatMonitorTime(timestamp) {
  const value = Number(timestamp || 0);
  if (!Number.isFinite(value) || value <= 0) return "--:--:--";
  return new Date(value * 1000).toTimeString().slice(0, 8);
}

function formatMonitorValues(values) {
  const list = Array.isArray(values) ? values : [];
  if (!list.length) return "-";
  return `[${list.map((value) => value === null || value === undefined ? "nan" : Number(value).toFixed(3)).join(", ")}]`;
}

function formatMonitorScalar(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(3) : "-";
}

function monitorBadge(text) {
  const span = document.createElement("span");
  const label = String(text || "STOPPED");
  span.className = `monitor-badge ${label}`;
  span.textContent = label;
  return span;
}

function appendMonitorLine(section, label, value) {
  const row = document.createElement("div");
  row.className = "monitor-line";
  const key = document.createElement("span");
  key.className = "monitor-label";
  key.textContent = label;
  const val = document.createElement("span");
  val.className = "monitor-value";
  val.textContent = value;
  row.append(key, val);
  section.appendChild(row);
}

function appendMonitorTitle(parent, title, badges = []) {
  const row = document.createElement("div");
  row.className = "monitor-title";
  const label = document.createElement("span");
  label.textContent = title;
  row.appendChild(label);
  badges.forEach((badge) => row.appendChild(monitorBadge(badge)));
  parent.appendChild(row);
}

function formatPoseSummary(pose) {
  if (!pose?.valid) return "-";
  return `xyz ${formatMonitorValues(pose.xyz)} rpy ${formatMonitorValues(pose.rpy_deg)}`;
}

function formatHandPoints(hand) {
  if (!hand?.valid) return "-";
  return `centroid ${formatMonitorValues(hand.centroid)} first ${formatMonitorValues(hand.first)} spread ${formatMonitorScalar(hand.spread)}`;
}

function renderInputMonitor(monitor) {
  const box = $("inputMonitor");
  if (!box) return;

  const time = $("inputMonitorTime");
  if (time) time.textContent = formatMonitorTime(monitor?.timestamp);

  if (!monitor) {
    const empty = document.createElement("div");
    empty.className = "monitor-empty";
    empty.textContent = "No input monitor data.";
    box.replaceChildren(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  const mode = monitor.mode || {};
  const leader = monitor.leader_arm || {};
  const vr = monitor.vr || {};

  const summary = document.createElement("section");
  summary.className = "monitor-section";
  appendMonitorTitle(summary, "Mode");
  appendMonitorLine(summary, "applied", mode.applied || "-");
  appendMonitorLine(summary, "device", mode.teleop_device || "-");
  appendMonitorLine(summary, "hand src", mode.hand_source || "-");
  fragment.appendChild(summary);

  const leaderSection = document.createElement("section");
  leaderSection.className = "monitor-section";
  appendMonitorTitle(leaderSection, "Leader Arm", [
    leader.enabled ? leader.connection : "STOPPED",
    leader.arm_signal || "WAIT",
  ]);
  appendMonitorLine(leaderSection, "leader ros", leader.leader_ros_status || "STOPPED");
  appendMonitorLine(
    leaderSection,
    "workers",
    (leader.leader_ros_workers || []).map((item) => `${item.name}:${item.alive ? "on" : "off"}`).join(" ") || "-",
  );
  appendMonitorLine(
    leaderSection,
    "arm",
    `norm ${formatMonitorScalar(leader.act_arm?.norm)} ${leader.arm_changed ? "changed" : "steady"}`,
  );
  appendMonitorLine(leaderSection, "act_arm", formatMonitorValues(leader.act_arm?.values));
  appendMonitorLine(
    leaderSection,
    "hand",
    `src ${leader.hand_source || "-"} norm ${formatMonitorScalar(leader.act_hand?.norm)} ${leader.hand_changed ? "changed" : "steady"}`,
  );
  appendMonitorLine(leaderSection, "act_hand", formatMonitorValues(leader.act_hand?.values));
  fragment.appendChild(leaderSection);

  const vrSection = document.createElement("section");
  vrSection.className = "monitor-section";
  appendMonitorTitle(vrSection, "VR", [
    vr.enabled ? vr.connection : "STOPPED",
    vr.signal || "WAIT",
  ]);
  appendMonitorLine(vrSection, "ros tcp", vr.ros_tcp_alive ? "ALIVE" : "STOPPED");
  appendMonitorLine(vrSection, "poses", `${vr.valid_pose_count || 0} valid ${vr.changed ? "changed" : "steady"}`);
  appendMonitorLine(
    vrSection,
    "torso task",
    `alpha ${formatMonitorScalar(vr.torso_alpha)} valid ${formatMonitorScalar(vr.torso_source_valid)}`,
  );
  appendMonitorLine(
    vrSection,
    "chest task",
    `alpha ${formatMonitorScalar(vr.chest_alpha)} valid ${formatMonitorScalar(vr.chest_source_valid)} c ${formatMonitorScalar(vr.left_controller_confidence)}/${formatMonitorScalar(vr.right_controller_confidence)}`,
  );
  appendMonitorLine(vrSection, "head", formatPoseSummary(vr.poses?.head));
  appendMonitorLine(vrSection, "torso", formatPoseSummary(vr.poses?.torso));
  appendMonitorLine(vrSection, "chest", formatPoseSummary(vr.poses?.chest));
  appendMonitorLine(vrSection, "left", formatPoseSummary(vr.poses?.left_wrist));
  appendMonitorLine(vrSection, "right", formatPoseSummary(vr.poses?.right_wrist));
  appendMonitorLine(vrSection, "l hand", formatHandPoints(vr.hands?.left));
  appendMonitorLine(vrSection, "r hand", formatHandPoints(vr.hands?.right));
  fragment.appendChild(vrSection);

  box.replaceChildren(fragment);
}

function normalizeViserUrl(value) {
  const clean = String(value || "").trim();
  if (!clean) return "";
  if (/^https?:\/\//i.test(clean)) return clean;
  return `http://${clean}`;
}

function setViserUrl(value, updateInput = true, forceReload = false) {
  const url = normalizeViserUrl(value);
  state.viserUrl = url;
  const input = $("viserUrlInput");
  const link = $("viserOpenLink");
  if (input && updateInput) input.value = url;
  if (link) {
    if (url) {
      link.href = url;
      link.classList.remove("disabled");
    } else {
      link.removeAttribute("href");
      link.classList.add("disabled");
    }
  }
  if (state.activePage === "visualizer") loadViserFrame(forceReload);
}

function loadViserFrame(forceReload = false) {
  const frame = $("viserFrame");
  if (!frame || !state.viserUrl) return;
  if (forceReload || !state.viserLoaded || frame.src !== state.viserUrl) {
    frame.src = state.viserUrl;
    state.viserLoaded = true;
  }
}

function cameraImages() {
  return Array.from(document.querySelectorAll(".camera-grid img"));
}

function initCameraStreams() {
  cameraImages().forEach((img) => {
    if (!img.dataset.streamSrc) {
      const src = img.getAttribute("src") || "";
      if (src) img.dataset.streamSrc = src;
    }
  });
}

function setCameraStreamsActive(active) {
  state.cameraStreamTimers.forEach((timer) => clearTimeout(timer));
  state.cameraStreamTimers = [];
  state.cameraStreamsActive = Boolean(active);

  cameraImages().forEach((img, idx) => {
    const src = img.dataset.streamSrc || "";
    if (!src) return;

    if (!active) {
      if (img.getAttribute("src")) img.removeAttribute("src");
      return;
    }

    if (img.getAttribute("src") === src) return;
    const timer = setTimeout(() => {
      if (state.activePage === "teleop") img.setAttribute("src", src);
    }, idx * 180);
    state.cameraStreamTimers.push(timer);
  });
}

function showPage(page) {
  const next = page === "hybrid"
    ? "hybrid"
    : page === "experiments"
      ? "experiments"
      : page === "visualizer"
        ? "visualizer"
        : page === "dataset"
          ? "dataset"
          : "teleop";
  state.activePage = next;
  if (window.location.hash !== `#${next}`) {
    window.history.replaceState(null, "", `#${next}`);
  }
  document.querySelectorAll("[data-page-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.pageTab === next);
  });
  $("teleopPage").classList.toggle("hidden", next !== "teleop");
  $("hybridPage").classList.toggle("hybrid-inactive", next !== "hybrid");
  $("experimentsPage").classList.toggle("experiments-inactive", next !== "experiments");
  $("visualizerPage").classList.toggle("visualizer-inactive", next !== "visualizer");
  $("datasetPage").classList.toggle("dataset-inactive", next !== "dataset");
  setCameraStreamsActive(next === "teleop");
  if (next !== "dataset") setDatasetPlaying(false);
  if (next === "visualizer") {
    loadViserFrame(false);
    renderRuntimeDiagnostics(state.status?.runtime_diagnostics || null);
    pollTelemetry();
  }
  if (next === "hybrid") refreshHybridPreviewImages(true);
  if (next === "dataset") {
    loadDatasetListOnce();
    redrawDatasetViewer();
  }
}

function fillSelect(el, values, selected, includeNone = false) {
  const previous = selected ?? el.value;
  el.innerHTML = "";
  if (includeNone) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "<None>";
    el.appendChild(option);
  }
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = SELECT_LABELS[value] || value;
    el.appendChild(option);
  });
  el.value = previous ?? "";
}

function selectedMode() {
  return $("modeSelect").value || null;
}

function selectedPayload() {
  const inferencePolicy = $("inferencePolicySelect").value || null;
  const nAction = inferencePolicy === "diffusion_policy"
    ? Number($("inferenceDiffusionActionStepInput").value || 0) || null
    : Number($("inferenceActionStepInput").value || 0) || null;
  return {
    mode: selectedMode(),
    teleop_device: $("teleopDeviceSelect").value || null,
    teleop_hand_source: $("teleopHandSourceSelect").value || null,
    camera_mode: $("cameraModeSelect").value,
    walking_policy_profile: $("walkingProfileInput").value || null,
    walking_policy_path: $("walkingPolicyPathInput").value || null,
    walking_startup_blend_enabled: $("walkingStartupBlendInput").checked,
    inference_policy: inferencePolicy,
    inference_dataset_folder: $("inferenceDatasetInput").value || null,
    inference_pretrained_rel: $("inferenceCheckpointInput").value || null,
    inference_chunk_size: Number($("inferenceChunkInput").value || 0) || null,
    inference_horizon: Number($("inferenceHorizonInput").value || 0) || null,
    inference_n_action_step: nAction,
    inference_use_dataset_state: $("inferenceUseStateInput").checked,
    inference_use_dataset_camera: $("inferenceUseVideoInput").checked,
    inference_use_dataset_tau: $("inferenceUseTorqueInput").checked,
    inference_instruction: $("inferenceInstructionInput").value || null,
    replay_dataset_folder: $("replayDatasetInput").value || null,
  };
}

function syncOptions(status) {
  if (!status || state.optionsReady) return;
  const options = status.options || {};
  fillSelect($("modeSelect"), options.modes || [], status.selected.mode, true);
  fillSelect($("teleopDeviceSelect"), options.teleop_devices || [], status.selected.teleop_device, true);
  fillSelect($("cameraModeSelect"), options.camera_modes || [], status.camera_mode, false);
  const reliabilityVariants = options.reliability_model_variants || ["rnn", "histgb", "always_1"];
  fillSelect(
    $("reliabilityHandModelSelect"),
    reliabilityVariants,
    status.reliability_runtime?.hand_model_variant || reliabilityVariants[0],
    false,
  );
  const hybridDevices = options.hybrid_camera_devices || [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const hybridConfig = status.hybrid_teleop?.config || {};
  fillSelect($("hybridLeftDeviceSelect"), hybridDevices, hybridConfig.left_device ?? 1, false);
  fillSelect($("hybridRightDeviceSelect"), hybridDevices, hybridConfig.right_device ?? 0, false);
  if (!state.hybridSettingsReady) {
    $("hybridTrapezoidToggle").checked = hybridConfig.trapezoid_preprocess ?? true;
    $("hybridSwapLrToggle").checked = hybridConfig.swap_lr ?? true;
    $("hybridMirrorToggle").checked = hybridConfig.mirror ?? false;
    const bottomWidth = Number(hybridConfig.trapezoid_bottom_width ?? 320);
    $("hybridBottomWidthRange").value = String(bottomWidth);
    $("hybridBottomWidthInput").value = String(bottomWidth);
    state.hybridSettingsReady = true;
  }
  fillSelect(
    $("reliabilityControllerModelSelect"),
    reliabilityVariants,
    status.reliability_runtime?.controller_model_variant || reliabilityVariants[0],
    false,
  );
  state.experimentTasks = [...(options.experiment_tasks || status.experiments?.tasks || [])];
  const experimentSelect = $("experimentTaskSelect");
  experimentSelect.innerHTML = "";
  const experimentPlaceholder = document.createElement("option");
  experimentPlaceholder.value = "";
  experimentPlaceholder.textContent = "Select task";
  experimentSelect.appendChild(experimentPlaceholder);
  state.experimentTasks.forEach((task) => {
    const option = document.createElement("option");
    option.value = String(task.task_id);
    option.textContent = task.name;
    experimentSelect.appendChild(option);
  });
  const requestedTaskId = Number(status.experiments?.requested_task_id || 0);
  if (state.experimentTasks.some((task) => Number(task.task_id) === requestedTaskId)) {
    experimentSelect.value = String(requestedTaskId);
  }
  $("collectDatasetInput").value = options.collect_dataset_repo_id || "IGRIS_C";
  state.tasks = [...(options.collect_tasks || ["Pick and Place"])];
  $("taskInput").value = status.record?.task_name || state.tasks[0] || "";
  state.walkingPolicyDefaults = options.walking_policy_defaults || {};
  fillSelect(
    $("walkingProfileInput"),
    options.walking_profiles || Object.keys(state.walkingPolicyDefaults),
    options.initial_walking_policy_profile,
    false,
  );
  $("walkingPolicyPathInput").value = options.initial_walking_policy_path
    || state.walkingPolicyDefaults[$("walkingProfileInput").value]
    || "";
  setViserUrl(options.viser_url || "http://127.0.0.1:8080/");
  state.optionsReady = true;
  updateHandSourceOptions(false);
  renderTaskList();
}

function updateHandSourceOptions(push = true) {
  const status = state.status;
  const byDevice = status?.options?.teleop_hand_sources_by_device || {};
  const device = $("teleopDeviceSelect").value || null;
  const values = byDevice[device] || [];
  const preferred = $("teleopHandSourceSelect").value || status?.selected?.teleop_hand_source;
  fillSelect($("teleopHandSourceSelect"), values, preferred, values.length === 0);
  updateModeVisibility();
  refreshModeWorkers();
  if (push) pushSelection();
}

function updatePolicyVisibility() {
  const policy = $("inferencePolicySelect").value;
  document.querySelectorAll(".act-policy-field").forEach((el) => {
    el.classList.toggle("hidden", policy !== "act");
  });
  document.querySelectorAll(".diffusion-policy-field").forEach((el) => {
    el.classList.toggle("hidden", policy !== "diffusion_policy");
  });
  document.querySelectorAll(".instruction-policy-field").forEach((el) => {
    el.classList.toggle("hidden", !["pi0", "pi0.5"].includes(policy));
  });
}

function updateModeVisibility() {
  const mode = selectedMode();
  document.querySelectorAll(".mode-field").forEach((el) => {
    el.classList.toggle("hidden", el.dataset.mode !== mode);
  });
  document.querySelectorAll(".teleop-field").forEach((el) => {
    el.classList.toggle("hidden", mode !== "teleop");
  });
  document.querySelectorAll(".teleop-hand-field").forEach((el) => {
    const show = mode === "teleop" && $("teleopDeviceSelect").value === "vr_masterarm";
    el.classList.toggle("hidden", !show);
  });
  document.querySelectorAll(".teleop-reliability-field").forEach((el) => {
    el.classList.toggle("hidden", mode !== "teleop");
  });
  updatePolicyVisibility();
}

async function pushSelection() {
  if (state.status?.ui?.selection_locked) return;
  try {
    const status = await api("/api/select-mode", selectedPayload());
    renderStatus(status);
  } catch (err) {
    setMessage(err.message, true);
  }
}

async function refreshModeWorkers() {
  const payload = selectedPayload();
  const params = new URLSearchParams();
  if (payload.mode) params.set("mode", payload.mode);
  if (payload.teleop_device) params.set("teleop_device", payload.teleop_device);
  if (payload.inference_policy) params.set("inference_policy", payload.inference_policy);
  try {
    const data = await api(`/api/mode-workers?${params.toString()}`);
    renderModeWorkers((data.workers || []).map((name) => ({name, status: "STOPPED"})), hintForSelection());
  } catch (err) {
    setMessage(err.message, true);
  }
}

function hintForSelection() {
  const mode = selectedMode();
  if (!mode) return "mode를 먼저 선택하세요.";
  if (mode === "teleop" && !$("teleopDeviceSelect").value) return "teleop device를 선택하세요.";
  return "";
}

function renderModeWorkers(workers, hint = "") {
  const box = $("modeWorkers");
  box.innerHTML = "";
  state.modeWorkerNames = workers.map((worker) => worker.name);
  if (hint) {
    const div = document.createElement("div");
    div.className = "mode-worker-hint";
    div.textContent = hint;
    box.appendChild(div);
    return;
  }
  if (!workers.length) {
    const div = document.createElement("div");
    div.className = "mode-worker-hint";
    div.textContent = "선택 가능한 mode worker가 없습니다.";
    box.appendChild(div);
    return;
  }
  workers.forEach((worker) => {
    const row = document.createElement("div");
    row.className = "mode-worker-row";
    row.dataset.worker = worker.name;
    const name = document.createElement("span");
    name.textContent = worker.name;
    const status = document.createElement("span");
    status.className = `badge ${worker.status || "STOPPED"}`;
    status.textContent = worker.status || "STOPPED";
    row.append(name, status);
    box.appendChild(row);
  });
}

function selectedWorkers() {
  return [...state.modeWorkerNames];
}

function selectedTask() {
  return $("taskInput").value.trim();
}

function setSelectedTask(task) {
  const clean = (task || "").trim();
  $("taskInput").value = clean;
  if (clean && !state.tasks.includes(clean)) state.tasks.push(clean);
  renderTaskList();
  if (state.status) renderStatus(state.status);
}

function renderTaskList() {
  const box = $("taskList");
  if (!box) return;
  box.innerHTML = "";
  const selected = selectedTask();
  if (!state.tasks.length) {
    const row = document.createElement("div");
    row.className = "task-row";
    row.textContent = "No task";
    box.appendChild(row);
    return;
  }
  state.tasks.forEach((task) => {
    const row = document.createElement("div");
    row.className = `task-row${task === selected ? " selected" : ""}`;
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = task === selected;
    checkbox.tabIndex = -1;
    const label = document.createElement("span");
    label.textContent = task;
    row.append(checkbox, label);
    row.onclick = () => setSelectedTask(task);
    box.appendChild(row);
  });
}

function fileBrowserProfile(kind) {
  return kind === "walking_policy" ? $("walkingProfileInput").value || null : null;
}

function updateWalkingModelForProfile() {
  const profile = $("walkingProfileInput").value || null;
  const nextDefault = profile ? state.walkingPolicyDefaults[profile] : null;
  if (!nextDefault) return;

  const current = $("walkingPolicyPathInput").value.trim();
  const knownDefaults = new Set(Object.values(state.walkingPolicyDefaults || {}));
  if (!current || knownDefaults.has(current)) {
    $("walkingPolicyPathInput").value = nextDefault;
  }
}

function syncWalkingProfileToModel(value) {
  const lower = String(value || "").toLowerCase();
  if (lower.endsWith(".onnx") && $("walkingProfileInput").value !== "v2_fast_sac") {
    $("walkingProfileInput").value = "v2_fast_sac";
  } else if (lower.endsWith(".pt") && $("walkingProfileInput").value !== "v1") {
    $("walkingProfileInput").value = "v1";
  }
}

async function openFileDialog(kind, targetInputId) {
  state.fileDialog = {
    kind,
    targetInputId,
    parent: null,
    selectedValue: null,
    selectedLabel: "",
    selectType: "directory",
  };
  $("fileDialog").classList.remove("hidden");
  await loadFileDialog(null);
}

function closeFileDialog() {
  $("fileDialog").classList.add("hidden");
  state.fileDialog = null;
}

async function loadFileDialog(path = null) {
  const dialog = state.fileDialog;
  if (!dialog) return;
  const params = new URLSearchParams();
  params.set("kind", dialog.kind);
  if (path) params.set("path", path);
  const profile = fileBrowserProfile(dialog.kind);
  if (profile) params.set("profile", profile);
  try {
    const data = await api(`/api/file-browser?${params.toString()}`);
    renderFileDialog(data);
  } catch (err) {
    setMessage(err.message, true);
  }
}

function setFileDialogSelection(value, label) {
  const dialog = state.fileDialog;
  if (!dialog) return;
  dialog.selectedValue = value;
  dialog.selectedLabel = label || value;
  $("fileDialogSelected").textContent = dialog.selectedLabel ? `selected: ${dialog.selectedLabel}` : "";
  $("fileDialogSelectButton").disabled = !dialog.selectedValue;
}

function selectFileDialogValue(value) {
  const dialog = state.fileDialog;
  if (!dialog || !value) return;
  $(dialog.targetInputId).value = value;
  if (dialog.targetInputId === "walkingPolicyPathInput") {
    syncWalkingProfileToModel(value);
  }
  closeFileDialog();
  if (dialog.targetInputId === "datasetRootInput") {
    state.datasetViewer.ready = false;
    state.datasetViewer.selectedDataset = null;
    state.datasetViewer.data = null;
    state.datasetViewer.frameIdx = 0;
    setDatasetPlaying(false);
    renderDatasetViewerData();
    loadDatasetList();
    return;
  }
  if (state.status) renderStatus(state.status);
}

function renderFileDialog(data) {
  const dialog = state.fileDialog;
  if (!dialog) return;
  dialog.parent = data.parent || null;
  dialog.selectType = data.select_type || "directory";
  dialog.selectedValue = null;
  dialog.selectedLabel = "";

  $("fileDialogTitle").textContent = data.title || "Select path";
  $("fileDialogRoot").textContent = `root: ${data.base_display || data.base_root || ""}`
    + (data.preferred_root ? ` | default: ${data.preferred_root}` : "");
  $("fileDialogCwd").textContent = `current: ${data.cwd_display || data.cwd || ""}`;
  $("fileDialogUpButton").disabled = !dialog.parent;
  $("fileDialogSelectCurrentButton").classList.toggle("hidden", !data.current_selectable);
  $("fileDialogSelectCurrentButton").disabled = !data.current_selectable;
  $("fileDialogSelectCurrentButton").onclick = () => {
    setFileDialogSelection(data.current_value, data.cwd_display || data.current_value);
  };

  const box = $("fileDialogEntries");
  box.innerHTML = "";
  (data.entries || []).forEach((entry) => {
    const row = document.createElement("div");
    row.className = "file-entry";
    row.dataset.path = entry.path;

    const icon = document.createElement("span");
    icon.textContent = entry.type === "directory" ? "dir" : "file";
    icon.className = "file-entry-type";
    const name = document.createElement("span");
    name.className = "file-entry-name";
    name.textContent = entry.name;
    const type = document.createElement("span");
    type.className = "file-entry-type";
    type.textContent = entry.selectable ? "selectable" : entry.type;

    row.append(icon, name, type);
    row.onclick = () => {
      document.querySelectorAll(".file-entry").forEach((el) => el.classList.remove("selected"));
      row.classList.add("selected");
      if (entry.selectable) {
        setFileDialogSelection(entry.display, entry.display);
      } else {
        setFileDialogSelection(null, "");
      }
    };
    row.ondblclick = () => {
      if (entry.type === "directory") {
        loadFileDialog(entry.path);
      } else if (entry.selectable) {
        selectFileDialogValue(entry.display);
      }
    };
    box.appendChild(row);
  });
  if (!(data.entries || []).length) {
    const empty = document.createElement("div");
    empty.className = "file-entry";
    empty.textContent = "No entries";
    box.appendChild(empty);
  }
  setFileDialogSelection(null, "");
}

function computeCanApply(status) {
  const mode = selectedMode();
  const canApplyMode = Boolean(mode && (mode !== "teleop" || $("teleopDeviceSelect").value));
  const cameraDirty = $("cameraModeSelect").value && $("cameraModeSelect").value !== status.camera_mode;
  return canApplyMode || cameraDirty;
}

function renderStatus(status) {
  if (!status) return;
  state.status = status;
  syncOptions(status);
  updateModeVisibility();

  const levels = status.levels || {};
  const ui = status.ui || {};
  const applied = status.applied || {};
  const record = status.record || {};
  const dataset = status.dataset || {};
  const walking = status.walking || {};
  const isWalking = selectedMode() === "walking";

  document.querySelectorAll("[data-level]").forEach((button) => {
    const level = button.dataset.level;
    button.dataset.active = levels[level] ? "true" : "false";
    button.disabled = false;
    if (level === "ready") button.disabled = !ui.ready_enabled;
    if (level === "start") button.disabled = !ui.start_enabled;
    if (level === "home") button.disabled = !ui.home_enabled;
    if (level === "hand_init") {
      button.disabled = !ui.hand_init_enabled && !levels[level];
      button.textContent = levels[level] ? "pending" : "trigger";
    }
  });
  $("shutdownStateLabel").textContent = levels.shutdown ? "SET" : "CLEAR";
  $("cameraStartButton").dataset.active = levels.camera ? "true" : "false";
  $("cameraStartButton").textContent = levels.camera ? "Stop" : "Start";

  const simStereo = status.sim_stereo || {};
  const simStereoEnabled = Boolean(simStereo.enabled);
  const simStereoRow = $("simStereoBaselineRow");
  const simStereoRange = $("simStereoBaselineRange");
  const simStereoInput = $("simStereoBaselineInput");
  const simStereoApply = $("simStereoBaselineApplyButton");
  simStereoRow.classList.toggle("hidden", !simStereoEnabled);
  if (!simStereoEnabled) state.simStereoBaselineDirty = false;

  const minBaselineMm = Math.round(Number(simStereo.min_baseline_m || 0.04) * 1000);
  const maxBaselineMm = Math.round(Number(simStereo.max_baseline_m || 0.20) * 1000);
  simStereoRange.min = String(minBaselineMm);
  simStereoRange.max = String(maxBaselineMm);
  simStereoInput.min = String(minBaselineMm);
  simStereoInput.max = String(maxBaselineMm);
  if (
    !state.simStereoBaselineDirty
    && document.activeElement !== simStereoRange
    && document.activeElement !== simStereoInput
  ) {
    const baselineMm = Math.round(Number(simStereo.baseline_m || 0.12) * 1000);
    simStereoRange.value = String(baselineMm);
    simStereoInput.value = String(baselineMm);
  }
  const editedBaselineMm = Number(simStereoInput.value);
  const validBaseline = Number.isFinite(editedBaselineMm)
    && editedBaselineMm >= minBaselineMm
    && editedBaselineMm <= maxBaselineMm;
  simStereoRange.disabled = !simStereoEnabled;
  simStereoInput.disabled = !simStereoEnabled;
  simStereoApply.disabled = !simStereoEnabled || !state.simStereoBaselineDirty || !validBaseline;

  $("modeSelect").disabled = Boolean(ui.selection_locked);
  $("teleopDeviceSelect").disabled = Boolean(ui.selection_locked) || selectedMode() !== "teleop";
  $("teleopHandSourceSelect").disabled = Boolean(ui.selection_locked)
    || selectedMode() !== "teleop"
    || $("teleopDeviceSelect").value !== "vr_masterarm";
  $("inferencePolicySelect").disabled = Boolean(ui.selection_locked) || selectedMode() !== "inference";
  $("walkingStartupBlendInput").disabled = Boolean(ui.selection_locked) || selectedMode() !== "walking";
  $("walkingPolicyBrowseButton").disabled = !isWalking;
  $("inferenceDatasetBrowseButton").disabled = selectedMode() !== "inference";
  $("inferenceCheckpointBrowseButton").disabled = selectedMode() !== "inference";
  $("replayDatasetBrowseButton").disabled = selectedMode() !== "replay";
  $("applyModeButton").disabled = !computeCanApply(status);

  ["walkVx", "walkVy", "walkDyaw"].forEach((id) => { $(id).disabled = !isWalking; });
  $("walkingUpdateButton").disabled = !isWalking;
  $("walkingZeroButton").disabled = !isWalking;
  $("walkingStartButton").disabled = !ui.walking_start_enabled;
  $("walkingStartButton").dataset.active = walking.policy_enabled ? "true" : "false";
  $("walkingStartButton").textContent = walking.policy_enabled ? "Walking Stop" : "Walking Start";

  if (document.activeElement !== $("walkVx")) $("walkVx").value = Number(walking.vx || 0).toFixed(3);
  if (document.activeElement !== $("walkVy")) $("walkVy").value = Number(walking.vy || 0).toFixed(3);
  if (document.activeElement !== $("walkDyaw")) $("walkDyaw").value = Number(walking.dyaw || 0).toFixed(3);

  document.querySelector("[data-record='start']").disabled = !ui.record_start_enabled || !$("taskInput").value.trim();
  document.querySelector("[data-record='done']").disabled = !ui.record_done_enabled;
  document.querySelector("[data-record='reset']").disabled = !ui.record_reset_enabled;
  $("collectDatasetInput").disabled = Boolean((status.workers?.manual_running || []).includes("collect_data"));
  if (record.task_name && !selectedTask()) setSelectedTask(record.task_name);
  renderTaskList();

  $("recordStatus").textContent = `${record.state || "idle"}${selectedTask() ? ` | task: ${selectedTask()}` : ""}`;
  $("datasetStatus").textContent = dataset.current_episode_frames > 0
    ? `recording frames: ${dataset.current_episode_frames}`
    : dataset.folder
      ? `dataset folder: ${dataset.folder}`
      : `total saved frames: ${dataset.total_saved_frames || 0}, episodes: ${dataset.num_episodes || 0}`;

  const modeWorkerStatus = status.workers?.mode_workers || {};
  if (modeWorkerStatus.workers) renderModeWorkers(modeWorkerStatus.workers, modeWorkerStatus.hint || "");
  renderManualWorkers(status.workers?.manual_groups || []);

  if (!state.logPaused) renderLogs(status.logs || []);
  renderInputMonitor(status.input_monitor || null);
  renderRuntimeDiagnostics(status.runtime_diagnostics || null);
  renderReliabilityRuntime(status.reliability_runtime || {});
  renderHybridTeleop(status.hybrid_teleop || {});
  renderExperimentStatus(status.experiments || {});

  const selectedText = applied.mode ? `applied: ${applied.mode}` : "mode not applied";
  setMessage(`${selectedText} | ${status.message || "ready"}`);
}

function selectedExperimentTask() {
  const taskId = Number($("experimentTaskSelect")?.value || 0);
  return state.experimentTasks.find((task) => Number(task.task_id) === taskId) || null;
}

function renderExperimentTaskSpec() {
  const task = selectedExperimentTask();
  $("experimentWorkspace").textContent = task?.workspace || "-";
  $("experimentObjects").textContent = task?.objects || "-";
  $("experimentGoal").textContent = task?.goal || "-";
}

function renderExperimentStatus(experiments) {
  const status = experiments || {};
  const badge = $("experimentSceneStatus");
  const sceneState = String(status.state || "SIM_OFF");
  badge.className = `badge ${sceneState}`;
  badge.textContent = sceneState.replace("_", " ");

  const enabled = Boolean(status.enabled);
  const pending = Boolean(status.pending);
  const hasTask = Boolean(selectedExperimentTask());
  $("experimentTaskSelect").disabled = !enabled || pending;
  $("experimentSpawnButton").disabled = !enabled || pending || !hasTask;
  $("experimentTaskResetButton").disabled = !enabled || pending || !hasTask;
  $("experimentAllResetButton").disabled = !enabled || pending || !Number(status.active_task_id || 0);
  $("experimentActiveTask").textContent = status.active_task_name || "robot only";
  $("experimentRuntimeMeta").textContent = !enabled
    ? "simulator stopped"
    : pending
      ? `applying command ${status.command_seq || 0}`
      : `command ${status.applied_seq || 0} applied`;
  renderExperimentTaskSpec();
}

function hybridPreviewImages() {
  return Array.from(document.querySelectorAll("[data-hybrid-preview]"));
}

function refreshHybridPreviewImages(force = false) {
  if (state.activePage !== "hybrid") return;
  const source = state.status?.hybrid_teleop?.source || "stopped";
  if (!force && source === "stopped") return;
  const stamp = Date.now();
  hybridPreviewImages().forEach((img, idx) => {
    const stream = img.dataset.hybridPreview;
    if (!stream) return;
    const separator = stream.includes("?") ? "&" : "?";
    setTimeout(() => {
      if (state.activePage === "hybrid") {
        img.src = `/hybrid-preview/${stream}.jpg${separator}t=${stamp}`;
      }
    }, idx * 20);
  });
}

function initHybridPreviewImages() {
  hybridPreviewImages().forEach((img) => {
    img.addEventListener("load", () => img.closest("figure")?.classList.remove("no-frame"));
    img.addEventListener("error", () => img.closest("figure")?.classList.add("no-frame"));
  });
}

function formatHybridCameraStatus(camera) {
  if (camera?.camera_ok === false && camera?.device != null) {
    return `device ${camera.device} unavailable`;
  }
  if (!camera?.fresh) return "waiting for frames";
  const detections = camera.detections || [];
  if (!detections.length) return `device ${camera.device} | no hand`;
  const labels = detections.map((item) => {
    const raw = item.raw_label || "?";
    const adjusted = item.label || "?";
    const target = item.publish_label || "?";
    const score = Number(item.score || 0).toFixed(2);
    return `${raw}->${adjusted}->${target} ${score}`;
  });
  return `device ${camera.device} | ${labels.join(", ")}`;
}

function renderHybridTeleop(hybrid) {
  const testRunning = Boolean(hybrid.test?.running);
  const reliabilityRunning = Boolean(state.status?.reliability_runtime?.running);
  const source = hybrid.source || "stopped";
  const badge = $("hybridPreviewSource");
  badge.className = `badge ${source === "stopped" ? "STOPPED" : "ALIVE"}`;
  badge.textContent = source === "reliability" ? "RELIABILITY" : source === "test" ? "TEST" : "STOPPED";

  const testButton = $("hybridTestButton");
  testButton.textContent = testRunning ? "Stop MediaPipe test" : "Start MediaPipe test";
  testButton.dataset.active = testRunning ? "true" : "false";
  testButton.disabled = reliabilityRunning;
  $("hybridApplyButton").disabled = reliabilityRunning;
  [
    "hybridLeftDeviceSelect",
    "hybridRightDeviceSelect",
    "hybridTrapezoidToggle",
    "hybridSwapLrToggle",
    "hybridMirrorToggle",
    "hybridBottomWidthRange",
    "hybridBottomWidthInput",
  ].forEach((id) => { $(id).disabled = reliabilityRunning; });

  $("hybridLeftStatus").textContent = formatHybridCameraStatus(hybrid.left);
  $("hybridRightStatus").textContent = formatHybridCameraStatus(hybrid.right);
  const runtime = hybrid.test || {};
  if (reliabilityRunning) {
    $("hybridRuntimeMeta").textContent = "camera pipeline owned by reliability runtime";
  } else if (testRunning) {
    $("hybridRuntimeMeta").textContent = `pid ${runtime.pid || "-"} | ${Number(runtime.uptime_sec || 0).toFixed(1)}s`;
  } else if (runtime.last_error) {
    $("hybridRuntimeMeta").textContent = runtime.last_error;
  } else {
    $("hybridRuntimeMeta").textContent = "stopped";
  }
}

function renderReliabilityRuntime(runtime) {
  const running = Boolean(runtime?.running);
  const status = runtime?.status || "STOPPED";
  const badge = $("reliabilityRuntimeStatus");
  if (badge) {
    badge.className = `badge ${status}`;
    badge.textContent = status;
  }

  const button = $("reliabilityRuntimeButton");
  if (button) {
    button.textContent = running ? "Stop reliability" : "Start reliability";
    button.dataset.active = running ? "true" : "false";
  }

  const handSelect = $("reliabilityHandModelSelect");
  const controllerSelect = $("reliabilityControllerModelSelect");
  if (handSelect) {
    handSelect.disabled = running;
    if (runtime?.hand_model_variant && running) handSelect.value = runtime.hand_model_variant;
  }
  if (controllerSelect) {
    controllerSelect.disabled = running;
    if (runtime?.controller_model_variant && running) controllerSelect.value = runtime.controller_model_variant;
  }

  const meta = $("reliabilityRuntimeMeta");
  if (!meta) return;
  if (running) {
    const uptime = runtime.uptime_sec === null || runtime.uptime_sec === undefined
      ? ""
      : ` | ${Number(runtime.uptime_sec).toFixed(1)}s`;
    meta.textContent = `pid ${runtime.pid || "-"} | hand ${runtime.hand_model_variant || "-"} | controller ${runtime.controller_model_variant || "-"}${uptime}`;
  } else if (runtime.last_error) {
    meta.textContent = runtime.last_error;
  } else if (runtime.last_exit_code !== null && runtime.last_exit_code !== undefined) {
    meta.textContent = `last exit ${runtime.last_exit_code}`;
  } else {
    meta.textContent = "stopped";
  }
}

function renderManualWorkers(groups) {
  const box = $("manualWorkers");
  box.innerHTML = "";
  let collectGroup = null;
  groups.forEach((group) => {
    if (group.name === "collect_data") {
      collectGroup = group;
      return;
    }
    const row = document.createElement("div");
    row.className = "manual-row";
    const name = document.createElement("div");
    name.className = "manual-name";
    name.textContent = group.label;
    const button = document.createElement("button");
    button.textContent = group.running ? "Stop" : "Start";
    button.disabled = !group.can_toggle;
    button.onclick = () => toggleManual(group.name);
    const badge = document.createElement("span");
    badge.className = `badge ${group.status}`;
    badge.textContent = group.status;
    row.append(name, button, badge);
    box.appendChild(row);
  });
  if (collectGroup) {
    $("collectDataButton").textContent = collectGroup.running ? "Stop" : "Start";
    $("collectDataButton").disabled = !collectGroup.enabled;
    $("collectDataStatus").className = `state-text badge ${collectGroup.status}`;
    $("collectDataStatus").textContent = collectGroup.status;
  }
}

async function pollStatus() {
  if (state.statusInFlight) return;
  state.statusInFlight = true;
  try {
    const status = await api("/api/status");
    renderStatus(status);
  } catch (err) {
    setMessage(err.message, true);
  } finally {
    state.statusInFlight = false;
  }
}

async function pollTelemetry() {
  if (state.activePage !== "visualizer" || state.telemetryInFlight) return;
  state.telemetryInFlight = true;
  try {
    const data = await api("/api/telemetry");
    if (state.activePage !== "visualizer") return;
    if (!data.ok) return;
    const now = data.timestamp || Date.now() / 1000;
    if (data.obs_arm && data.act_arm) {
      state.history.arm.push({t: now, obs: data.obs_arm, act: data.act_arm});
      while (state.history.arm.length > 160) state.history.arm.shift();
    }
    if (data.obs_leg && data.act_leg) {
      state.history.leg.push({t: now, obs: data.obs_leg, act: data.act_leg});
      while (state.history.leg.length > 160) state.history.leg.shift();
    }
    drawVectorGrid($("armChart"), state.history.arm, ARM_LABELS, "act_arm vs obs_arm (rad)");
    drawVectorGrid($("legChart"), state.history.leg, LEG_LABELS, "act_leg vs obs_leg (rad)");
  } catch {
    // Telemetry is optional while workers are still starting.
  } finally {
    state.telemetryInFlight = false;
  }
}

function formatHz(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${number >= 100 ? number.toFixed(0) : number.toFixed(1)} Hz`;
}

function formatMs(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  if (number >= 100) return `${number.toFixed(0)} ms`;
  if (number >= 10) return `${number.toFixed(1)} ms`;
  return `${number.toFixed(2)} ms`;
}

function formatAge(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${number.toFixed(1)} s`;
}

function runtimeLoopLabel(row) {
  const worker = String(row?.worker || row?.key || "");
  const loop = String(row?.loop || "main");
  return loop === "main" ? worker : `${worker}:${loop}`;
}

function renderRuntimeDiagnostics(diagnostics) {
  const box = $("runtimeDiagnostics");
  const summary = $("runtimeDiagnosticsSummary");
  if (!box) return;

  const rows = Array.isArray(diagnostics?.rows) ? diagnostics.rows : [];
  const liveRows = rows.filter((row) => row.status !== "STOPPED");
  if (summary) {
    const staleCount = liveRows.filter((row) => row.status === "STALE").length;
    summary.textContent = liveRows.length
      ? `${liveRows.length} loops${staleCount ? ` | ${staleCount} stale` : ""}`
      : "no data";
  }

  box.innerHTML = "";
  const header = document.createElement("div");
  header.className = "runtime-row runtime-header";
  ["Loop", "Target", "Actual", "Jitter", "Latency", "Age", "Status"].forEach((label) => {
    const cell = document.createElement("span");
    cell.textContent = label;
    header.appendChild(cell);
  });
  box.appendChild(header);

  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "runtime-empty";
    empty.textContent = "No runtime diagnostics yet.";
    box.appendChild(empty);
    return;
  }

  rows.forEach((row) => {
    const div = document.createElement("div");
    div.className = `runtime-row status-${row.status || "WAIT"}`;
    div.title = `max jitter ${formatMs(row.max_jitter_ms)} | max latency ${formatMs(row.max_latency_ms)} | samples ${row.samples || 0}`;

    const loop = document.createElement("span");
    loop.className = "runtime-loop";
    loop.textContent = runtimeLoopLabel(row);

    const target = document.createElement("span");
    target.textContent = formatHz(row.target_hz);

    const actual = document.createElement("span");
    actual.textContent = formatHz(row.actual_hz);

    const jitter = document.createElement("span");
    jitter.textContent = formatMs(row.jitter_ms);

    const latency = document.createElement("span");
    latency.textContent = formatMs(row.latency_ms);

    const age = document.createElement("span");
    age.textContent = formatAge(row.age_s);

    const status = document.createElement("span");
    status.className = `badge ${row.status || "WAIT"}`;
    status.textContent = row.status || "WAIT";

    div.append(loop, target, actual, jitter, latency, age, status);
    box.appendChild(div);
  });
}

function cleanChartNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatChartValue(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  if (Math.abs(number) >= 100) return number.toFixed(1);
  if (Math.abs(number) >= 10) return number.toFixed(2);
  return number.toFixed(4);
}

function truncateChartLabel(value, maxLength = 18) {
  const text = String(value ?? "");
  return text.length <= maxLength ? text : `${text.slice(0, Math.max(1, maxLength - 3))}...`;
}

function chartFor(container) {
  if (!container) return null;
  if (!window.echarts) {
    container.textContent = "chart library unavailable";
    return null;
  }
  let chart = window.echarts.getInstanceByDom(container);
  if (!chart) {
    chart = window.echarts.init(container, null, {renderer: "canvas"});
  }
  state.charts[container.id] = chart;
  return chart;
}

function resizeCharts() {
  Object.values(state.charts).forEach((chart) => {
    try {
      chart.resize();
    } catch {
      // Chart may have been disposed by a browser reload path.
    }
  });
}

function setChartMessage(container, title, message) {
  const chart = chartFor(container);
  if (!chart) return;
  chart.clear();
  chart.setOption({
    animation: false,
    backgroundColor: "#ffffff",
    title: {
      text: title,
      left: 12,
      top: 6,
      textStyle: {color: "#111827", fontSize: 13, fontWeight: 600},
    },
    graphic: {
      type: "text",
      left: 14,
      top: 42,
      style: {text: message, fill: "#64748b", font: "12px sans-serif"},
    },
  }, true);
}

function drawVectorGrid(container, samples, labels, title) {
  const chart = chartFor(container);
  if (!chart) return;
  chart.resize();
  if (!samples.length) {
    setChartMessage(container, title, "waiting for telemetry");
    return;
  }

  const rect = container.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width || container.clientWidth || 320));
  const height = Math.max(280, Math.floor(rect.height || container.clientHeight || 280));
  const minCellWidth = labels.length > 12 ? 112 : 118;
  const cols = Math.max(2, Math.min(labels.length, Math.floor(width / minCellWidth)));
  const rows = Math.ceil(labels.length / cols);
  const top = 38;
  const gap = 9;
  const cellW = Math.max(80, (width - gap * (cols + 1)) / cols);
  const cellH = Math.max(54, (height - top - gap * (rows + 1)) / rows);
  const grids = [];
  const xAxis = [];
  const yAxis = [];
  const series = [];
  const titles = [{
    text: title,
    left: 12,
    top: 6,
    textStyle: {color: "#111827", fontSize: 13, fontWeight: 600},
  }];
  const maxSample = Math.max(1, samples.length - 1);

  labels.forEach((label, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const x = gap + col * (cellW + gap);
    const y = top + gap + row * (cellH + gap);
    const plotX = x + 6;
    const plotY = y + 20;
    const plotW = Math.max(36, cellW - 12);
    const plotH = Math.max(28, cellH - 28);

    titles.push({
      text: truncateChartLabel(label, Math.max(8, Math.floor(plotW / 7))),
      left: x + 5,
      top: y + 2,
      textStyle: {color: "#1f2937", fontSize: 10, fontWeight: 500},
    });
    grids.push({left: plotX, top: plotY, width: plotW, height: plotH, containLabel: false});
    xAxis.push({
      type: "value",
      gridIndex: idx,
      min: 0,
      max: maxSample,
      axisLabel: {show: false},
      axisTick: {show: false},
      axisLine: {lineStyle: {color: "#cbd5e1"}},
      splitLine: {show: false},
    });
    yAxis.push({
      type: "value",
      gridIndex: idx,
      scale: true,
      axisLabel: {show: false},
      axisTick: {show: false},
      axisLine: {lineStyle: {color: "#cbd5e1"}},
      splitLine: {lineStyle: {color: "#eef2f7", width: 1}},
    });
    series.push({
      name: "act",
      type: "line",
      xAxisIndex: idx,
      yAxisIndex: idx,
      data: samples.map((sample, sampleIdx) => [sampleIdx, cleanChartNumber(sample.act?.[idx])]),
      showSymbol: false,
      connectNulls: false,
      animation: false,
      lineStyle: {color: TELEMETRY_ACT_COLOR, width: 1.4},
      itemStyle: {color: TELEMETRY_ACT_COLOR},
    });
    series.push({
      name: "obs",
      type: "line",
      xAxisIndex: idx,
      yAxisIndex: idx,
      data: samples.map((sample, sampleIdx) => [sampleIdx, cleanChartNumber(sample.obs?.[idx])]),
      showSymbol: false,
      connectNulls: false,
      animation: false,
      lineStyle: {color: TELEMETRY_OBS_COLOR, width: 1.4},
      itemStyle: {color: TELEMETRY_OBS_COLOR},
    });
  });

  chart.setOption({
    animation: false,
    backgroundColor: "#ffffff",
    color: [TELEMETRY_ACT_COLOR, TELEMETRY_OBS_COLOR],
    title: titles,
    legend: {
      data: ["act", "obs"],
      right: 12,
      top: 4,
      itemWidth: 16,
      itemHeight: 8,
      textStyle: {color: "#475569", fontSize: 11},
    },
    tooltip: {
      trigger: "axis",
      confine: true,
      formatter: telemetryTooltipFormatter,
    },
    grid: grids,
    xAxis,
    yAxis,
    series,
  }, true, true);
}

function telemetryTooltipFormatter(params) {
  const items = Array.isArray(params) ? params : [params];
  const sample = items[0]?.data?.[0] ?? items[0]?.axisValue ?? "";
  const lines = [`sample ${sample}`];
  items.forEach((item) => {
    const value = Array.isArray(item.data) ? item.data[1] : item.value;
    if (Number.isFinite(Number(value))) {
      lines.push(`${item.marker}${item.seriesName}: ${formatChartValue(value)}`);
    }
  });
  return lines.join("<br/>");
}

function datasetParams(extra = {}) {
  const params = new URLSearchParams();
  const root = $("datasetRootInput")?.value || "";
  if (root) params.set("root", root);
  Object.entries(extra).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "") params.set(key, value);
  });
  return params;
}

async function loadDatasetListOnce() {
  if (state.datasetViewer.ready || state.datasetViewer.loadInFlight) return;
  await loadDatasetList();
}

async function loadDatasetList() {
  const viewer = state.datasetViewer;
  viewer.loadInFlight = true;
  setMessage("loading datasets");
  try {
    const data = await api(`/api/dataset-viewer/datasets?${datasetParams().toString()}`);
    viewer.ready = true;
    viewer.datasets = data.datasets || [];
    if ($("datasetRootInput") && data.root) $("datasetRootInput").value = data.root;
    renderDatasetList();
    const suffix = data.truncated ? " (truncated)" : "";
    setMessage(`datasets: ${viewer.datasets.length}${suffix}`);
  } catch (err) {
    setMessage(err.message, true);
    renderDatasetList(err.message);
  } finally {
    viewer.loadInFlight = false;
  }
}

function renderDatasetList(error = "") {
  const box = $("datasetList");
  if (!box) return;
  box.innerHTML = "";
  if (error) {
    const row = document.createElement("div");
    row.className = "dataset-row";
    row.textContent = error;
    box.appendChild(row);
    return;
  }
  const datasets = state.datasetViewer.datasets || [];
  if (!datasets.length) {
    const row = document.createElement("div");
    row.className = "dataset-row";
    row.innerHTML = "<span class='dataset-row-title'>No datasets found</span><span class='dataset-row-meta'>Check the root path and refresh.</span>";
    box.appendChild(row);
    return;
  }
  datasets.forEach((dataset) => {
    const row = document.createElement("div");
    row.className = `dataset-row${dataset.path === state.datasetViewer.selectedDataset?.path ? " selected" : ""}`;
    const title = document.createElement("div");
    title.className = "dataset-row-title";
    title.textContent = dataset.display || dataset.name || dataset.path;
    const meta = document.createElement("div");
    meta.className = "dataset-row-meta";
    meta.textContent = `${dataset.total_episodes || 0} eps | ${dataset.total_frames || 0} frames | ${Number(dataset.fps || 0).toFixed(1)} fps`;
    row.append(title, meta);
    row.onclick = () => selectDataset(dataset);
    box.appendChild(row);
  });
}

async function selectDataset(dataset) {
  state.datasetViewer.selectedDataset = dataset;
  state.datasetViewer.data = null;
  state.datasetViewer.frameIdx = 0;
  state.datasetViewer.selectedSegment = -1;
  state.datasetViewer.segmentDirty = false;
  state.datasetViewer.classDirty = false;
  setDatasetPlaying(false);
  renderDatasetList();
  fillDatasetEpisodes(dataset.total_episodes || 0);
  await loadDatasetEpisode(0);
}

function fillDatasetEpisodes(totalEpisodes) {
  const select = $("datasetEpisodeSelect");
  if (!select) return;
  select.innerHTML = "";
  const count = Math.max(0, Number(totalEpisodes || 0));
  if (!count) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No episodes";
    select.appendChild(option);
    return;
  }
  for (let idx = 0; idx < count; idx += 1) {
    const option = document.createElement("option");
    option.value = String(idx);
    option.textContent = `Episode ${String(idx).padStart(3, "0")}`;
    select.appendChild(option);
  }
  select.value = "0";
}

function fillDatasetImageStreams(data) {
  const select = $("datasetImageStreamSelect");
  if (!select) return;
  const previous = data?.image_key || select.value || "";
  select.innerHTML = "";
  const streams = data?.dataset?.image_streams || [];
  if (!streams.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No image";
    select.appendChild(option);
    return;
  }
  streams.forEach((stream) => {
    const option = document.createElement("option");
    option.value = stream.key;
    option.textContent = stream.name || stream.key;
    select.appendChild(option);
  });
  select.value = streams.some((stream) => stream.key === previous) ? previous : streams[0].key;
}

function datasetLabelName(classId) {
  const data = state.datasetViewer.data;
  if (Number(classId) < 0) return "unassigned";
  const item = (data?.classes || []).find((entry) => Number(entry.id) === Number(classId));
  return item?.name || `class_${classId}`;
}

function datasetSegmentIndexForFrame(frameNumber) {
  const data = state.datasetViewer.data;
  if (!data) return -1;
  return (data.segments || []).findIndex((segment) => (
    Number(segment.start_frame) <= Number(frameNumber)
    && Number(frameNumber) <= Number(segment.end_frame)
  ));
}

function datasetFrameArrayIndex(frameNumber) {
  const frames = state.datasetViewer.data?.frame_numbers_full || [];
  const target = Number(frameNumber);
  const exact = frames.findIndex((frame) => Number(frame) === target);
  if (exact >= 0) return exact;
  for (let idx = 0; idx < frames.length; idx += 1) {
    if (Number(frames[idx]) >= target) return idx;
  }
  return Math.max(0, frames.length - 1);
}

function normalizeDatasetSegmentObjects() {
  const data = state.datasetViewer.data;
  if (!data) return;
  data.classes = Array.isArray(data.classes) ? data.classes : [];
  data.segments = Array.isArray(data.segments) ? data.segments : [];
  data.segments.sort((a, b) => Number(a.start_frame) - Number(b.start_frame));
  data.segments.forEach((segment) => {
    segment.start_frame = Number(segment.start_frame || 0);
    segment.end_frame = Number(segment.end_frame || segment.start_frame || 0);
    segment.class_id = Number(segment.class_id ?? -1);
    segment.class_name = datasetLabelName(segment.class_id);
  });
}

function recomputeDatasetLabelSeries() {
  const data = state.datasetViewer.data;
  if (!data) return;
  normalizeDatasetSegmentObjects();
  const frames = data.frame_numbers_full || [];
  const labelIds = [];
  const labelNames = [];
  let segmentIdx = 0;
  frames.forEach((frame) => {
    const frameNumber = Number(frame);
    while (
      segmentIdx < data.segments.length - 1
      && frameNumber > Number(data.segments[segmentIdx].end_frame)
    ) {
      segmentIdx += 1;
    }
    const segment = data.segments[segmentIdx];
    const classId = segment
      && Number(segment.start_frame) <= frameNumber
      && frameNumber <= Number(segment.end_frame)
        ? Number(segment.class_id)
        : -1;
    labelIds.push(classId);
    labelNames.push(datasetLabelName(classId));
  });
  data.label_ids_full = labelIds;
  data.label_names_full = labelNames;
}

function datasetHasUnassignedSegments() {
  const data = state.datasetViewer.data;
  return Boolean((data?.segments || []).some((segment) => Number(segment.class_id) < 0));
}

function markDatasetLabelsDirty(message = "Unsaved label changes.") {
  state.datasetViewer.segmentDirty = true;
  recomputeDatasetLabelSeries();
  renderDatasetLabelControls(message);
  updateDatasetInfoText();
  redrawDatasetViewer();
}

function initDatasetLabelState(data) {
  state.datasetViewer.selectedSegment = -1;
  state.datasetViewer.segmentDirty = false;
  state.datasetViewer.classDirty = false;
  state.datasetViewer.savingLabels = false;
  if (!data) return;
  normalizeDatasetSegmentObjects();
  recomputeDatasetLabelSeries();
}

function setDatasetSelectedSegment(index, {jumpToSegment = false} = {}) {
  const data = state.datasetViewer.data;
  if (!data) return;
  const segments = data.segments || [];
  const next = Math.max(-1, Math.min(segments.length - 1, Number(index)));
  state.datasetViewer.selectedSegment = next;
  if (jumpToSegment && next >= 0) {
    updateDatasetFrame(datasetFrameArrayIndex(segments[next].start_frame));
    return;
  }
  renderDatasetLabelControls();
}

function renderDatasetLabelControls(message = "") {
  const viewer = state.datasetViewer;
  const data = viewer.data;
  const segmentList = $("datasetSegmentList");
  const classSelect = $("datasetSegmentClassSelect");
  const manageSelect = $("datasetClassManageSelect");
  const nameInput = $("datasetClassNameInput");
  const status = $("datasetLabelStatus");
  const splitButton = $("datasetSplitSegmentButton");
  const mergePrevButton = $("datasetMergePrevButton");
  const mergeNextButton = $("datasetMergeNextButton");
  const resetButton = $("datasetResetLabelsButton");
  const saveButton = $("datasetSaveLabelsButton");
  const addButton = $("datasetAddClassButton");
  const renameButton = $("datasetRenameClassButton");

  if (!data) {
    if (segmentList) segmentList.innerHTML = "<div class='dataset-segment-row'>No episode loaded.</div>";
    [classSelect, manageSelect, nameInput, splitButton, mergePrevButton, mergeNextButton, resetButton, saveButton, addButton, renameButton]
      .forEach((el) => { if (el) el.disabled = true; });
    if (status) status.textContent = "No labels loaded.";
    return;
  }

  normalizeDatasetSegmentObjects();
  const segments = data.segments || [];
  const classes = data.classes || [];
  const frameNumber = (data.frame_numbers_full || [])[viewer.frameIdx || 0] ?? 0;
  const currentSegmentIdx = datasetSegmentIndexForFrame(frameNumber);
  if (currentSegmentIdx !== viewer.selectedSegment) {
    viewer.selectedSegment = currentSegmentIdx;
  }
  const selected = segments[viewer.selectedSegment] || null;

  if (segmentList) {
    segmentList.innerHTML = "";
    if (!segments.length) {
      const empty = document.createElement("div");
      empty.className = "dataset-segment-row";
      empty.textContent = "No segments.";
      segmentList.appendChild(empty);
    } else {
      segments.forEach((segment, idx) => {
        const row = document.createElement("button");
        row.type = "button";
        row.className = `dataset-segment-row${idx === viewer.selectedSegment ? " selected" : ""}`;
        const number = document.createElement("span");
        number.textContent = String(idx).padStart(2, "0");
        const range = document.createElement("span");
        range.className = "range";
        range.textContent = `${segment.start_frame} - ${segment.end_frame}`;
        const label = document.createElement("span");
        label.className = "label";
        label.textContent = segment.class_name;
        row.append(number, range, label);
        row.addEventListener("click", () => setDatasetSelectedSegment(idx, {jumpToSegment: true}));
        segmentList.appendChild(row);
      });
    }
  }

  if (classSelect) {
    classSelect.innerHTML = "";
    const unassigned = document.createElement("option");
    unassigned.value = "-1";
    unassigned.textContent = "unassigned";
    classSelect.appendChild(unassigned);
    classes.forEach((entry) => {
      const option = document.createElement("option");
      option.value = String(entry.id);
      option.textContent = entry.name;
      classSelect.appendChild(option);
    });
    classSelect.value = selected ? String(selected.class_id) : "-1";
    classSelect.disabled = !selected;
  }

  if (manageSelect) {
    const previous = manageSelect.value;
    manageSelect.innerHTML = "";
    classes.forEach((entry) => {
      const option = document.createElement("option");
      option.value = String(entry.id);
      option.textContent = entry.name;
      manageSelect.appendChild(option);
    });
    if (classes.some((entry) => String(entry.id) === previous)) {
      manageSelect.value = previous;
    }
    manageSelect.disabled = classes.length <= 0;
  }
  if (nameInput && manageSelect) {
    const managed = classes.find((entry) => String(entry.id) === String(manageSelect.value));
    if (document.activeElement !== nameInput) nameInput.value = managed?.name || "";
    nameInput.disabled = false;
  }

  const currentSegment = currentSegmentIdx >= 0 ? segments[currentSegmentIdx] : null;
  const prevFrameNumber = (data.frame_numbers_full || [])[Math.max(0, (viewer.frameIdx || 0) - 1)] ?? frameNumber;
  const canSplit = Boolean(
    currentSegment
    && (viewer.frameIdx || 0) > 0
    && Number(prevFrameNumber) >= Number(currentSegment.start_frame)
    && Number(frameNumber) > Number(currentSegment.start_frame)
  );
  if (splitButton) splitButton.disabled = !canSplit;
  if (mergePrevButton) mergePrevButton.disabled = !(selected && viewer.selectedSegment > 0);
  if (mergeNextButton) mergeNextButton.disabled = !(selected && viewer.selectedSegment >= 0 && viewer.selectedSegment < segments.length - 1);
  if (resetButton) resetButton.disabled = !segments.length;
  if (addButton) addButton.disabled = !nameInput;
  if (renameButton) renameButton.disabled = !(manageSelect && manageSelect.value !== "");

  const dirty = viewer.segmentDirty || viewer.classDirty;
  const hasUnassigned = datasetHasUnassignedSegments();
  if (saveButton) {
    saveButton.disabled = !dirty || hasUnassigned || viewer.savingLabels;
    saveButton.textContent = viewer.savingLabels ? "Saving..." : "Save Labels";
  }
  if (status) {
    if (message) status.textContent = message;
    else if (viewer.savingLabels) status.textContent = "Saving labels.";
    else if (hasUnassigned && dirty) status.textContent = "Assign every dirty segment.";
    else if (dirty) status.textContent = "Unsaved label changes.";
    else status.textContent = `Current frame label: ${(data.label_names_full || [])[viewer.frameIdx || 0] || "unassigned"}`;
  }
}

function assignDatasetSegmentClass(classId) {
  const data = state.datasetViewer.data;
  const row = state.datasetViewer.selectedSegment;
  if (!data || row < 0 || row >= (data.segments || []).length) return;
  const nextClassId = Number(classId);
  if (Number(data.segments[row].class_id) === nextClassId) {
    renderDatasetLabelControls();
    return;
  }
  data.segments[row].class_id = nextClassId;
  data.segments[row].class_name = datasetLabelName(nextClassId);
  markDatasetLabelsDirty();
}

function splitDatasetSegmentAtFrame() {
  const data = state.datasetViewer.data;
  if (!data) return;
  const frameIdx = state.datasetViewer.frameIdx || 0;
  const frameNumber = (data.frame_numbers_full || [])[frameIdx];
  const prevFrameNumber = (data.frame_numbers_full || [])[frameIdx - 1];
  const segmentIdx = datasetSegmentIndexForFrame(frameNumber);
  if (segmentIdx < 0 || frameIdx <= 0) return;
  const segment = data.segments[segmentIdx];
  if (Number(prevFrameNumber) < Number(segment.start_frame) || Number(frameNumber) <= Number(segment.start_frame)) return;
  const left = {
    start_frame: Number(segment.start_frame),
    end_frame: Number(prevFrameNumber),
    class_id: Number(segment.class_id),
    class_name: datasetLabelName(segment.class_id),
  };
  const right = {
    start_frame: Number(frameNumber),
    end_frame: Number(segment.end_frame),
    class_id: Number(segment.class_id),
    class_name: datasetLabelName(segment.class_id),
  };
  data.segments.splice(segmentIdx, 1, left, right);
  state.datasetViewer.selectedSegment = segmentIdx + 1;
  markDatasetLabelsDirty();
}

function mergeDatasetSegment(direction) {
  const data = state.datasetViewer.data;
  if (!data) return;
  const row = state.datasetViewer.selectedSegment;
  const segments = data.segments || [];
  if (direction === "prev") {
    if (row <= 0 || row >= segments.length) return;
    const merged = {
      start_frame: Number(segments[row - 1].start_frame),
      end_frame: Number(segments[row].end_frame),
      class_id: Number(segments[row].class_id),
      class_name: datasetLabelName(segments[row].class_id),
    };
    segments.splice(row - 1, 2, merged);
    state.datasetViewer.selectedSegment = row - 1;
  } else {
    if (row < 0 || row >= segments.length - 1) return;
    const merged = {
      start_frame: Number(segments[row].start_frame),
      end_frame: Number(segments[row + 1].end_frame),
      class_id: Number(segments[row].class_id),
      class_name: datasetLabelName(segments[row].class_id),
    };
    segments.splice(row, 2, merged);
    state.datasetViewer.selectedSegment = row;
  }
  markDatasetLabelsDirty();
}

function resetDatasetEpisodeLabels() {
  const data = state.datasetViewer.data;
  const frames = data?.frame_numbers_full || [];
  if (!data || !frames.length) return;
  data.segments = [{
    start_frame: Number(frames[0]),
    end_frame: Number(frames[frames.length - 1]),
    class_id: -1,
    class_name: "unassigned",
  }];
  state.datasetViewer.selectedSegment = 0;
  markDatasetLabelsDirty("Episode labels reset.");
}

function addDatasetLabelClass() {
  const data = state.datasetViewer.data;
  const input = $("datasetClassNameInput");
  const name = String(input?.value || "").trim();
  if (!data || !name) {
    renderDatasetLabelControls("Enter a class name.");
    return;
  }
  const existing = new Set((data.classes || []).map((entry) => String(entry.name).trim().toLowerCase()));
  if (name.toLowerCase() === "unassigned" || existing.has(name.toLowerCase())) {
    renderDatasetLabelControls(`Class already exists: ${name}`);
    return;
  }
  const nextId = Math.max(-1, ...(data.classes || []).map((entry) => Number(entry.id))) + 1;
  data.classes.push({id: nextId, name});
  data.classes.sort((a, b) => Number(a.id) - Number(b.id));
  state.datasetViewer.classDirty = true;
  recomputeDatasetLabelSeries();
  const manageSelect = $("datasetClassManageSelect");
  if (manageSelect) manageSelect.value = String(nextId);
  renderDatasetLabelControls(`Added class: ${name}`);
  updateDatasetInfoText();
  redrawDatasetViewer();
}

function renameDatasetLabelClass() {
  const data = state.datasetViewer.data;
  const manageSelect = $("datasetClassManageSelect");
  const input = $("datasetClassNameInput");
  const classId = Number(manageSelect?.value);
  const newName = String(input?.value || "").trim();
  if (!data || !Number.isFinite(classId)) {
    renderDatasetLabelControls("Select a class to rename.");
    return;
  }
  if (!newName) {
    renderDatasetLabelControls("Enter a class name.");
    return;
  }
  if (newName.toLowerCase() === "unassigned") {
    renderDatasetLabelControls("Reserved class name: unassigned");
    return;
  }
  const duplicate = (data.classes || []).find((entry) => (
    Number(entry.id) !== classId
    && String(entry.name).trim().toLowerCase() === newName.toLowerCase()
  ));
  if (duplicate) {
    renderDatasetLabelControls(`Class already exists: ${newName}`);
    return;
  }
  const target = (data.classes || []).find((entry) => Number(entry.id) === classId);
  if (!target || target.name === newName) {
    renderDatasetLabelControls();
    return;
  }
  target.name = newName;
  state.datasetViewer.classDirty = true;
  recomputeDatasetLabelSeries();
  renderDatasetLabelControls(`Renamed class: ${newName}`);
  updateDatasetInfoText();
  redrawDatasetViewer();
}

async function saveDatasetLabels() {
  const viewer = state.datasetViewer;
  const data = viewer.data;
  if (!data || viewer.savingLabels) return;
  if (datasetHasUnassignedSegments()) {
    renderDatasetLabelControls("Assign every segment before saving.");
    return;
  }
  viewer.savingLabels = true;
  renderDatasetLabelControls();
  const frameIdx = viewer.frameIdx || 0;
  try {
    const payload = await api("/api/dataset-viewer/labels", {
      root: data.root || null,
      dataset: data.dataset?.path || "",
      episode: data.episode || 0,
      classes: data.classes || [],
      segments: (data.segments || []).map((segment) => ({
        start_frame: Number(segment.start_frame),
        end_frame: Number(segment.end_frame),
        class_id: Number(segment.class_id),
      })),
    });
    if (Array.isArray(payload.classes)) data.classes = payload.classes;
    if (Array.isArray(payload.segments)) data.segments = payload.segments;
    if (Array.isArray(payload.label_ids_full)) data.label_ids_full = payload.label_ids_full;
    if (Array.isArray(payload.label_names_full)) data.label_names_full = payload.label_names_full;
    viewer.segmentDirty = false;
    viewer.classDirty = false;
    setMessage(`saved labels for episode ${String(data.episode || 0).padStart(3, "0")}`);
    recomputeDatasetLabelSeries();
    updateDatasetFrame(frameIdx, {updateImage: false});
  } catch (err) {
    setMessage(err.message, true);
    renderDatasetLabelControls(err.message);
  } finally {
    viewer.savingLabels = false;
    renderDatasetLabelControls();
  }
}

async function loadDatasetEpisode(episode = null) {
  const viewer = state.datasetViewer;
  const selected = viewer.selectedDataset;
  if (!selected || viewer.loadInFlight) return;
  viewer.loadInFlight = true;
  setDatasetPlaying(false);
  setMessage("loading episode");
  try {
    const episodeValue = episode === null
      ? Number($("datasetEpisodeSelect").value || 0)
      : Number(episode || 0);
    const params = datasetParams({
      dataset: selected.path,
      episode: episodeValue,
      image_key: $("datasetImageStreamSelect")?.value || null,
      max_points: 900,
    });
    const data = await api(`/api/dataset-viewer/episode?${params.toString()}`);
    viewer.data = data;
    viewer.frameIdx = 0;
    initDatasetLabelState(data);
    if ($("datasetEpisodeSelect")) $("datasetEpisodeSelect").value = String(data.episode || 0);
    fillDatasetImageStreams(data);
    renderDatasetViewerData();
    setMessage(`loaded dataset episode ${String(data.episode || 0).padStart(3, "0")}`);
  } catch (err) {
    viewer.data = null;
    initDatasetLabelState(null);
    setMessage(err.message, true);
    renderDatasetViewerData(err.message);
  } finally {
    viewer.loadInFlight = false;
  }
}

function renderDatasetViewerData(error = "") {
  const data = state.datasetViewer.data;
  const summary = $("datasetViewerSummary");
  const info = $("datasetInfoText");
  const frameLabel = $("datasetFrameLabel");
  const slider = $("datasetFrameSlider");
  if (error) {
    if (summary) summary.textContent = error;
    if (info) info.textContent = error;
    if (frameLabel) frameLabel.textContent = "0 / 0";
    if (slider) {
      slider.max = "0";
      slider.value = "0";
    }
    clearDatasetImage();
    renderDatasetLabelControls();
    redrawDatasetViewer();
    return;
  }
  if (!data) {
    if (summary) summary.textContent = "Select a dataset.";
    if (info) info.textContent = "No dataset loaded.";
    clearDatasetImage();
    renderDatasetLabelControls();
    redrawDatasetViewer();
    return;
  }
  const frameCount = Number(data.frame_count || 0);
  if (slider) {
    slider.max = String(Math.max(0, frameCount - 1));
    slider.value = "0";
    slider.disabled = frameCount <= 0;
  }
  if ($("datasetPlayButton")) $("datasetPlayButton").disabled = frameCount <= 1;
  resetDatasetImageLoader(true);
  updateDatasetFrame(0, {updateImage: true});
  renderDatasetLabelControls();
  redrawDatasetViewer();
}

function clearDatasetImage() {
  resetDatasetImageLoader(true);
  const img = $("datasetFrameImage");
  if (img) img.removeAttribute("src");
  const caption = $("datasetFrameCaption");
  if (caption) caption.textContent = "frame";
}

function resetDatasetImageLoader(clearElement = false) {
  const viewer = state.datasetViewer;
  viewer.imageLoading = false;
  viewer.pendingImageFrame = null;
  viewer.loadedImageFrame = null;
  viewer.imageRequestToken += 1;
  if (clearElement) {
    const img = $("datasetFrameImage");
    if (img) {
      img.onload = null;
      img.onerror = null;
      img.removeAttribute("src");
    }
  }
}

function datasetFrameUrl(frameIdx) {
  const data = state.datasetViewer.data;
  if (!data?.dataset?.path || !data.image_key) return "";
  const params = datasetParams({
    dataset: data.dataset.path,
    episode: data.episode || 0,
    frame: frameIdx,
    image_key: data.image_key,
    max_width: 1100,
    t: Date.now(),
  });
  return `/api/dataset-viewer/frame.jpg?${params.toString()}`;
}

function queueDatasetFrameImage(frameIdx) {
  const viewer = state.datasetViewer;
  const data = viewer.data;
  if (!data) return;
  const idx = Math.max(0, Math.min(Number(data.frame_count || 1) - 1, Number(frameIdx || 0)));
  if (!viewer.imageLoading && viewer.loadedImageFrame === idx) return;
  viewer.pendingImageFrame = idx;
  if (!viewer.imageLoading) loadPendingDatasetFrameImage();
}

function loadPendingDatasetFrameImage() {
  const viewer = state.datasetViewer;
  const data = viewer.data;
  const img = $("datasetFrameImage");
  if (!data || !img || viewer.pendingImageFrame === null) return;

  const frameIdx = viewer.pendingImageFrame;
  const url = datasetFrameUrl(frameIdx);
  if (!url) return;

  viewer.pendingImageFrame = null;
  viewer.imageLoading = true;
  const token = viewer.imageRequestToken + 1;
  viewer.imageRequestToken = token;

  img.onload = () => {
    if (viewer.imageRequestToken !== token) return;
    viewer.imageLoading = false;
    viewer.loadedImageFrame = frameIdx;
    if (viewer.pendingImageFrame !== null && viewer.pendingImageFrame !== frameIdx) {
      loadPendingDatasetFrameImage();
    }
  };
  img.onerror = () => {
    if (viewer.imageRequestToken !== token) return;
    viewer.imageLoading = false;
    viewer.loadedImageFrame = null;
    if (viewer.pendingImageFrame !== null) {
      loadPendingDatasetFrameImage();
    } else {
      setMessage("dataset frame image failed to load", true);
    }
  };
  img.src = url;
}

function updateDatasetFrame(frameIdx, {updateImage = true} = {}) {
  const data = state.datasetViewer.data;
  if (!data) return;
  const frameCount = Math.max(0, Number(data.frame_count || 0));
  const idx = Math.max(0, Math.min(frameCount - 1, Number(frameIdx || 0)));
  state.datasetViewer.frameIdx = idx;

  const slider = $("datasetFrameSlider");
  if (slider && Number(slider.value) !== idx) slider.value = String(idx);
  const frameLabel = $("datasetFrameLabel");
  if (frameLabel) frameLabel.textContent = frameCount ? `${idx + 1} / ${frameCount}` : "0 / 0";

  const fullFrames = data.frame_numbers_full || [];
  const timestamps = data.timestamps_full || [];
  const frameNumber = fullFrames[idx] ?? idx;
  const timestamp = timestamps[idx] ?? 0;
  const label = (data.label_names_full || [])[idx] || "-";
  const imageName = (data.dataset?.image_streams || []).find((stream) => stream.key === data.image_key)?.name || data.image_key || "-";
  const summary = $("datasetViewerSummary");
  if (summary) {
    summary.textContent = `${data.dataset.display || data.dataset.name} | episode=${String(data.episode).padStart(3, "0")} | frame=${frameNumber} | t=${Number(timestamp || 0).toFixed(3)}s | label=${label} | image=${imageName}`;
  }
  const caption = $("datasetFrameCaption");
  if (caption) caption.textContent = `${imageName} | frame ${frameNumber}`;

  if (updateImage) {
    queueDatasetFrameImage(idx);
  }
  updateDatasetInfoText();
  renderDatasetLabelControls();
  redrawDatasetViewer();
}

function updateDatasetInfoText() {
  const data = state.datasetViewer.data;
  const info = $("datasetInfoText");
  if (!info || !data) return;
  const idx = state.datasetViewer.frameIdx || 0;
  const frameNumber = (data.frame_numbers_full || [])[idx] ?? idx;
  const timestamp = (data.timestamps_full || [])[idx] ?? 0;
  const label = (data.label_names_full || [])[idx] || "-";
  const segment = (data.segments || []).find((item) => frameNumber >= item.start_frame && frameNumber <= item.end_frame);
  const lines = [
    `dataset: ${data.dataset.display || data.dataset.name || "-"}`,
    `root: ${data.root || "-"}`,
    `episode: ${data.episode}`,
    `frame: ${idx + 1}/${data.frame_count || 0}`,
    `frame_index: ${frameNumber}`,
    `timestamp: ${Number(timestamp || 0).toFixed(3)}s`,
    `segment_label: ${label}`,
    `segment_range: ${segment ? `${segment.start_frame} - ${segment.end_frame}` : "-"}`,
    `image_stream: ${data.image_key || "-"}`,
    `observation_dim: ${data.dimensions?.observation || 0}`,
    `action_dim: ${data.dimensions?.action || 0}`,
    `ee_pose_dim: ${data.dimensions?.ee_pose || 0}`,
    "",
    `total_episodes: ${data.dataset.total_episodes || 0}`,
    `total_frames: ${data.dataset.total_frames || 0}`,
    `dataset_fps: ${Number(data.dataset.fps || 0).toFixed(1)}`,
    `chunks_size: ${data.dataset.chunks_size || 0}`,
    "",
    "image_streams:",
    ...(data.dataset.image_streams || []).map((stream) => `${stream.key === data.image_key ? "*" : "-"} ${stream.name || stream.key}`),
    "",
    "segments:",
    ...(data.segments || []).slice(0, 80).map((segmentItem, i) => `${String(i).padStart(2, "0")} | ${segmentItem.start_frame} - ${segmentItem.end_frame} | ${segmentItem.class_name}`),
  ];
  info.textContent = lines.join("\n");
}

function setDatasetPlaying(playing) {
  const viewer = state.datasetViewer;
  viewer.playing = Boolean(playing && viewer.data && Number(viewer.data.frame_count || 0) > 1);
  if (viewer.playTimer) {
    clearInterval(viewer.playTimer);
    viewer.playTimer = null;
  }
  const button = $("datasetPlayButton");
  if (button) button.textContent = viewer.playing ? "Pause" : "Play";
  if (!viewer.playing) return;

  const fps = Math.max(1, Math.min(12, Number(viewer.data?.dataset?.fps || 30)));
  viewer.playTimer = setInterval(() => {
    const data = viewer.data;
    if (!data) {
      setDatasetPlaying(false);
      return;
    }
    const next = (viewer.frameIdx || 0) + 1;
    if (next >= Number(data.frame_count || 0)) {
      updateDatasetFrame(Number(data.frame_count || 1) - 1);
      setDatasetPlaying(false);
      return;
    }
    updateDatasetFrame(next);
  }, Math.max(16, Math.round(1000 / fps)));
}

function redrawDatasetViewer() {
  const data = state.datasetViewer.data;
  const plotMap = new Map((data?.plots || []).map((plot) => [plot.name, plot]));
  drawDatasetPlot($("datasetPlotHand"), plotMap.get("hand"), data, "Hand");
  drawDatasetPlot($("datasetPlotArm"), plotMap.get("arm"), data, "Arm");
  drawDatasetPlot($("datasetPlotNeck"), plotMap.get("neck"), data, "Neck");
  drawDatasetPlot($("datasetPlotWaist"), plotMap.get("waist"), data, "Waist");
}

function drawDatasetPlot(container, plot, data, title) {
  const chart = chartFor(container);
  if (!chart) return;
  chart.resize();

  const xValues = data?.frame_numbers || [];
  const labels = plot?.labels || [];
  const obsRows = plot?.obs || [];
  const actRows = plot?.act || [];
  const dimCount = Math.max(labels.length, obsRows[0]?.length || 0, actRows[0]?.length || 0);
  if (!data || !plot || !xValues.length || dimCount <= 0) {
    setChartMessage(container, title, "n/a");
    return;
  }

  let xMin = Number(xValues[0]);
  let xMax = Number(xValues[xValues.length - 1]);
  if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) {
    setChartMessage(container, title, "n/a");
    return;
  }
  if (xMin === xMax) {
    xMin -= 1;
    xMax += 1;
  }
  const values = [];
  for (let rowIdx = 0; rowIdx < xValues.length; rowIdx += 1) {
    for (let dim = 0; dim < dimCount; dim += 1) {
      const obs = Number(obsRows[rowIdx]?.[dim]);
      const act = Number(actRows[rowIdx]?.[dim]);
      if (Number.isFinite(obs)) values.push(obs);
      if (Number.isFinite(act)) values.push(act);
    }
  }
  let yMin = Math.min(...values);
  let yMax = Math.max(...values);
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
    yMin = -0.05; yMax = 0.05;
  }
  if (yMin === yMax) {
    yMin -= 0.05; yMax += 0.05;
  } else {
    const margin = (yMax - yMin) * 0.1;
    yMin -= margin; yMax += margin;
  }

  const markAreaData = datasetSegmentMarkAreas(data.segments || []);
  const currentFrame = (data.frame_numbers_full || [])[state.datasetViewer.frameIdx || 0] ?? Number(xValues[0]);
  const series = [];
  for (let dim = 0; dim < dimCount; dim += 1) {
    const label = labels[dim] || `dim_${dim}`;
    const color = CHART_COLORS[dim % CHART_COLORS.length];
    const obsSeries = {
      name: `${label} obs`,
      type: "line",
      data: xValues.map((x, rowIdx) => [Number(x), cleanChartNumber(obsRows[rowIdx]?.[dim])]),
      showSymbol: false,
      connectNulls: false,
      animation: false,
      sampling: "lttb",
      lineStyle: {color, width: 1.35},
      itemStyle: {color},
    };
    if (dim === 0) {
      obsSeries.markLine = {
        silent: true,
        symbol: "none",
        label: {show: false},
        lineStyle: {color: "#ef4444", width: 1.5},
        data: [{xAxis: currentFrame}],
      };
      if (markAreaData.length) {
        obsSeries.markArea = {
          silent: true,
          itemStyle: {color: "transparent"},
          data: markAreaData,
        };
      }
    }
    series.push(obsSeries);
    series.push({
      name: `${label} act`,
      type: "line",
      data: xValues.map((x, rowIdx) => [Number(x), cleanChartNumber(actRows[rowIdx]?.[dim])]),
      showSymbol: false,
      connectNulls: false,
      animation: false,
      sampling: "lttb",
      lineStyle: {color, width: 1.1, type: "dashed", opacity: 0.62},
      itemStyle: {color},
    });
  }

  const labelGraphics = labels.slice(0, 5).map((label, idx) => ({
    type: "text",
    left: 54 + idx * 84,
    top: 10,
    style: {
      text: truncateChartLabel(label, 11),
      fill: CHART_COLORS[idx % CHART_COLORS.length],
      font: "10px sans-serif",
    },
  }));

  chart.setOption({
    animation: false,
    backgroundColor: "#ffffff",
    color: CHART_COLORS,
    title: {
      text: title,
      left: 12,
      top: 4,
      textStyle: {color: "#111827", fontSize: 12, fontWeight: 600},
    },
    graphic: labelGraphics,
    tooltip: {
      trigger: "axis",
      confine: true,
      axisPointer: {type: "line"},
      formatter: datasetTooltipFormatter,
    },
    grid: {left: 46, right: 12, top: 30, bottom: 24, containLabel: false},
    xAxis: {
      type: "value",
      min: xMin,
      max: xMax,
      axisLabel: {color: "#64748b", fontSize: 10},
      axisTick: {lineStyle: {color: "#cbd5e1"}},
      axisLine: {lineStyle: {color: "#cbd5e1"}},
      splitLine: {lineStyle: {color: "#eef2f7", width: 1}},
    },
    yAxis: {
      type: "value",
      min: yMin,
      max: yMax,
      axisLabel: {color: "#64748b", fontSize: 10, formatter: formatChartValue},
      axisTick: {lineStyle: {color: "#cbd5e1"}},
      axisLine: {lineStyle: {color: "#cbd5e1"}},
      splitLine: {lineStyle: {color: "#e2e8f0", width: 1}},
    },
    series,
  }, true, true);
}

function datasetSegmentMarkAreas(segments) {
  const colorFor = (name) => {
    const clean = String(name || "").toLowerCase();
    if (clean === "success") return "rgba(34, 197, 94, 0.13)";
    if (clean === "fail" || clean === "failure") return "rgba(239, 68, 68, 0.13)";
    if (clean === "recovery") return "rgba(245, 158, 11, 0.15)";
    return null;
  };
  return segments.map((segment) => {
    const color = colorFor(segment.class_name);
    const start = Number(segment.start_frame);
    const end = Number(segment.end_frame);
    if (!color || !Number.isFinite(start) || !Number.isFinite(end)) return null;
    return [{xAxis: start, itemStyle: {color}}, {xAxis: end}];
  }).filter(Boolean);
}

function datasetTooltipFormatter(params) {
  const items = Array.isArray(params) ? params : [params];
  const frame = items[0]?.data?.[0] ?? items[0]?.axisValue ?? "";
  const lines = [`frame ${frame}`];
  const visible = items.filter((item) => {
    const value = Array.isArray(item.data) ? item.data[1] : item.value;
    return Number.isFinite(Number(value));
  });
  visible.slice(0, 12).forEach((item) => {
    const value = Array.isArray(item.data) ? item.data[1] : item.value;
    lines.push(`${item.marker}${item.seriesName}: ${formatChartValue(value)}`);
  });
  if (visible.length > 12) {
    lines.push(`... ${visible.length - 12} more`);
  }
  return lines.join("<br/>");
}

async function applyMode() {
  try {
    await api("/api/apply-mode", {...selectedPayload(), selected_mode_workers: selectedWorkers()});
    await pollStatus();
  } catch (err) {
    setMessage(err.message, true);
  }
}

async function toggleManual(group) {
  try {
    await api("/api/manual", {
      ...selectedPayload(),
      group,
      action: "toggle",
      collect_dataset_repo_id: $("collectDatasetInput").value || null,
    });
    await pollStatus();
  } catch (err) {
    setMessage(err.message, true);
  }
}

async function toggleLevel(level) {
  const current = Boolean(state.status?.levels?.[level]);
  try {
    await api("/api/level", {name: level, value: !current});
    await pollStatus();
  } catch (err) {
    setMessage(err.message, true);
  }
}

async function toggleCamera() {
  const current = Boolean(state.status?.levels?.camera);
  try {
    await api("/api/level", {name: "camera", value: !current, camera_mode: $("cameraModeSelect").value});
    await pollStatus();
  } catch (err) {
    setMessage(err.message, true);
  }
}

function editSimStereoBaseline(source) {
  const range = $("simStereoBaselineRange");
  const input = $("simStereoBaselineInput");
  if (source === "range") {
    input.value = range.value;
  } else {
    const value = Number(input.value);
    if (Number.isFinite(value)) range.value = String(value);
  }
  state.simStereoBaselineDirty = true;
  if (state.status) renderStatus(state.status);
}

async function applySimStereoBaseline() {
  const baselineMm = Number($("simStereoBaselineInput").value);
  if (!Number.isFinite(baselineMm)) {
    setMessage("Enter a valid sim eye distance", true);
    return;
  }
  try {
    const status = await api("/api/sim-stereo", {baseline_m: baselineMm / 1000});
    state.simStereoBaselineDirty = false;
    renderStatus(status);
  } catch (err) {
    setMessage(err.message, true);
  }
}

async function updateExperimentScene(action) {
  const task = selectedExperimentTask();
  if (action !== "all_reset" && !task) {
    setMessage("Select an experiment task", true);
    return;
  }
  try {
    const status = await api("/api/experiment-scene", {
      action,
      task_id: task?.task_id || null,
    });
    if (action === "all_reset") $("experimentTaskSelect").value = "";
    renderStatus(status);
  } catch (err) {
    setMessage(err.message, true);
  }
}

async function triggerRecord(action) {
  try {
    await api("/api/record", {action, task_name: $("taskInput").value || null});
    await pollStatus();
  } catch (err) {
    setMessage(err.message, true);
  }
}

async function updateWalking(policyEnabled = null) {
  try {
    await api("/api/walking-command", {
      profile: $("walkingProfileInput").value || null,
      vx: Number($("walkVx").value || 0),
      vy: Number($("walkVy").value || 0),
      dyaw: Number($("walkDyaw").value || 0),
      policy_enabled: policyEnabled,
    });
    await pollStatus();
  } catch (err) {
    setMessage(err.message, true);
  }
}

async function toggleReliabilityRuntime() {
  const running = Boolean(state.status?.reliability_runtime?.running);
  try {
    await api("/api/reliability-runtime", {
      action: running ? "stop" : "start",
      hand_model_variant: $("reliabilityHandModelSelect").value || null,
      controller_model_variant: $("reliabilityControllerModelSelect").value || null,
      ...hybridTeleopPayload(),
    });
    await pollStatus();
  } catch (err) {
    setMessage(err.message, true);
  }
}

function hybridTeleopPayload() {
  return {
    left_device: $("hybridLeftDeviceSelect").value,
    right_device: $("hybridRightDeviceSelect").value,
    trapezoid_preprocess: $("hybridTrapezoidToggle").checked,
    trapezoid_bottom_width: Number($("hybridBottomWidthInput").value),
    swap_lr: $("hybridSwapLrToggle").checked,
    mirror: $("hybridMirrorToggle").checked,
  };
}

async function updateHybridTeleop(action) {
  try {
    const status = await api("/api/hybrid-teleop", {
      action,
      ...hybridTeleopPayload(),
    });
    renderStatus(status);
    if (action === "start_test" || action === "save") refreshHybridPreviewImages(true);
  } catch (err) {
    setMessage(err.message, true);
  }
}

function syncHybridBottomWidth(source) {
  const range = $("hybridBottomWidthRange");
  const input = $("hybridBottomWidthInput");
  if (source === "range") {
    input.value = range.value;
  } else {
    const value = Math.max(1, Math.min(640, Number(input.value || 320)));
    input.value = String(Math.round(value));
    range.value = input.value;
  }
}

function setupEvents() {
  $("modeSelect").addEventListener("change", () => {
    updateModeVisibility();
    refreshModeWorkers();
    pushSelection();
  });
  $("teleopDeviceSelect").addEventListener("change", () => updateHandSourceOptions(true));
  $("teleopHandSourceSelect").addEventListener("change", pushSelection);
  $("inferencePolicySelect").addEventListener("change", () => {
    updatePolicyVisibility();
    refreshModeWorkers();
    pushSelection();
  });
  $("cameraModeSelect").addEventListener("change", () => renderStatus(state.status));
  $("simStereoBaselineRange").addEventListener("input", () => editSimStereoBaseline("range"));
  $("simStereoBaselineInput").addEventListener("input", () => editSimStereoBaseline("input"));
  $("simStereoBaselineInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter") applySimStereoBaseline();
  });
  $("simStereoBaselineApplyButton").addEventListener("click", applySimStereoBaseline);
  $("experimentTaskSelect").addEventListener("change", renderExperimentTaskSpec);
  $("experimentSpawnButton").addEventListener("click", () => updateExperimentScene("spawn"));
  $("experimentTaskResetButton").addEventListener("click", () => updateExperimentScene("task_reset"));
  $("experimentAllResetButton").addEventListener("click", () => updateExperimentScene("all_reset"));
  $("applyModeButton").addEventListener("click", applyMode);
  $("shutdownWorkersButton").addEventListener("click", async () => {
    await api("/api/shutdown", {});
    await pollStatus();
  });
  $("cameraStartButton").addEventListener("click", toggleCamera);
  $("reliabilityRuntimeButton").addEventListener("click", toggleReliabilityRuntime);
  $("hybridApplyButton").addEventListener("click", () => updateHybridTeleop("save"));
  $("hybridTestButton").addEventListener("click", () => {
    const running = Boolean(state.status?.hybrid_teleop?.test?.running);
    updateHybridTeleop(running ? "stop_test" : "start_test");
  });
  $("hybridBottomWidthRange").addEventListener("input", () => syncHybridBottomWidth("range"));
  $("hybridBottomWidthInput").addEventListener("change", () => syncHybridBottomWidth("input"));
  $("hybridBottomWidthInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      syncHybridBottomWidth("input");
      updateHybridTeleop("save");
    }
  });
  $("collectDataButton").addEventListener("click", () => toggleManual("collect_data"));
  $("walkingPolicyBrowseButton").addEventListener("click", () => {
    openFileDialog("walking_policy", "walkingPolicyPathInput");
  });
  $("inferenceDatasetBrowseButton").addEventListener("click", () => {
    openFileDialog("inference_dataset", "inferenceDatasetInput");
  });
  $("inferenceCheckpointBrowseButton").addEventListener("click", () => {
    openFileDialog("inference_checkpoint", "inferenceCheckpointInput");
  });
  $("replayDatasetBrowseButton").addEventListener("click", () => {
    openFileDialog("replay_dataset", "replayDatasetInput");
  });
  $("walkingProfileInput").addEventListener("change", () => {
    updateWalkingModelForProfile();
    if (state.status) renderStatus(state.status);
  });
  $("fileDialogCloseButton").addEventListener("click", closeFileDialog);
  $("fileDialogCancelButton").addEventListener("click", closeFileDialog);
  $("fileDialogUpButton").addEventListener("click", () => {
    if (state.fileDialog?.parent) loadFileDialog(state.fileDialog.parent);
  });
  $("fileDialogSelectButton").addEventListener("click", () => {
    selectFileDialogValue(state.fileDialog?.selectedValue);
  });
  $("fileDialog").addEventListener("click", (event) => {
    if (event.target === $("fileDialog")) closeFileDialog();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.fileDialog) closeFileDialog();
  });
  document.querySelectorAll("[data-level]").forEach((button) => {
    button.addEventListener("click", () => toggleLevel(button.dataset.level));
  });
  document.querySelectorAll("[data-record]").forEach((button) => {
    button.addEventListener("click", () => triggerRecord(button.dataset.record));
  });
  $("taskAddButton").addEventListener("click", () => {
    const task = prompt("Task", $("taskInput").value || "");
    if (task && task.trim()) setSelectedTask(task);
  });
  $("taskRemoveButton").addEventListener("click", () => {
    const selected = selectedTask();
    state.tasks = state.tasks.filter((task) => task !== selected);
    $("taskInput").value = state.tasks[0] || "";
    renderTaskList();
    renderStatus(state.status);
  });
  $("walkingUpdateButton").addEventListener("click", () => updateWalking(null));
  $("walkingZeroButton").addEventListener("click", () => {
    $("walkVx").value = "0";
    $("walkVy").value = "0";
    $("walkDyaw").value = "0";
    updateWalking(false);
  });
  $("walkingStartButton").addEventListener("click", () => {
    updateWalking(!state.status?.walking?.policy_enabled);
  });
  document.querySelectorAll("[data-page-tab]").forEach((button) => {
    button.addEventListener("click", () => showPage(button.dataset.pageTab));
  });
  ["logLevelFilter", "logSourceFilter"].forEach((id) => {
    $(id).addEventListener("change", () => renderLogs(state.status?.logs || []));
  });
  $("logSearchInput").addEventListener("input", () => renderLogs(state.status?.logs || []));
  $("logPauseButton").addEventListener("click", () => {
    state.logPaused = !state.logPaused;
    $("logPauseButton").textContent = state.logPaused ? "Resume" : "Pause";
    $("logPauseButton").dataset.active = state.logPaused ? "true" : "false";
    if (!state.logPaused) renderLogs(state.status?.logs || []);
  });
  $("viserReloadButton").addEventListener("click", () => {
    setViserUrl($("viserUrlInput").value, true, true);
  });
  $("viserUrlInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter") setViserUrl($("viserUrlInput").value, true, true);
  });
  $("datasetRootBrowseButton").addEventListener("click", () => {
    openFileDialog("dataset_viewer_root", "datasetRootInput");
  });
  $("datasetRootRefreshButton").addEventListener("click", () => {
    state.datasetViewer.ready = false;
    state.datasetViewer.selectedDataset = null;
    state.datasetViewer.data = null;
    state.datasetViewer.selectedSegment = -1;
    state.datasetViewer.segmentDirty = false;
    state.datasetViewer.classDirty = false;
    setDatasetPlaying(false);
    loadDatasetList();
    renderDatasetViewerData();
  });
  $("datasetRootInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter") $("datasetRootRefreshButton").click();
  });
  $("datasetEpisodeSelect").addEventListener("change", () => {
    loadDatasetEpisode(Number($("datasetEpisodeSelect").value || 0));
  });
  $("datasetImageStreamSelect").addEventListener("change", () => {
    loadDatasetEpisode(Number($("datasetEpisodeSelect").value || 0));
  });
  $("datasetPlayButton").addEventListener("click", () => {
    setDatasetPlaying(!state.datasetViewer.playing);
  });
  $("datasetFrameSlider").addEventListener("input", () => {
    setDatasetPlaying(false);
    updateDatasetFrame(Number($("datasetFrameSlider").value || 0));
  });
  $("datasetSegmentClassSelect").addEventListener("change", () => {
    assignDatasetSegmentClass($("datasetSegmentClassSelect").value);
  });
  $("datasetClassManageSelect").addEventListener("change", () => {
    const data = state.datasetViewer.data;
    const selected = (data?.classes || []).find((entry) => String(entry.id) === $("datasetClassManageSelect").value);
    $("datasetClassNameInput").value = selected?.name || "";
    renderDatasetLabelControls();
  });
  $("datasetSplitSegmentButton").addEventListener("click", splitDatasetSegmentAtFrame);
  $("datasetMergePrevButton").addEventListener("click", () => mergeDatasetSegment("prev"));
  $("datasetMergeNextButton").addEventListener("click", () => mergeDatasetSegment("next"));
  $("datasetResetLabelsButton").addEventListener("click", resetDatasetEpisodeLabels);
  $("datasetAddClassButton").addEventListener("click", addDatasetLabelClass);
  $("datasetRenameClassButton").addEventListener("click", renameDatasetLabelClass);
  $("datasetSaveLabelsButton").addEventListener("click", saveDatasetLabels);
}

setupEvents();
initCameraStreams();
initHybridPreviewImages();
setCameraStreamsActive(true);
showPage(window.location.hash.slice(1) || "teleop");
pollStatus().then(refreshModeWorkers);
setInterval(pollStatus, 1000);
setInterval(pollTelemetry, 250);
setInterval(() => refreshHybridPreviewImages(false), 300);
window.addEventListener("resize", () => {
  resizeCharts();
  pollTelemetry();
  if (state.activePage === "dataset") redrawDatasetViewer();
});
window.addEventListener("hashchange", () => {
  showPage(window.location.hash.slice(1));
});
