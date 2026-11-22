/* Frontend controller: Single-Screen Studio (Live Webcam / Upload Gambar / Upload Video + WS backpressure) */
(() => {
  const els = {
    statusDot: document.getElementById("status-dot"),
    modelStatus: document.getElementById("model-status"),
    liveFps: document.getElementById("live-fps"),
    liveMs: document.getElementById("live-ms"),
    modeBtns: Array.from(document.querySelectorAll(".mode-btn")),
    video: document.getElementById("video-source"),
    canvas: document.getElementById("overlay-canvas"),
    emptyOverlay: document.getElementById("empty-overlay"),
    emptyHint: document.getElementById("empty-hint"),
    uploadBar: document.getElementById("upload-bar"),
    uploadLabel: document.getElementById("upload-label"),
    fileInput: document.getElementById("file-input"),
    fileName: document.getElementById("file-name"),
    btnStart: document.getElementById("btn-start"),
    btnStop: document.getElementById("btn-stop"),
    btnReset: document.getElementById("btn-reset"),
    btnCapture: document.getElementById("btn-capture"),
    btnExportCsv: document.getElementById("btn-export-csv"),
    btnExportJson: document.getElementById("btn-export-json"),
    confSlider: document.getElementById("conf-slider"),
    confValue: document.getElementById("conf-value"),
    stageTiming: document.getElementById("stage-timing"),
    sessionCount: document.getElementById("session-count"),
    personSelector: document.getElementById("person-selector"),
    statusBadge: document.getElementById("status-badge"),
    statusReason: document.getElementById("status-reason"),
    clfBadge: document.getElementById("clf-badge"),
    probSesuaiVal: document.getElementById("prob-sesuai-val"),
    probSesuaiBar: document.getElementById("prob-sesuai-bar"),
    probTidakVal: document.getElementById("prob-tidak-val"),
    probTidakBar: document.getElementById("prob-tidak-bar"),
    attrSummary: document.getElementById("attr-summary"),
    attrChecklist: document.getElementById("attr-checklist"),
    toast: document.getElementById("toast"),
  };

  const ctx = els.canvas.getContext("2d");
  const sendCanvas = document.createElement("canvas");
  const sendCtx = sendCanvas.getContext("2d");

  const state = {
    mode: "webcam", // webcam | image | video
    running: false,
    inFlight: false,
    ws: null,
    stream: null,
    videoUrl: null,
    loadedImage: null,
    classes: [],
    lastResult: null,
    selectedTrackId: null,
    sessionLogs: [],
  };

  /* Canvas overlay colors strictly derived from DESIGN.md palette */
  const CANVAS_COLORS = {
    sesuai: "#0066cc",       // Action Blue
    tidakSesuai: "#1d1d1f",  // Near-Black Ink
    belumLengkap: "#7a7a7a", // Ink Muted 48
    attrRequired: "#2997ff", // Sky Link Blue
    attrViolation: "#1d1d1f",
    unassigned: "#cccccc",
  };

  function showToast(text) {
    els.toast.textContent = text;
    els.toast.classList.remove("hidden");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => els.toast.classList.add("hidden"), 2800);
  }

  async function init() {
    try {
      const [healthRes, cfgRes] = await Promise.all([
        fetch("/api/health"),
        fetch("/api/config"),
      ]);
      const health = await healthRes.json();
      const cfg = await cfgRes.json();

      state.classes = cfg.classes || [];
      els.confSlider.value = String(cfg.default_conf ?? 0.45);
      els.confValue.textContent = Number(els.confSlider.value).toFixed(2);
      renderChecklist({});

      els.statusDot.classList.add("online");
      const detName = health.attribute_detector || "belum ada";
      const clfText = health.classifier ? "Hybrid Fusion" : "Rule-Only";
      els.modelStatus.textContent = `${detName} · ${clfText}`;
      els.clfBadge.textContent = health.classifier ? "Aktif" : "Belum dilatih";
    } catch (err) {
      els.statusDot.classList.add("error");
      els.modelStatus.textContent = "Server tidak terhubung";
      console.error(err);
    }
  }

  function setMode(nextMode) {
    stopDetection();
    state.mode = nextMode;
    els.modeBtns.forEach((btn) => {
      const isMatch = btn.dataset.mode === nextMode;
      btn.classList.toggle("active", isMatch);
      btn.setAttribute("aria-selected", String(isMatch));
    });
    els.fileInput.value = "";
    els.fileName.textContent = "Belum ada file dipilih";
    state.loadedImage = null;

    if (nextMode === "webcam") {
      els.uploadBar.classList.add("hidden");
      els.emptyHint.innerHTML = "Klik <strong>Mulai Deteksi</strong> di bilah atas untuk mengaktifkan kamera lokal.";
    } else if (nextMode === "image") {
      els.uploadBar.classList.remove("hidden");
      els.uploadLabel.textContent = "Pilih file gambar (JPG/PNG)";
      els.fileInput.accept = "image/*";
      els.emptyHint.innerHTML = "Pilih gambar lalu klik <strong>Mulai Deteksi</strong>.";
    } else {
      els.uploadBar.classList.remove("hidden");
      els.uploadLabel.textContent = "Pilih file video (MP4/MOV/WEBM)";
      els.fileInput.accept = "video/*";
      els.emptyHint.innerHTML = "Pilih video lalu klik <strong>Mulai Deteksi</strong>.";
    }
    els.emptyOverlay.classList.remove("hidden");
  }

  function openWebSocket(trackMode) {
    return new Promise((resolve, reject) => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws/infer`);
      ws.onopen = () => {
        ws.send(JSON.stringify({
          type: "config",
          conf: Number(els.confSlider.value),
          track: trackMode,
        }));
        resolve(ws);
      };
      ws.onerror = (e) => reject(e);
      ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        if (data.type) return;
        state.inFlight = false;
        handleAnalysisResult(data);
        if (state.running && (state.mode === "webcam" || state.mode === "video")) {
          requestAnimationFrame(pumpVideoFrame);
        }
      };
      ws.onclose = () => {
        state.inFlight = false;
      };
      state.ws = ws;
    });
  }

  async function startDetection() {
    if (state.mode === "image") {
      await runSingleImage();
      return;
    }

    try {
      if (state.mode === "webcam") {
        state.stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        els.video.srcObject = state.stream;
        await els.video.play();
      } else {
        if (!state.videoUrl) {
          showToast("Pilih file video terlebih dulu.");
          return;
        }
        els.video.srcObject = null;
        els.video.src = state.videoUrl;
        els.video.loop = true;
        await els.video.play();
      }

      await openWebSocket(true);
      state.running = true;
      state.inFlight = false;
      els.emptyOverlay.classList.add("hidden");
      els.btnStart.disabled = true;
      els.btnStop.disabled = false;
      requestAnimationFrame(pumpVideoFrame);
    } catch (err) {
      showToast(`Gagal memulai: ${err.message || err}`);
      stopDetection();
    }
  }

  function stopDetection() {
    state.running = false;
    state.inFlight = false;
    if (state.ws) {
      state.ws.close();
      state.ws = null;
    }
    if (state.stream) {
      state.stream.getTracks().forEach((t) => t.stop());
      state.stream = null;
    }
    els.video.pause();
    els.btnStart.disabled = false;
    els.btnStop.disabled = true;
  }

  function pumpVideoFrame() {
    if (!state.running || state.inFlight || !state.ws || state.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    const vw = els.video.videoWidth;
    const vh = els.video.videoHeight;
    if (!vw || !vh) {
      requestAnimationFrame(pumpVideoFrame);
      return;
    }

    const scale = Math.min(1, 640 / Math.max(vw, vh));
    const sw = Math.round(vw * scale);
    const sh = Math.round(vh * scale);
    sendCanvas.width = sw;
    sendCanvas.height = sh;
    sendCtx.drawImage(els.video, 0, 0, sw, sh);

    state.inFlight = true;
    sendCanvas.toBlob((blob) => {
      if (!blob || !state.ws || state.ws.readyState !== WebSocket.OPEN) {
        state.inFlight = false;
        return;
      }
      state.ws.send(blob);
    }, "image/jpeg", 0.75);
  }

  async function runSingleImage() {
    const file = els.fileInput.files?.[0];
    if (!file) {
      showToast("Pilih file gambar terlebih dulu.");
      return;
    }
    const form = new FormData();
    form.append("file", file);
    const conf = Number(els.confSlider.value);
    try {
      const res = await fetch(`/api/infer-image?conf=${conf}`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      els.emptyOverlay.classList.add("hidden");
      handleAnalysisResult(data);
    } catch (err) {
      showToast(`Inferensi gambar gagal: ${err.message || err}`);
    }
  }

  function handleAnalysisResult(data) {
    state.lastResult = data;
    els.liveFps.textContent = data.timing.fps.toFixed(1);
    els.liveMs.textContent = `${data.timing.total_ms.toFixed(1)} ms`;
    els.stageTiming.textContent =
      `Person ${data.timing.person_ms} ms · Atribut ${data.timing.attribute_ms} ms · ` +
      `EffNet ${data.timing.classifier_ms} ms · Fusion ${data.timing.fusion_ms} ms`;

    drawCanvasOverlay(data);
    updatePersonSelector(data.persons);
    const active = pickSelectedPerson(data.persons);
    updateInspector(active, data.classifier_loaded);
    recordSessionLogs(data);
  }

  function drawCanvasOverlay(data) {
    let srcEl = null;
    let srcW = data.frame_width;
    let srcH = data.frame_height;

    if (state.mode === "image" && state.loadedImage) {
      srcEl = state.loadedImage;
      srcW = state.loadedImage.naturalWidth;
      srcH = state.loadedImage.naturalHeight;
    } else if (els.video.videoWidth > 0) {
      srcEl = els.video;
      srcW = els.video.videoWidth;
      srcH = els.video.videoHeight;
    }
    if (!srcEl) return;

    els.canvas.width = srcW;
    els.canvas.height = srcH;
    ctx.drawImage(srcEl, 0, 0, srcW, srcH);

    const sx = srcW / data.frame_width;
    const sy = srcH / data.frame_height;

    for (const attr of data.unassigned_attributes || []) {
      drawBox(
        attr.bbox,
        `${attr.class_name} ${(attr.confidence * 100).toFixed(0)}%`,
        CANVAS_COLORS.unassigned,
        "#1d1d1f",
        sx,
        sy,
        true
      );
    }

    for (const p of data.persons || []) {
      const isSesuai = p.status === "SESUAI";
      const isTidak = p.status === "TIDAK SESUAI";
      const color = isSesuai
        ? CANVAS_COLORS.sesuai
        : isTidak
          ? CANVAS_COLORS.tidakSesuai
          : CANVAS_COLORS.belumLengkap;
      const prefix = isSesuai ? "✓" : isTidak ? "✕" : "◔";
      drawBox(p.bbox, `${prefix} #${p.track_id} ${p.status}`, color, "#ffffff", sx, sy, false);

      for (const attr of p.attributes || []) {
        const isViol = attr.class_name === "kaos_oblong" || attr.class_name === "sandal";
        const aColor = isViol ? CANVAS_COLORS.attrViolation : CANVAS_COLORS.attrRequired;
        drawBox(
          attr.bbox,
          `${attr.class_name} ${(attr.confidence * 100).toFixed(0)}%`,
          aColor,
          "#ffffff",
          sx,
          sy,
          true
        );
      }
    }
  }

  function drawBox(bbox, label, color, textColor, sx, sy, thin) {
    const x1 = bbox[0] * sx;
    const y1 = bbox[1] * sy;
    const w = (bbox[2] - bbox[0]) * sx;
    const h = (bbox[3] - bbox[1]) * sy;

    ctx.lineWidth = thin ? 2 : 3;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, w, h);

    ctx.font = thin
      ? '600 13px "SF Pro Text", system-ui, -apple-system, sans-serif'
      : '600 15px "SF Pro Display", system-ui, -apple-system, sans-serif';
    const padX = 8;
    const textW = ctx.measureText(label).width + padX * 2;
    const textH = thin ? 22 : 26;
    const labelY = Math.max(0, y1 - textH);
    ctx.fillStyle = color;
    ctx.fillRect(x1, labelY, textW, textH);
    ctx.fillStyle = textColor;
    ctx.fillText(label, x1 + padX, labelY + textH - 7);
  }

  function updatePersonSelector(persons) {
    if (!persons || persons.length <= 1) {
      els.personSelector.classList.add("hidden");
      state.selectedTrackId = persons?.[0]?.track_id ?? null;
      return;
    }
    els.personSelector.classList.remove("hidden");
    const ids = persons.map((p) => p.track_id);
    if (!ids.includes(state.selectedTrackId)) {
      state.selectedTrackId = ids[0];
    }
    els.personSelector.innerHTML = persons
      .map((p) => `<option value="${p.track_id}">Mahasiswa #${p.track_id} (${p.status})</option>`)
      .join("");
    els.personSelector.value = String(state.selectedTrackId);
  }

  function pickSelectedPerson(persons) {
    if (!persons || persons.length === 0) return null;
    return persons.find((p) => p.track_id === state.selectedTrackId) || persons[0];
  }

  function updateInspector(person, classifierLoaded) {
    if (!person) {
      els.statusBadge.className = "status-badge status-idle";
      els.statusBadge.textContent = "MENUNGGU";
      els.statusReason.textContent = "Belum ada mahasiswa terdeteksi pada frame.";
      els.probSesuaiVal.textContent = "—";
      els.probTidakVal.textContent = "—";
      els.probSesuaiBar.style.width = "0%";
      els.probTidakBar.style.width = "0%";
      renderChecklist({});
      return;
    }

    const clsMap = {
      SESUAI: "status-sesuai",
      "TIDAK SESUAI": "status-tidak",
      "BELUM LENGKAP": "status-belum",
    };
    const symbolMap = {
      SESUAI: "✓ ",
      "TIDAK SESUAI": "✕ ",
      "BELUM LENGKAP": "◔ ",
    };
    els.statusBadge.className = `status-badge ${clsMap[person.status] || "status-idle"}`;
    els.statusBadge.textContent = `${symbolMap[person.status] || ""}${person.status}`;
    els.statusReason.textContent = person.reason;

    if (classifierLoaded && person.prob_sesuai !== null) {
      const ps = (person.prob_sesuai * 100).toFixed(1);
      const pt = (person.prob_tidak_sesuai * 100).toFixed(1);
      els.clfBadge.textContent = "Aktif";
      els.probSesuaiVal.textContent = `${ps}%`;
      els.probTidakVal.textContent = `${pt}%`;
      els.probSesuaiBar.style.width = `${ps}%`;
      els.probTidakBar.style.width = `${pt}%`;
    } else {
      els.clfBadge.textContent = "Mode Rule-Only";
      els.probSesuaiVal.textContent = "N/A";
      els.probTidakVal.textContent = "N/A";
      els.probSesuaiBar.style.width = "0%";
      els.probTidakBar.style.width = "0%";
    }

    const bestConfByClass = {};
    for (const a of person.attributes || []) {
      bestConfByClass[a.class_name] = Math.max(bestConfByClass[a.class_name] || 0, a.confidence);
    }
    renderChecklist(bestConfByClass);
  }

  function renderChecklist(confByClass) {
    let detectedCount = 0;
    els.attrChecklist.innerHTML = state.classes
      .map((c) => {
        const score = confByClass[c.name] || 0;
        const found = score > 0;
        if (found) detectedCount += 1;
        const isViolation = c.role === "violation";
        const itemCls = found ? (isViolation ? "detected-bad" : "detected-ok") : "";
        const barCls = !found ? "bar-muted" : isViolation ? "bar-ink" : "bar-primary";
        const pct = (score * 100).toFixed(0);
        const icon = found ? (isViolation ? "✕" : "✓") : "○";
        const tag = isViolation
          ? '<span class="attr-tag violation">Pelanggaran</span>'
          : '<span class="attr-tag">Wajib</span>';
        return `
          <li class="attr-item ${itemCls}">
            <div class="attr-top">
              <span class="attr-name" title="${c.display_name}">
                <span class="attr-symbol">${icon}</span>
                <span>${c.display_name}</span>
              </span>
              <strong class="attr-score">${found ? `${pct}%` : "—"}</strong>
            </div>
            <div class="bar-track">
              <div class="bar-fill ${barCls}" style="width: ${pct}%"></div>
            </div>
            <div class="attr-meta">${tag}</div>
          </li>
        `;
      })
      .join("");
    els.attrSummary.textContent = `${detectedCount}/${state.classes.length} terdeteksi`;
  }

  function recordSessionLogs(data) {
    const now = new Date().toISOString();
    for (const p of data.persons || []) {
      state.sessionLogs.push({
        timestamp: now,
        track_id: p.track_id,
        status: p.status,
        reason: p.reason,
        prob_sesuai: p.prob_sesuai ?? "",
        attributes: (p.attributes || []).map((a) => `${a.class_name}:${a.confidence.toFixed(2)}`).join(";"),
        total_ms: data.timing.total_ms,
        fps: data.timing.fps,
      });
    }
    if (state.sessionLogs.length > 5000) {
      state.sessionLogs.splice(0, state.sessionLogs.length - 5000);
    }
    els.sessionCount.textContent = `Log: ${state.sessionLogs.length}`;
  }

  async function resetSession() {
    state.sessionLogs = [];
    state.lastResult = null;
    els.sessionCount.textContent = "Log: 0";
    updateInspector(null, false);
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "reset" }));
    } else {
      await fetch("/api/reset", { method: "POST" }).catch(() => {});
    }
    showToast("Sesi dan tracker berhasil di-reset.");
  }

  async function captureCurrent() {
    if (!els.canvas.width || !state.lastResult) {
      showToast("Belum ada hasil deteksi untuk di-capture.");
      return;
    }
    const imageBase64 = els.canvas.toDataURL("image/jpeg", 0.9);
    const res = await fetch("/api/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: imageBase64, analysis: state.lastResult }),
    });
    if (res.ok) {
      showToast("Frame & metadata tersimpan ke outputs/sessions/captures/");
    } else {
      showToast("Gagal menyimpan capture.");
    }
  }

  async function exportLogs(format) {
    if (state.sessionLogs.length === 0) {
      showToast("Belum ada log sesi untuk diekspor.");
      return;
    }
    const res = await fetch("/api/sessions/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ format, records: state.sessionLogs }),
    });
    if (!res.ok) {
      showToast("Gagal mengekspor log sesi.");
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `attire_session.${format}`;
    a.click();
    URL.revokeObjectURL(url);
    showToast(`Export ${format.toUpperCase()} selesai (${state.sessionLogs.length} entri).`);
  }

  // Event bindings
  els.modeBtns.forEach((btn) => btn.addEventListener("click", () => setMode(btn.dataset.mode)));
  els.btnStart.addEventListener("click", startDetection);
  els.btnStop.addEventListener("click", stopDetection);
  els.btnReset.addEventListener("click", resetSession);
  els.btnCapture.addEventListener("click", captureCurrent);
  els.btnExportCsv.addEventListener("click", () => exportLogs("csv"));
  els.btnExportJson.addEventListener("click", () => exportLogs("json"));

  els.confSlider.addEventListener("input", () => {
    const val = Number(els.confSlider.value);
    els.confValue.textContent = val.toFixed(2);
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "config", conf: val }));
    }
  });

  els.personSelector.addEventListener("change", () => {
    state.selectedTrackId = Number(els.personSelector.value);
    if (state.lastResult) {
      updateInspector(pickSelectedPerson(state.lastResult.persons), state.lastResult.classifier_loaded);
    }
  });

  els.fileInput.addEventListener("change", () => {
    const file = els.fileInput.files?.[0];
    if (!file) return;
    els.fileName.textContent = file.name;
    if (state.mode === "image") {
      const img = new Image();
      img.onload = () => {
        state.loadedImage = img;
        els.canvas.width = img.naturalWidth;
        els.canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        els.emptyOverlay.classList.add("hidden");
      };
      img.src = URL.createObjectURL(file);
    } else if (state.mode === "video") {
      if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
      state.videoUrl = URL.createObjectURL(file);
    }
  });

  init();
})();
