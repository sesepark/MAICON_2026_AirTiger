// 드론 화면 수집 · 라벨링 웹 화면 (../label_drone_frames.py). 외부 라이브러리 없음.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

async function api(path, opts) {
  const r = await fetch(path, opts);
  let j = null;
  try { j = await r.json(); } catch (_) { /* 본문 없음 */ }
  if (!r.ok) throw new Error((j && j.error) || "HTTP " + r.status);
  return j;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
const q = (o) => Object.entries(o).map(([k, v]) => k + "=" + encodeURIComponent(v)).join("&");
const imgUrl = (session, variant, file) => "/img?" + q({ session, variant, file });

const S = {
  status: null, classes: [], cls: "box",
  sessions: [], session: "", variant: "", labelVariant: "undist_top",
  images: [], view: [], cur: -1, viewKey: "",
  img: null, file: null, boxes: [], labStatus: null, sel: -1, dirty: false, history: [],
  quadMode: false, quad: [], note: "", tf: { s: 1, ox: 0, oy: 0 }, drag: null, space: false, mouse: null,
  loadToken: 0,
};

// ───────────────────────── 탭
function showTab(name) {
  document.querySelectorAll("button.tab").forEach((x) => x.classList.toggle("active", x.dataset.tab === name));
  $("tab-collect").classList.toggle("hidden", name !== "collect");
  $("tab-label").classList.toggle("hidden", name !== "label");
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  if (name === "label") { loadSessions(); resizeCanvas(); }
}
document.querySelectorAll("button.tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ───────────────────────── 상태 (1초마다)
let defaultsFilled = false;
async function pollStatus() {
  try {
    const st = await api("/api/status");
    S.status = st;
    if (!S.classes.length) { S.classes = st.classes; S.cls = st.classes[0].id; renderClasses(); }
    if (!defaultsFilled) {
      defaultsFilled = true;
      if (!$("c-server").value) $("c-server").value = st.defaults.server || "";
      if (st.defaults.interval) $("c-interval").value = st.defaults.interval;
      if (st.defaults.cameras) document.querySelectorAll(".c-cam").forEach((x) => { x.checked = st.defaults.cameras.includes(x.value); });
      if (st.defaults.label_mode) S.labelVariant = "undist_" + st.defaults.label_mode;
    }
    renderCollect(st);
    renderPrep(st.prep);
    const warn = st.prep_ok ? "" : "⚠ " + st.prep_why;
    $("c-prepwarn").textContent = warn; $("c-prepwarn").classList.toggle("hidden", !warn);
  } catch (e) {
    $("c-summary").textContent = "서버와 연결 안 됨: " + e.message;
  }
}
setInterval(pollStatus, 1000);

// ───────────────────────── 수집
let lastCamFiles = {};
function renderCollect(st) {
  const c = st.collect;
  const running = !!(c && c.running);
  $("collect-badge").classList.toggle("hidden", !running);
  $("c-start").disabled = running;
  $("c-stop").disabled = !running;
  $("c-code").placeholder = st.code_set ? "(설정됨 — 비워 두면 그대로)" : "임무코드";
  if (!c) { $("c-summary").textContent = "수집 중이 아닙니다."; return; }
  const since = c.started ? Math.round(((c.ended || Date.now() / 1000) - c.started)) : 0;
  const total = Object.values(c.cameras).reduce((a, x) => a + x.saved, 0);
  $("c-summary").innerHTML = (running ? "<b>수집 중</b>" : "멈춤" + (c.end_reason ? " — " + esc(c.end_reason) : "")) +
    " · 세션 <b>" + esc(st.session) + "</b> · " + fmtDur(since) + " · 저장 " + total + " 장" +
    (c.free_gb != null ? " · 디스크 여유 " + c.free_gb.toFixed(1) + " GB" : "");
  const grid = $("c-cams");
  for (const [cam, x] of Object.entries(c.cameras)) {
    let box = document.getElementById("cam-" + cam);
    if (!box) {
      box = document.createElement("div");
      box.className = "cam"; box.id = "cam-" + cam;
      box.innerHTML = "<b>" + esc(cam) + "</b><img alt=''><div class='st'></div>";
      grid.appendChild(box);
    }
    box.querySelector(".st").textContent = "저장 " + x.saved + " · 실패 " + x.errors + " — " + x.state;
    if (x.last_file && lastCamFiles[cam] !== x.last_file) {
      lastCamFiles[cam] = x.last_file;
      box.querySelector("img").src = imgUrl(st.session, "raw", x.last_file);
    }
  }
  for (const el of [...grid.children]) if (!(el.id.slice(4) in c.cameras)) el.remove();
  const log = $("c-log");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.textContent = c.log.join("\n");
  if (atBottom) log.scrollTop = log.scrollHeight;
}
function fmtDur(s) { const m = Math.floor(s / 60); return (m ? m + "분 " : "") + (s % 60) + "초"; }

$("c-start").addEventListener("click", async () => {
  const cameras = [...document.querySelectorAll(".c-cam")].filter((x) => x.checked).map((x) => x.value);
  try {
    const r = await post("/api/collect/start", {
      server: $("c-server").value.trim(), code: $("c-code").value.trim(),
      cameras, name: $("c-name").value.trim(), resume: $("c-resume").value, interval: parseFloat($("c-interval").value) || 7,
    });
    if (r.note) alert(r.note);
    $("c-code").value = "";
    lastCamFiles = {}; $("c-cams").innerHTML = "";
    $("c-summary").textContent = "시작: " + r.session;
    pollStatus();
  } catch (e) { alert("시작 못 함: " + e.message); }
});
$("c-stop").addEventListener("click", async () => {
  $("c-stop").disabled = true;
  try { await post("/api/collect/stop"); } catch (e) { alert(e.message); }
  pollStatus(); loadSessions();
});

// ───────────────────────── 세션 · 그림 목록
async function loadSessions() {
  try { S.sessions = await api("/api/sessions"); } catch (e) { return; }
  const sel = $("l-session");
  const keep = S.session || sel.value;
  sel.innerHTML = S.sessions.map((s) => "<option value='" + esc(s.id) + "'>" + esc(s.id) + (s.collecting ? " ●" : "") + "</option>").join("");
  if (!S.sessions.length) sel.innerHTML = "<option value=''>(세션 없음)</option>";
  if (keep && S.sessions.some((s) => s.id === keep)) sel.value = keep;
  const rs = $("c-resume");
  const rkeep = rs.value;
  rs.innerHTML = "<option value=''>새 세션</option>" + S.sessions.filter((s) => !s.id.startsWith("@"))
    .map((s) => "<option value='" + esc(s.id) + "'>" + esc(s.id) + "</option>").join("");
  rs.value = rkeep;
  renderVariants();
  if (sel.value && (sel.value !== S.session || !S.images.length)) await selectSession(sel.value);
}
function curSession() { return S.sessions.find((s) => s.id === S.session); }
function renderVariants() {
  const s = S.sessions.find((x) => x.id === $("l-session").value);
  const vs = s ? s.variants.filter((v) => v.id.startsWith("undist_")) : []; // 라벨은 왜곡을 편 그림에만
  const sel = $("l-variant");
  if (!vs.length) { sel.innerHTML = "<option value=''>(전처리 전 — 아래 「전처리」)</option>"; return; }
  sel.innerHTML = vs.map((v) => {
    const done = (v.labels && v.labels.done) || 0;
    return "<option value='" + v.id + "'>" + v.id + " (" + v.frames + "장 · 완료 " + done + ")</option>";
  }).join("");
  if (vs.some((v) => v.id === S.variant)) sel.value = S.variant;
  else if (vs.some((v) => v.id === S.labelVariant)) sel.value = S.labelVariant;
}
$("l-session").addEventListener("change", (e) => selectSession(e.target.value));
$("l-variant").addEventListener("change", async (e) => { await leaveImage(); clearImage(); S.variant = e.target.value; S.cur = -1; await loadImages(); });
function clearImage() {
  S.loadToken++; S.img = null; S.file = null; S.boxes = []; S.sel = -1; S.dirty = false; S.history = []; S.quad = [];
  $("empty").classList.remove("hidden"); draw(); updateStatus(); setState("");
}
for (const id of ["l-camera", "l-filter", "l-stride"]) $(id).addEventListener("change", () => buildView());

async function selectSession(id) {
  await leaveImage();
  clearImage();
  S.session = id; S.cur = -1;
  $("l-session").value = id;
  renderVariants();
  S.variant = $("l-variant").value || "";
  await loadImages();
}
async function loadImages() {
  if (!S.session || !S.variant) { S.images = []; buildView(); return; }
  try { S.images = await api("/api/images?" + q({ session: S.session, variant: S.variant })); } catch (e) { S.images = []; }
  const cams = [...new Set(S.images.map((x) => x.camera))].sort();
  const cs = $("l-camera"), keep = cs.value;
  cs.innerHTML = "<option value=''>모든 카메라</option>" + cams.map((c) => "<option>" + esc(c) + "</option>").join("");
  if (cams.includes(keep)) cs.value = keep;
  buildView();
  if (S.cur < 0 && !S.file && S.view.length && !$("tab-label").classList.contains("hidden")) {
    const todo = S.view.findIndex((x) => !x.status);
    openAt(todo >= 0 ? todo : 0); // 처음 열면 안 한 첫 장
  }
}
function buildView() {
  const cam = $("l-camera").value, filt = $("l-filter").value;
  const stride = Math.max(1, parseInt($("l-stride").value, 10) || 1);
  const curFile = S.view[S.cur] && S.view[S.cur].file;
  const perCam = {};
  S.view = S.images.filter((x) => {
    if (cam && x.camera !== cam) return false;
    const k = (perCam[x.camera] = (perCam[x.camera] ?? -1) + 1);
    const st = x.status || "todo";
    if (!(k % stride === 0 || x.status)) return false;
    return filt === "all" || filt === st;
  });
  S.cur = curFile ? S.view.findIndex((x) => x.file === curFile) : -1;
  renderList();
}
function renderList() {
  const done = S.images.filter((x) => x.status === "done").length;
  $("l-progress").textContent = "완료 " + done + " · 목록 " + S.view.length + " / " + S.images.length;
  $("l-list").innerHTML = S.view.map((x, i) =>
    "<li data-i='" + i + "'" + (i === S.cur ? " class='cur'" : "") + "><span class='dot " + (x.status || "") + "'></span>" +
    esc(x.camera) + " " + esc(shortName(x.file)) + (x.n ? " · " + x.n : "") + "</li>").join("");
  const cur = $("l-list").querySelector("li.cur");
  if (cur) cur.scrollIntoView({ block: "nearest" });
}
function shortName(f) { const b = f.split("/").pop().replace(/\.jpe?g$/i, ""); const m = b.match(/^\d{8}_(\d{2})(\d{2})(\d{2})_(\d{3})/); return m ? m[1] + ":" + m[2] + ":" + m[3] + "." + m[4] : b; }
$("l-list").addEventListener("click", (e) => { const li = e.target.closest("li"); if (li) openAt(+li.dataset.i); });

// 수집 중인 세션이면 그림 목록을 가끔 새로
setInterval(() => {
  const s = curSession();
  const p = S.status && S.status.prep;
  const following = p && p.running && p.session === S.session;
  if (s && (s.collecting || following) && !$("tab-label").classList.contains("hidden")) loadImages();
}, 5000);
setInterval(() => { if (!$("tab-label").classList.contains("hidden")) loadSessionsQuiet(); }, 10000);
async function loadSessionsQuiet() {
  try { S.sessions = await api("/api/sessions"); } catch (_) { return; }
  renderVariants();
  if (!S.variant && $("l-variant").value) { S.variant = $("l-variant").value; await loadImages(); } // 전처리가 막 생겼다
}

// ───────────────────────── 그림 열기 · 저장
async function leaveImage() {
  if (S.file && S.dirty) await save(S.labStatus === "done" ? "done" : "draft");
}
async function openAt(i) {
  if (i < 0 || i >= S.view.length) return;
  await leaveImage();
  const token = ++S.loadToken;
  S.cur = i; renderList();
  const item = S.view[i];
  const img = new Image();
  img.src = imgUrl(S.session, S.variant, item.file);
  try { await img.decode(); } catch (e) { if (token === S.loadToken) setState("그림을 못 읽음"); return; }
  let lab = null;
  try { lab = await api("/api/label?" + q({ session: S.session, variant: S.variant, image: item.file })); } catch (_) { /* 없음 */ }
  if (token !== S.loadToken) return;
  const firstImage = !S.img || S.img.naturalWidth !== img.naturalWidth || S.img.naturalHeight !== img.naturalHeight;
  S.img = img; S.file = item.file; S.sel = -1; S.history = []; S.quad = []; S.dirty = false;
  S.boxes = lab ? lab.boxes.map((b) => ({ cls: b.cls, pts: b.pts.map((p) => [p[0], p[1]]) })) : [];
  S.labStatus = lab ? lab.status : null;
  S.note = "";
  $("empty").classList.add("hidden");
  if (firstImage) fit();
  if (!lab && $("t-autocopy").checked) await copyPrev(true);
  draw(); updateStatus();
}
async function save(status) {
  if (!S.file) return;
  const body = { session: S.session, variant: S.variant, image: S.file, width: S.img.naturalWidth, height: S.img.naturalHeight,
    status, boxes: S.boxes };
  try {
    const r = await post("/api/label", body);
    if (r.saved) {
      S.labStatus = status; S.dirty = false;
      for (const x of S.images) if (x.file === S.file) { x.status = status; x.n = S.boxes.length; }
    }
  } catch (e) { alert("저장 못 함: " + e.message); return false; }
  renderList(); updateStatus();
  return true;
}
async function next(d) {
  if (!S.view.length) return;
  const i = S.cur < 0 ? 0 : S.cur + d;
  if (i >= 0 && i < S.view.length) await openAt(i);
  else { S.note = d > 0 ? "목록의 마지막 장입니다" : "목록의 첫 장입니다"; updateStatus(); }
}
async function markDone() { if (await save("done")) next(1); }
async function markSkip() { if (await save("skip")) next(1); }

// 같은 카메라의 앞(없으면 뒤) 라벨 — 박스는 경기 중 안 움직이니 그대로 가져와 고치면 빠르다
async function copyPrev(auto) {
  if (!S.file) return;
  const all = S.images, me = all.findIndex((x) => x.file === S.file);
  const cam = all[me] && all[me].camera;
  const ok = (x) => x.camera === cam && (x.status === "done" || x.status === "draft") && x.n > 0;
  let src = null;
  for (let i = me - 1; i >= 0 && !src; i--) if (ok(all[i])) src = all[i];
  for (let i = me + 1; i < all.length && !src; i++) if (ok(all[i])) src = all[i];
  if (!src) { if (!auto) { S.note = "가져올 라벨이 없습니다"; updateStatus(); } return; }
  const lab = await api("/api/label?" + q({ session: S.session, variant: S.variant, image: src.file }));
  if (!lab || lab.width !== S.img.naturalWidth || lab.height !== S.img.naturalHeight) return;
  pushHistory();
  S.boxes = lab.boxes.map((b) => ({ cls: b.cls, pts: b.pts.map((p) => [p[0], p[1]]) }));
  S.sel = -1;
  S.dirty = !auto; // 자동으로 채운 것은 손대지 않으면 저장하지 않는다
  S.note = (auto ? "자동 복사: " : "복사: ") + shortName(src.file) + " 의 박스 " + S.boxes.length + "개 — 배치가 바뀐 장이면 고치세요";
  draw(); updateStatus();
}

// ───────────────────────── 종류 단추
function renderClasses() {
  $("t-classes").innerHTML = S.classes.map((c, i) =>
    "<button class='cls-btn" + (c.id === S.cls ? " on" : "") + "' data-cls='" + c.id + "'><i style='background:" + c.color + "'></i>" +
    (i + 1) + " " + esc(c.name) + "</button>").join(" ");
}
$("t-classes").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setClass(b.dataset.cls); });
function setClass(id) {
  S.cls = id; renderClasses();
  if (S.sel >= 0 && S.boxes[S.sel].cls !== id) { pushHistory(); S.boxes[S.sel].cls = id; changed(); }
}
const colorOf = (cls) => (S.classes.find((c) => c.id === cls) || { color: "#fff" }).color;

// ───────────────────────── 캔버스
const cv = $("cv"), ctx = cv.getContext("2d");
function resizeCanvas() {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(1, Math.round(r.width * dpr)); cv.height = Math.max(1, Math.round(r.height * dpr));
  draw();
}
window.addEventListener("resize", resizeCanvas);
// 캔버스 픽셀 크기를 화면 크기와 늘 같게 — 다르면 그림이 늘어나 보이고 마우스 위치와 박스가 어긋난다
new ResizeObserver(() => { const had = !!S.img && S.fitted; resizeCanvas(); if (had) fit(); }).observe(cv);
function fit() {
  if (!S.img) return;
  const dpr = window.devicePixelRatio || 1, rr = cv.getBoundingClientRect();
  if (cv.width !== Math.max(1, Math.round(rr.width * dpr)) || cv.height !== Math.max(1, Math.round(rr.height * dpr))) {
    cv.width = Math.max(1, Math.round(rr.width * dpr)); cv.height = Math.max(1, Math.round(rr.height * dpr));
  }
  S.fitted = true;
  const r = cv.getBoundingClientRect();
  const s = Math.min(r.width / S.img.naturalWidth, r.height / S.img.naturalHeight) * 0.98;
  S.tf = { s, ox: (r.width - S.img.naturalWidth * s) / 2, oy: (r.height - S.img.naturalHeight * s) / 2 };
  draw();
}
const toScr = (p) => [p[0] * S.tf.s + S.tf.ox, p[1] * S.tf.s + S.tf.oy];
const toImg = (x, y) => [(x - S.tf.ox) / S.tf.s, (y - S.tf.oy) / S.tf.s];

function draw() {
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!S.img) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = S.tf.s < 2;
  ctx.drawImage(S.img, S.tf.ox, S.tf.oy, S.img.naturalWidth * S.tf.s, S.img.naturalHeight * S.tf.s);
  S.boxes.forEach((b, i) => drawPoly(b.pts, colorOf(b.cls), i === S.sel, i + 1));
  const d = S.drag;
  if (d && d.kind === "rect") drawPoly(rectPts(d.start, d.now), colorOf(S.cls), false, null, true);
  if (S.quadMode && S.quad.length) {
    const pts = S.mouse ? S.quad.concat([toImg(S.mouse[0], S.mouse[1])]) : S.quad;
    ctx.strokeStyle = colorOf(S.cls); ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.beginPath(); pts.forEach((p, k) => { const s = toScr(p); k ? ctx.lineTo(s[0], s[1]) : ctx.moveTo(s[0], s[1]); }); ctx.stroke();
    ctx.setLineDash([]);
    S.quad.forEach((p) => handle(toScr(p), colorOf(S.cls)));
  }
  if (S.mouse && !S.drag) { // 십자선
    const r = cv.getBoundingClientRect();
    ctx.strokeStyle = "rgba(255,255,255,.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, S.mouse[1] + .5); ctx.lineTo(r.width, S.mouse[1] + .5);
    ctx.moveTo(S.mouse[0] + .5, 0); ctx.lineTo(S.mouse[0] + .5, r.height); ctx.stroke();
  }
}
function drawPoly(pts, color, selected, num, dashed) {
  ctx.beginPath();
  pts.forEach((p, k) => { const s = toScr(p); k ? ctx.lineTo(s[0], s[1]) : ctx.moveTo(s[0], s[1]); });
  ctx.closePath();
  ctx.fillStyle = color + (selected ? "40" : "1c"); ctx.fill();
  ctx.strokeStyle = color; ctx.lineWidth = selected ? 2.5 : 1.5;
  if (dashed) ctx.setLineDash([5, 4]);
  ctx.stroke(); ctx.setLineDash([]);
  if (selected) pts.forEach((p) => handle(toScr(p), color));
  if (num != null) {
    const s = toScr(pts[0]);
    ctx.fillStyle = color; ctx.font = "11px system-ui"; ctx.fillText(String(num), s[0] + 3, s[1] - 3);
  }
}
function handle(s, color) { ctx.fillStyle = "#000"; ctx.fillRect(s[0] - 4, s[1] - 4, 8, 8); ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.strokeRect(s[0] - 4, s[1] - 4, 8, 8); }
function rectPts(a, b) { const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]); return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]; }
function inPoly(p, pts) {
  let c = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}
function hitVertex(sx, sy) {
  // 선택한 박스의 꼭짓점이 먼저, 그다음 위에 그려진 것부터
  const order = [...S.boxes.keys()].reverse().sort((a, b) => (b === S.sel) - (a === S.sel));
  for (const bi of order) {
    const pts = S.boxes[bi].pts;
    for (let k = 0; k < 4; k++) { const s = toScr(pts[k]); if (Math.abs(s[0] - sx) <= 7 && Math.abs(s[1] - sy) <= 7) return [bi, k]; }
  }
  return null;
}
function pushHistory() { S.history.push(JSON.stringify(S.boxes)); if (S.history.length > 100) S.history.shift(); }
function changed() { S.dirty = true; draw(); updateStatus(); }
function undo() { if (!S.history.length) return; S.boxes = JSON.parse(S.history.pop()); S.sel = -1; changed(); }

const evPos = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
cv.addEventListener("contextmenu", (e) => e.preventDefault());
cv.addEventListener("mousedown", (e) => {
  if (!S.img) return;
  const [sx, sy] = evPos(e), p = toImg(sx, sy);
  if (e.button === 1 || e.button === 2 || S.space) {
    S.drag = { kind: "pan", sx, sy, ox: S.tf.ox, oy: S.tf.oy }; e.preventDefault(); return;
  }
  if (e.button !== 0) return;
  if (S.quadMode) {
    S.quad.push(p);
    if (S.quad.length === 4) { pushHistory(); S.boxes.push({ cls: S.cls, pts: S.quad }); S.sel = S.boxes.length - 1; S.quad = []; changed(); }
    draw(); return;
  }
  const hv = hitVertex(sx, sy);
  if (hv) { pushHistory(); S.sel = hv[0]; S.drag = { kind: "vertex", bi: hv[0], k: hv[1], moved: false }; draw(); return; }
  for (let i = S.boxes.length - 1; i >= 0; i--) {
    if (inPoly(p, S.boxes[i].pts)) {
      pushHistory(); S.sel = i;
      S.drag = { kind: "move", bi: i, last: p, moved: false }; draw(); updateStatus(); return;
    }
  }
  S.sel = -1;
  S.drag = { kind: "rect", start: p, now: p };
  draw(); updateStatus();
});
window.addEventListener("mousemove", (e) => {
  if (!S.img) return;
  const [sx, sy] = evPos(e);
  const r = cv.getBoundingClientRect();
  S.mouse = sx >= 0 && sy >= 0 && sx <= r.width && sy <= r.height ? [sx, sy] : null;
  const p = toImg(sx, sy);
  if (S.mouse) $("s-pos").textContent = "(" + p[0].toFixed(1) + ", " + p[1].toFixed(1) + ") · 확대 " + S.tf.s.toFixed(2);
  const d = S.drag;
  if (!d) { if (S.mouse || S.quadMode) draw(); return; }
  if (d.kind === "pan") { S.tf.ox = d.ox + sx - d.sx; S.tf.oy = d.oy + sy - d.sy; }
  else if (d.kind === "vertex") { S.boxes[d.bi].pts[d.k] = p; d.moved = true; S.dirty = true; }
  else if (d.kind === "move") {
    const dx = p[0] - d.last[0], dy = p[1] - d.last[1];
    S.boxes[d.bi].pts = S.boxes[d.bi].pts.map((q2) => [q2[0] + dx, q2[1] + dy]); d.last = p; d.moved = true; S.dirty = true;
  } else if (d.kind === "rect") d.now = p;
  draw();
});
window.addEventListener("mouseup", () => {
  const d = S.drag;
  if (!d) return;
  S.drag = null;
  if (d.kind === "rect") {
    const w = Math.abs(d.now[0] - d.start[0]), h = Math.abs(d.now[1] - d.start[1]);
    if (w >= 3 && h >= 3) { pushHistory(); S.boxes.push({ cls: S.cls, pts: rectPts(d.start, d.now) }); S.sel = S.boxes.length - 1; changed(); }
  } else if ((d.kind === "vertex" || d.kind === "move") && !d.moved) S.history.pop(); // 누르기만 한 것은 되돌리기 목록에 안 남긴다
  else if (d.kind !== "pan") changed();
  draw(); updateStatus();
});
cv.addEventListener("mouseleave", () => { S.mouse = null; draw(); });
cv.addEventListener("wheel", (e) => {
  if (!S.img) return;
  e.preventDefault();
  const [sx, sy] = evPos(e), p = toImg(sx, sy);
  const s = Math.min(20, Math.max(0.05, S.tf.s * Math.pow(1.0015, -e.deltaY)));
  S.tf = { s, ox: sx - p[0] * s, oy: sy - p[1] * s };
  draw();
}, { passive: false });

function rotateSel(deg) {
  if (S.sel < 0) return;
  const pts = S.boxes[S.sel].pts, cx = pts.reduce((a, p) => a + p[0], 0) / 4, cy = pts.reduce((a, p) => a + p[1], 0) / 4;
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  pushHistory();
  S.boxes[S.sel].pts = pts.map(([x, y]) => [cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c]);
  changed();
}
function nudgeSel(dx, dy) { pushHistory(); S.boxes[S.sel].pts = S.boxes[S.sel].pts.map(([x, y]) => [x + dx, y + dy]); changed(); }

// ───────────────────────── 키보드
window.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea")) return;
  if ($("tab-label").classList.contains("hidden")) return;
  const k = e.key;
  if (k === " ") { S.space = true; e.preventDefault(); return; }
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "z") { undo(); e.preventDefault(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const n = parseInt(k, 10);
  if (n >= 1 && n <= S.classes.length) { setClass(S.classes[n - 1].id); return; }
  if (e.code === "BracketLeft") { rotateSel(e.shiftKey ? -5 : -1); e.preventDefault(); return; }
  if (e.code === "BracketRight") { rotateSel(e.shiftKey ? 5 : 1); e.preventDefault(); return; }
  if (S.sel >= 0 && k.startsWith("Arrow")) {
    const st = e.shiftKey ? 5 : 1;
    nudgeSel(k === "ArrowLeft" ? -st : k === "ArrowRight" ? st : 0, k === "ArrowUp" ? -st : k === "ArrowDown" ? st : 0);
    e.preventDefault(); return;
  }
  switch (k.toLowerCase()) {
    case "q": toggleQuad(); break;
    case "escape": S.quad = []; S.sel = -1; if (S.quadMode) toggleQuad(); draw(); updateStatus(); break;
    case "delete": case "backspace":
      if (S.sel >= 0) { pushHistory(); S.boxes.splice(S.sel, 1); S.sel = -1; changed(); }
      e.preventDefault(); break;
    case "c": copyPrev(false); break;
    case "a": case "arrowleft": next(-1); break;
    case "d": case "arrowright": next(1); break;
    case "enter": markDone(); e.preventDefault(); break;
    case "x": markSkip(); break;
    case "f": fit(); break;
    default: return;
  }
});
window.addEventListener("keyup", (e) => { if (e.key === " ") S.space = false; });
function toggleQuad() { S.quadMode = !S.quadMode; S.quad = []; $("t-quad").classList.toggle("on", S.quadMode); draw(); }

$("t-quad").addEventListener("click", toggleQuad);
$("t-copy").addEventListener("click", () => copyPrev(false));
$("t-prev").addEventListener("click", () => next(-1));
$("t-next").addEventListener("click", () => next(1));
$("t-done").addEventListener("click", markDone);
$("t-skip").addEventListener("click", markSkip);
$("t-fit").addEventListener("click", fit);

function setState(msg) { $("s-state").textContent = msg; }
function updateStatus() {
  if (!S.file) { $("s-file").textContent = ""; return; }
  const names = { done: "완료", draft: "작성 중", skip: "건너뜀" };
  $("s-file").textContent = S.variant + " / " + S.file;
  setState((names[S.labStatus] || "라벨 없음") + (S.dirty ? " (저장 안 됨)" : "") + " · 박스 " + S.boxes.length +
    (S.sel >= 0 ? " · 선택 #" + (S.sel + 1) : "") + (S.quadMode ? " · 네 점 모드 " + S.quad.length + "/4" : "") +
    (S.note ? " · " + S.note : ""));
}
window.addEventListener("beforeunload", (e) => { if (S.dirty) { e.preventDefault(); e.returnValue = ""; } });

// ───────────────────────── 전처리 · 내보내기
$("p-run").addEventListener("click", async () => {
  if (!S.session) return;
  try { await post("/api/preprocess", { session: S.session, mode: $("p-mode").value }); } catch (e) { alert(e.message); }
});
let prepWasRunning = false;
function renderPrep(p) {
  if (!p) return;
  $("p-log").textContent = "[" + p.session + " · " + p.mode + (p.watch ? " · 수집 따라가는 중" : "") + "] " +
    (p.running ? "도는 중" : "끝 (코드 " + p.code + ")") + "\n" + p.log.join("\n");
  $("p-run").disabled = p.running;
  if (prepWasRunning && !p.running) loadSessions();
  prepWasRunning = p.running;
}
$("e-run").addEventListener("click", async () => {
  await leaveImage();
  $("e-log").textContent = "내보내는 중…";
  try {
    const r = await post("/api/export", { variant: S.variant, val_ratio: parseFloat($("e-val").value) || 0,
      sessions: $("e-all").checked ? null : [S.session] });
    $("e-log").textContent = "학습 " + r.counts.train + " · 검증 " + r.counts.val + " 장\n박스 " + JSON.stringify(r.boxes) +
      "\n" + r.out + (r.errors.length ? "\n오류: " + r.errors.join("; ") : "");
  } catch (e) { $("e-log").textContent = "실패: " + e.message; }
});

pollStatus();
showTab(location.hash === "#label" ? "label" : "collect");
loadSessions();
