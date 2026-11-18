"""
Direct in-memory extractor & interactive HTML Map Viewer builder for Hungry Worm - Greedy Worm.
Reads C:\\Users\\Lenovo\\Downloads\\Hungry+Worm+-+Greedy+Worm_1.0.9_APKPure\\com.funspace.hungryworm.zip
without extracting the archive.
"""

import json
import os
import struct
import zipfile

ZIP_PATH = r"C:\Users\Lenovo\Downloads\Hungry+Worm+-+Greedy+Worm_1.0.9_APKPure\com.funspace.hungryworm.zip"
OUT_DIR = r"C:\Su\tools\Attack_law\test"

GUID_WORM_CONFIGS = "assets/bin/Data/534dec325a9cf4fcd9424b457df291bd"
GUID_LEVEL_CONFIGS = "assets/bin/Data/c6056ca8882e7456981441771709c2be"
GUID_LEVEL_CONFIGS_B = "assets/bin/Data/382f8ba8f4b404e5d9ac6ae19951ee4b"


def extract_textasset_json(raw: bytes, asset_name: bytes):
    idx = raw.find(asset_name)
    if idx < 4:
        raise ValueError(f"Asset name {asset_name!r} not found")
    name_len = struct.unpack_from("<I", raw, idx - 4)[0]
    after_name = idx + name_len
    if after_name % 4 != 0:
        after_name += 4 - (after_name % 4)
    txt_len = struct.unpack_from("<I", raw, after_name)[0]
    txt = raw[after_name + 4 : after_name + 4 + txt_len].decode("utf-8")
    return json.loads(txt)


HTML_HEAD = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Hungry Worm - Map Viewer (240 Màn Chơi)</title>
<style>
  :root { --bg:#0f172a; --panel:#1e293b; --border:#334155; --text:#f8fafc; --muted:#94a3b8; --accent:#22c55e; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--text); display:flex; height:100vh; overflow:hidden; }
  .sidebar { width:280px; background:var(--panel); border-right:1px solid var(--border); display:flex; flex-direction:column; padding:14px; gap:10px; }
  .brand { font-size:18px; font-weight:700; color:#4ade80; }
  .pack-tabs { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
  .tab-btn { padding:8px; background:#0f172a; border:1px solid var(--border); color:var(--muted); border-radius:8px; cursor:pointer; font-weight:600; font-size:13px; }
  .tab-btn.active { background:var(--accent); color:#052e16; border-color:var(--accent); }
  .search-input { width:100%; padding:8px 12px; border-radius:8px; border:1px solid var(--border); background:#0f172a; color:var(--text); font-size:14px; }
  .level-grid { flex:1; overflow-y:auto; display:grid; grid-template-columns:repeat(4,1fr); gap:6px; align-content:start; padding-right:4px; }
  .lvl-btn { padding:8px 4px; background:#0f172a; border:1px solid var(--border); color:var(--text); border-radius:6px; cursor:pointer; font-weight:600; font-size:13px; }
  .lvl-btn:hover { border-color:#4ade80; }
  .lvl-btn.active { background:#22c55e; color:#052e16; border-color:#22c55e; }
  .main { flex:1; display:flex; flex-direction:column; padding:14px 20px; gap:12px; overflow-y:auto; }
  .topbar { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; background:var(--panel); padding:12px 16px; border-radius:12px; border:1px solid var(--border); }
  .title-group h1 { font-size:20px; }
  .title-group p { font-size:13px; color:var(--muted); margin-top:2px; }
  .controls { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
  .btn { padding:8px 14px; border-radius:8px; border:1px solid var(--border); background:#0f172a; color:var(--text); cursor:pointer; font-weight:600; font-size:13px; }
  .btn.primary { background:#2563eb; border-color:#3b82f6; }
  .btn.primary.on { background:#f59e0b; color:#1e1b4b; border-color:#fbbf24; }
  .canvas-wrap { flex:1; display:flex; align-items:center; justify-content:center; background:linear-gradient(180deg,#bae6fd 0%,#e0f2fe 65%,#bbf7d0 100%); border-radius:16px; border:2px solid var(--border); padding:10px; min-height:420px; }
  canvas { max-width:100%; max-height:100%; border-radius:8px; }
  .legend { display:flex; flex-wrap:wrap; gap:14px; background:var(--panel); padding:10px 16px; border-radius:10px; border:1px solid var(--border); font-size:13px; align-items:center; }
  .legend-item { display:flex; align-items:center; gap:6px; }
  .dot { width:14px; height:14px; border-radius:4px; display:inline-block; }
  .hints-bar { background:var(--panel); padding:10px 16px; border-radius:10px; border:1px solid var(--border); font-size:13px; color:#fde68a; display:none; line-height:1.6; }
  .hints-bar.visible { display:block; }
</style>
</head>
"""


HTML_BODY = """<body>
  <aside class="sidebar">
    <div class="brand">🐛 Hungry Worm Maps</div>
    <div class="pack-tabs">
      <button id="tabA" class="tab-btn active" onclick="switchPack('A')">Chính (210)</button>
      <button id="tabB" class="tab-btn" onclick="switchPack('B')">Bộ B (30)</button>
    </div>
    <input id="searchBox" class="search-input" type="number" min="1" max="210" placeholder="Nhập số màn (1 - 210)..." oninput="onSearch(this.value)">
    <div id="levelGrid" class="level-grid"></div>
  </aside>
  <main class="main">
    <div class="topbar">
      <div class="title-group">
        <h1 id="mapTitle">Level 1</h1>
        <p id="mapStats">Kích thước: 22 × 12</p>
      </div>
      <div class="controls">
        <button class="btn" onclick="prevLevel()">⬅️ Màn trước</button>
        <button class="btn" onclick="nextLevel()">Màn tiếp ➡️</button>
        <button id="btnGrid" class="btn" onclick="toggleGrid()"># Toạ độ: BẬT</button>
        <button id="btnHints" class="btn primary" onclick="toggleHints()">💡 Hiện lời giải (Hints)</button>
      </div>
    </div>
    <div class="canvas-wrap"><canvas id="mapCanvas" width="1100" height="600"></canvas></div>
    <div id="hintsBar" class="hints-bar"></div>
    <div class="legend">
      <strong>Chú giải Map:</strong>
      <span class="legend-item"><span class="dot" style="background:#854d0e;border:2px solid #22c55e"></span> 1: Đất/Cỏ</span>
      <span class="legend-item"><span class="dot" style="background:#64748b"></span> 2: Đá rơi</span>
      <span class="legend-item"><span class="dot" style="background:#ef4444"></span> 4: Táo đỏ</span>
      <span class="legend-item"><span class="dot" style="background:#16a34a"></span> 5-6: Đầu & Thân Sâu</span>
      <span class="legend-item"><span class="dot" style="background:#f97316"></span> 3/9/10/11: Gai nhọn</span>
      <span class="legend-item"><span class="dot" style="background:#334155"></span> 7: Bánh răng cưa</span>
      <span class="legend-item"><span class="dot" style="background:#7c3aed"></span> 8: Cổng đích (Hố đen)</span>
    </div>
  </main>
<script>
const DATA = __DATA_JSON__;
let curPack = 'A';
let curIdx = 0;
let showGrid = true;
let showHints = false;

function getLevels() { return curPack === 'A' ? DATA.levelsA : DATA.levelsB; }

function switchPack(p) {
  curPack = p;
  curIdx = 0;
  document.getElementById('tabA').classList.toggle('active', p === 'A');
  document.getElementById('tabB').classList.toggle('active', p === 'B');
  document.getElementById('searchBox').max = getLevels().length;
  renderSidebar();
  renderLevel();
}

function renderSidebar() {
  const grid = document.getElementById('levelGrid');
  const lvls = getLevels();
  grid.innerHTML = '';
  lvls.forEach((lv, i) => {
    const b = document.createElement('button');
    b.className = 'lvl-btn' + (i === curIdx ? ' active' : '');
    b.textContent = lv.id;
    b.onclick = () => { curIdx = i; updateActiveBtn(); renderLevel(); };
    grid.appendChild(b);
  });
}

function updateActiveBtn() {
  const btns = document.querySelectorAll('.lvl-btn');
  btns.forEach((b, i) => b.classList.toggle('active', i === curIdx));
  if (btns[curIdx]) btns[curIdx].scrollIntoView({ block: 'nearest' });
}

function onSearch(val) {
  const num = parseInt(val, 10);
  const lvls = getLevels();
  const idx = lvls.findIndex(l => l.id === num);
  if (idx !== -1) { curIdx = idx; updateActiveBtn(); renderLevel(); }
}

function prevLevel() {
  const lvls = getLevels();
  curIdx = (curIdx - 1 + lvls.length) % lvls.length;
  updateActiveBtn(); renderLevel();
}

function nextLevel() {
  const lvls = getLevels();
  curIdx = (curIdx + 1) % lvls.length;
  updateActiveBtn(); renderLevel();
}

function toggleGrid() {
  showGrid = !showGrid;
  document.getElementById('btnGrid').textContent = '# Toạ độ: ' + (showGrid ? 'BẬT' : 'TẮT');
  renderLevel();
}

function toggleHints() {
  showHints = !showHints;
  const btn = document.getElementById('btnHints');
  btn.classList.toggle('on', showHints);
  btn.textContent = showHints ? '💡 Đang hiện lời giải' : '💡 Hiện lời giải (Hints)';
  document.getElementById('hintsBar').classList.toggle('visible', showHints);
  renderLevel();
}
"""


HTML_DRAW_JS = """
function drawThorn(ctx, px, py, s, rotDeg) {
  ctx.save();
  ctx.translate(px + s/2, py + s/2);
  ctx.rotate((rotDeg * Math.PI) / 180);
  ctx.fillStyle = '#ea580c';
  ctx.strokeStyle = '#7c2d12';
  ctx.lineWidth = 2;
  for (let k = -1; k <= 1; k += 2) {
    ctx.beginPath();
    ctx.moveTo(k * s * 0.22 - s * 0.2, s * 0.45);
    ctx.lineTo(k * s * 0.22, -s * 0.35);
    ctx.lineTo(k * s * 0.22 + s * 0.2, s * 0.45);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawGear(ctx, px, py, s) {
  const cx = px + s/2, cy = py + s/2, r = s * 0.36;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.fillStyle = '#475569';
  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 2;
  for (let i = 0; i < 8; i++) {
    ctx.rotate(Math.PI / 4);
    ctx.fillRect(-4, -r - 5, 8, 8);
  }
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = '#cbd5e1';
  ctx.beginPath();
  ctx.arc(0, 0, r * 0.4, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function renderLevel() {
  const lv = getLevels()[curIdx];
  const W = lv.width || 22;
  const H = lv.height || 12;
  const canvas = document.getElementById('mapCanvas');
  const ctx = canvas.getContext('2d');
  const S = 50;
  canvas.width = W * S;
  canvas.height = H * S;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (showGrid) {
    ctx.strokeStyle = 'rgba(15, 23, 42, 0.10)';
    ctx.lineWidth = 1;
    ctx.font = '10px Segoe UI';
    ctx.fillStyle = 'rgba(15, 23, 42, 0.42)';
    for (let x = 0; x < W; x++) {
      for (let y = 0; y < H; y++) {
        const px = x * S, py = (H - 1 - y) * S;
        ctx.strokeRect(px, py, S, S);
        if (y === 0) ctx.fillText(x, px + 4, py + S - 4);
        if (x === 0) ctx.fillText('y=' + y, px + 3, py + 12);
      }
    }
  }

  const wallSet = new Set();
  lv.bricks.forEach(b => { if (b.id === 1) wallSet.add(b.x + ',' + b.y); });

  let apples = 0, stones = 0, hazards = 0;

  lv.bricks.forEach(b => {
    const px = b.x * S;
    const py = (H - 1 - b.y) * S;
    if (b.id === 1) {
      ctx.fillStyle = '#854d0e';
      ctx.fillRect(px + 1, py + 1, S - 2, S - 2);
      ctx.fillStyle = '#a16207';
      ctx.fillRect(px + 6, py + 10, S - 12, S - 16);
      if (!wallSet.has(b.x + ',' + (b.y + 1))) {
        ctx.fillStyle = '#22c55e';
        ctx.fillRect(px + 1, py + 1, S - 2, 10);
      }
    } else if (b.id === 2) {
      stones++;
      ctx.fillStyle = '#64748b';
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.roundRect(px + 5, py + 5, S - 10, S - 10, 12);
      ctx.fill(); ctx.stroke();
    } else if (b.id === 4) {
      apples++;
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(px + S/2, py + S/2 + 3, S * 0.34, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#16a34a';
      ctx.fillRect(px + S/2 - 2, py + 6, 4, 9);
      ctx.beginPath();
      ctx.ellipse(px + S/2 + 7, py + 10, 7, 4, -0.4, 0, Math.PI * 2);
      ctx.fill();
    } else if (b.id === 5 || b.id === 6) {
      ctx.fillStyle = b.id === 5 ? '#15803d' : '#22c55e';
      ctx.strokeStyle = '#052e16';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.roundRect(px + 4, py + 4, S - 8, S - 8, b.id === 5 ? 16 : 12);
      ctx.fill(); ctx.stroke();
      if (b.id === 5) {
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(px + S * 0.36, py + S * 0.38, 5, 0, Math.PI * 2);
        ctx.arc(px + S * 0.66, py + S * 0.38, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#0f172a';
        ctx.beginPath();
        ctx.arc(px + S * 0.38, py + S * 0.38, 2.5, 0, Math.PI * 2);
        ctx.arc(px + S * 0.68, py + S * 0.38, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    } else if (b.id === 7) {
      hazards++;
      drawGear(ctx, px, py, S);
    } else if (b.id === 8) {
      const cx = px + S/2, cy = py + S/2;
      const grad = ctx.createRadialGradient(cx, cy, 3, cx, cy, S * 0.44);
      grad.addColorStop(0, '#0f172a');
      grad.addColorStop(0.6, '#6d28d9');
      grad.addColorStop(1, '#c084fc');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, S * 0.42, 0, Math.PI * 2);
      ctx.fill();
    } else if (b.id === 3 || b.id === 9 || b.id === 10 || b.id === 11) {
      hazards++;
      const rotMap = { 3: 0, 9: -90, 10: 180, 11: 90 };
      drawThorn(ctx, px, py, S, rotMap[b.id] || 0);
    }
  });
"""

HTML_TAIL = """
  const hints = lv.hints || [];
  if (showHints && hints.length > 0) {
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.92)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    hints.forEach((h, idx) => {
      const cx = h.x * S + S/2;
      const cy = (H - 1 - h.y) * S + S/2;
      if (idx === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.stroke();

    const dirArrows = { 1: '←', 2: '→', 3: '↑', 4: '↓' };
    hints.forEach((h, idx) => {
      const cx = h.x * S + S/2;
      const cy = (H - 1 - h.y) * S + S/2;
      ctx.fillStyle = '#fef08a';
      ctx.strokeStyle = '#78350f';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 11, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#451a03';
      ctx.font = 'bold 10px Segoe UI';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText((idx + 1) + (dirArrows[h.d] || ''), cx, cy);
    });
    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';

    const dirWords = { 1: '⬅️Trái', 2: '➡️Phải', 3: '⬆️Lên', 4: '⬇️Xuống' };
    document.getElementById('hintsBar').innerHTML =
      '<strong>Đáp án (' + hints.length + ' bước):</strong> ' +
      hints.map((h, i) => (i + 1) + '.' + (dirWords[h.d] || '?')).join(' &nbsp;→&nbsp; ');
  } else if (showHints) {
    document.getElementById('hintsBar').innerHTML = 'Màn này không có dữ liệu gợi ý.';
  }

  document.getElementById('mapTitle').textContent =
    (curPack === 'A' ? 'Màn Chính — Level ' : 'Bộ B — Level ') + lv.id;
  document.getElementById('mapStats').textContent =
    `Lưới: ${W}×${H} | Tổng vật thể: ${lv.bricks.length} | 🍎 Táo: ${apples} | 🪨 Đá: ${stones} | ⚠️ Bẫy: ${hazards} | 💡 Bước giải: ${hints.length}`;
}

window.addEventListener('keydown', (e) => {
  if (document.activeElement.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') prevLevel();
  else if (e.key === 'ArrowRight') nextLevel();
});

renderSidebar();
renderLevel();
</script>
</body>
</html>
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        worm_elements = extract_textasset_json(zf.read(GUID_WORM_CONFIGS), b"WormConfigs")
        levels_a = extract_textasset_json(zf.read(GUID_LEVEL_CONFIGS), b"LevelConfigs")
        levels_b = extract_textasset_json(zf.read(GUID_LEVEL_CONFIGS_B), b"LevelConfigs_B")

    elements_path = os.path.join(OUT_DIR, "hungry_worm_elements.json")
    levels_path = os.path.join(OUT_DIR, "hungry_worm_levels.json")
    viewer_path = os.path.join(OUT_DIR, "hungry_worm_map_viewer.html")

    with open(elements_path, "w", encoding="utf-8") as f:
        json.dump(worm_elements, f, ensure_ascii=False, indent=2)

    full_data = {
        "elements": worm_elements,
        "levelsA": levels_a,
        "levelsB": levels_b,
    }
    with open(levels_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)

    compact_json = json.dumps(full_data, ensure_ascii=False, separators=(",", ":"))
    full_html = (HTML_HEAD + HTML_BODY + HTML_DRAW_JS + HTML_TAIL).replace(
        "__DATA_JSON__", compact_json
    )
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Exported elements ({len(worm_elements)} items): {elements_path}")
    print(f"Exported levels JSON ({len(levels_a)} + {len(levels_b)} maps): {levels_path}")
    print(f"Built interactive HTML Map Viewer: {viewer_path} ({os.path.getsize(viewer_path):,} bytes)")


if __name__ == "__main__":
    main()



