"""
data_pipeline/step2_label.py
Label your student recordings through a browser UI.
Opens http://localhost:5001

Run: python data_pipeline/step2_label.py --data data/dataset.csv
"""
import argparse, csv, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app  = Flask(__name__)
CORS(app)

DATA_PATH  = "data/dataset.csv"
ROWS       = []
FIELDNAMES = []


def load():
    global ROWS, FIELDNAMES
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        FIELDNAMES = list(reader.fieldnames or [])
        ROWS       = list(reader)
    print(f"Loaded {len(ROWS)} rows  ({sum(1 for r in ROWS if r.get('severity'))} labelled)")


def save():
    with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(ROWS)


@app.route("/")
def index():
    return HTML_UI


@app.route("/api/rows")
def get_rows():
    return jsonify([{
        "index":      i,
        "source":     r.get("source",""),
        "filename":   os.path.basename(r.get("filename","")),
        "transcript": r.get("transcript",""),
        "severity":   r.get("severity",""),
        "speaker_id": r.get("speaker_id",""),
    } for i, r in enumerate(ROWS) if r.get("source") == "student"])


@app.route("/api/save", methods=["POST"])
def save_label():
    body = request.get_json()
    idx  = int(body.get("index", -1))
    if 0 <= idx < len(ROWS):
        ROWS[idx]["severity"] = body.get("severity","")
        save()
        labelled = sum(1 for r in ROWS if r.get("severity"))
        return jsonify({"ok":True, "labelled":labelled, "total":len(ROWS)})
    return jsonify({"ok":False}), 400


@app.route("/api/export")
def export():
    """Export only labelled rows ready for training."""
    labelled = [r for r in ROWS if r.get("severity") in ("mild","moderate","severe")]
    out = "data/training_data.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(labelled)
    return send_file(os.path.abspath(out), as_attachment=True,
                     download_name="training_data.csv", mimetype="text/csv")


@app.route("/api/audio/<int:index>")
def audio(index):
    if 0 <= index < len(ROWS):
        path = ROWS[index].get("filename","")
        if path and os.path.exists(path):
            return send_file(path)
    return "Not found", 404


@app.route("/api/stats")
def stats():
    student = [r for r in ROWS if r.get("source")=="student"]
    labelled = sum(1 for r in student if r.get("severity"))
    counts = {"mild":0,"moderate":0,"severe":0}
    for r in student:
        s = r.get("severity","")
        if s in counts: counts[s]+=1
    return jsonify({"total":len(student),"labelled":labelled,"counts":counts})


HTML_UI = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Label Student Recordings</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f0f;color:#e0e0e0}
.header{background:#1a1a1a;border-bottom:1px solid #2a2a2a;padding:16px 24px;display:flex;align-items:center;gap:16px}
.header h1{font-size:17px;font-weight:600;color:#fff}
.prog-wrap{flex:1}
.prog-bar{height:5px;background:#2a2a2a;border-radius:3px;margin-top:5px;overflow:hidden}
.prog-fill{height:100%;background:#34d399;border-radius:3px;transition:width .3s}
.prog-lbl{font-size:12px;color:#888}
.export-btn{padding:8px 16px;background:#34d399;color:#000;border:none;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer}
.export-btn:hover{background:#2aba86}
.layout{display:flex;height:calc(100vh - 60px)}
.sidebar{width:260px;flex-shrink:0;border-right:1px solid #2a2a2a;overflow-y:auto;background:#141414}
.item{padding:10px 14px;border-bottom:1px solid #1e1e1e;cursor:pointer;transition:background .1s}
.item:hover{background:#1e1e1e}
.item.active{background:#1a2a1a;border-left:3px solid #34d399}
.item-name{font-size:12px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.item-meta{display:flex;align-items:center;gap:6px;margin-top:3px}
.chip{font-size:10px;padding:2px 7px;border-radius:10px;font-weight:500}
.chip-mild{background:#1a3a1a;color:#5db85d}
.chip-moderate{background:#3a2a0a;color:#d4a44a}
.chip-severe{background:#3a1a1a;color:#d46a6a}
.chip-none{background:#2a2a2a;color:#888}
.main{flex:1;overflow-y:auto;padding:20px}
.card{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:18px;margin-bottom:14px}
.card-title{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#555;margin-bottom:10px}
.transcript{font-size:15px;line-height:1.7;font-style:italic;padding:10px 14px;background:#111;border-radius:6px;border-left:3px solid #333}
audio{width:100%;margin-top:8px;filter:invert(1) hue-rotate(180deg)}
.sev-btns{display:flex;gap:8px;flex-wrap:wrap}
.sev-btn{padding:10px 22px;border-radius:8px;border:1.5px solid #333;background:#111;color:#aaa;font-size:14px;font-weight:500;cursor:pointer;transition:all .15s;font-family:inherit}
.sev-btn:hover{border-color:#555;color:#e0e0e0}
.sev-btn.sel-mild{background:#1a3a1a;border-color:#34d399;color:#34d399}
.sev-btn.sel-moderate{background:#3a2a0a;border-color:#d4a44a;color:#d4a44a}
.sev-btn.sel-severe{background:#3a1a1a;border-color:#f87171;color:#f87171}
.save-btn{padding:10px 24px;background:#185FA5;color:#fff;border:none;border-radius:6px;font-size:14px;font-weight:500;cursor:pointer;margin-top:12px;font-family:inherit}
.save-btn:hover{background:#1472be}
.nav-btns{display:flex;gap:8px;margin-top:14px}
.nav-btn{padding:8px 16px;background:#1e1e1e;border:1px solid #2a2a2a;border-radius:6px;color:#aaa;font-size:13px;cursor:pointer;font-family:inherit}
.nav-btn:hover{color:#e0e0e0;border-color:#444}
.hint{font-size:12px;color:#555;margin-top:8px}
.guide{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}
.guide-card{border-radius:8px;padding:10px 12px;font-size:12px;line-height:1.6}
.guide-mild{background:#1a3a1a;border:0.5px solid #34d399;color:#5db85d}
.guide-moderate{background:#3a2a0a;border:0.5px solid #d4a44a;color:#d4a44a}
.guide-severe{background:#3a1a1a;border:0.5px solid #f87171;color:#f87171}
.guide-title{font-weight:600;margin-bottom:4px}
</style>
</head>
<body>
<div class="header">
  <div><h1>Label Student Recordings</h1><div class="prog-lbl" id="prog-lbl">Loading...</div></div>
  <div class="prog-wrap"><div class="prog-bar"><div class="prog-fill" id="prog-fill" style="width:0%"></div></div></div>
  <button class="export-btn" onclick="exportCSV()">Export training CSV</button>
</div>
<div class="layout">
  <div class="sidebar" id="sidebar"></div>
  <div class="main" id="main">
    <div class="guide">
      <div class="guide-card guide-mild"><div class="guide-title">Mild</div>Full sentences, mostly understood, minor pronunciation issues, communicates independently</div>
      <div class="guide-card guide-moderate"><div class="guide-title">Moderate</div>Short phrases (2–4 words), needs repetition, noticeable errors, requires prompting</div>
      <div class="guide-card guide-severe"><div class="guide-title">Severe</div>Single words or sounds only, uses gestures, very hard to understand, minimal functional speech</div>
    </div>
    <div style="font-size:13px;color:#555">Select a recording from the left panel</div>
  </div>
</div>
<script>
let rows=[], current=0;
async function load(){
  const r=await fetch('/api/rows');
  rows=await r.json();
  renderSidebar();
  updateProgress();
  if(rows.length>0) select(0);
}
function renderSidebar(){
  document.getElementById('sidebar').innerHTML=rows.map((r,i)=>{
    const s=r.severity||'';
    const cc=s?'chip-'+s:'chip-none';
    const ct=s||'unlabelled';
    return `<div class="item ${i===current?'active':''}" onclick="select(${i})">
      <div class="item-name">${r.filename}</div>
      <div class="item-meta"><span class="chip ${cc}">${ct}</span></div>
    </div>`;
  }).join('');
}
function select(i){
  current=i;
  renderSidebar();
  const r=rows[i];
  window._sel=r.severity||'';
  document.getElementById('main').innerHTML=`
    <div class="guide">
      <div class="guide-card guide-mild"><div class="guide-title">Mild</div>Full sentences, mostly understood, minor pronunciation issues</div>
      <div class="guide-card guide-moderate"><div class="guide-title">Moderate</div>Short phrases, needs repetition, noticeable speech errors</div>
      <div class="guide-card guide-severe"><div class="guide-title">Severe</div>Single words/sounds, very hard to understand, uses gestures</div>
    </div>
    <div class="card">
      <div class="card-title">Transcription (what they should be saying)</div>
      <div class="transcript">${r.transcript||'(no transcript)'}</div>
      <audio controls src="/api/audio/${r.index}"></audio>
    </div>
    <div class="card">
      <div class="card-title">Select severity label for this student</div>
      <div class="sev-btns">
        <button class="sev-btn ${r.severity==='mild'?'sel-mild':''}" onclick="setSev('mild')">Mild</button>
        <button class="sev-btn ${r.severity==='moderate'?'sel-moderate':''}" onclick="setSev('moderate')">Moderate</button>
        <button class="sev-btn ${r.severity==='severe'?'sel-severe':''}" onclick="setSev('severe')">Severe</button>
        <button class="sev-btn" onclick="setSev('')" style="font-size:12px">Clear</button>
      </div>
      <div class="hint">Listen to the recording and pick based on how clearly the student actually speaks.</div>
      <button class="save-btn" onclick="save()">Save &amp; next</button>
    </div>
    <div class="nav-btns">
      <button class="nav-btn" onclick="nav(-1)">&#8592; Previous</button>
      <button class="nav-btn" onclick="nav(1)">Next &#8594;</button>
      <span style="font-size:12px;color:#555;align-self:center;margin-left:8px">${i+1} of ${rows.length}</span>
    </div>`;
}
function setSev(s){
  window._sel=s;
  rows[current].severity=s;
  document.querySelectorAll('.sev-btn').forEach(b=>{
    b.className='sev-btn';
    if(b.textContent.toLowerCase()===s) b.classList.add('sel-'+s);
  });
  renderSidebar();
}
async function save(){
  const resp=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({index:rows[current].index,severity:window._sel||''})});
  const d=await resp.json();
  if(d.ok){updateProgress(d.labelled,d.total);if(current<rows.length-1)nav(1);}
}
function nav(d){const n=current+d;if(n>=0&&n<rows.length)select(n);}
async function updateProgress(l,t){
  if(!l){const s=await(await fetch('/api/stats')).json();l=s.labelled;t=s.total;}
  const pct=t?Math.round(l/t*100):0;
  document.getElementById('prog-lbl').textContent=`${l}/${t} labelled (${pct}%)`;
  document.getElementById('prog-fill').style.width=pct+'%';
}
async function exportCSV(){
  const r=await fetch('/api/export');
  if(!r.ok){alert('Label at least some recordings first!');return;}
  const b=await r.blob();
  const a=document.createElement('a');
  a.href=URL.createObjectURL(b);a.download='training_data.csv';a.click();
}
load();
</script>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.csv")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    DATA_PATH = args.data
    if not Path(DATA_PATH).exists():
        print(f"File not found: {DATA_PATH}")
        print("Run step1_build_dataset.py first.")
        sys.exit(1)

    load()
    print(f"\nLabelling UI: http://localhost:{args.port}")
    print("Label your student recordings, then click 'Export training CSV'")
    print("Then run: python data_pipeline/step3_train.py\n")
    app.run(debug=False, port=args.port)
