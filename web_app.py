#!/usr/bin/env python3
"""
Web UI for Face Detection Lower Third Generator.

Upload videos, configure settings, label persons, and download outputs
— all from the browser. No terminal needed.

Usage:
    python web_app.py
    # Open http://localhost:5000
"""

import json
import os
import threading
import uuid

from flask import Flask, jsonify, render_template_string, request, send_from_directory
from werkzeug.utils import secure_filename

from face_detector import FaceDetectorPipeline, frame_to_timecode

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "test_videos"
app.config["OUTPUT_FOLDER"] = "output"
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB max

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# In-memory job state
jobs = {}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lower Third Generator</title>
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0d0d0d;
        color: #e0e0e0;
        min-height: 100vh;
    }
    .container { max-width: 900px; margin: 0 auto; padding: 40px 20px; }
    h1 { font-size: 28px; font-weight: 600; margin-bottom: 8px; color: #fff; }
    .subtitle { color: #888; margin-bottom: 32px; font-size: 14px; }

    /* Cards */
    .card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .card h2 { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: #fff; }

    /* Upload area */
    .upload-area {
        border: 2px dashed #333;
        border-radius: 8px;
        padding: 48px;
        text-align: center;
        cursor: pointer;
        transition: border-color 0.2s, background 0.2s;
    }
    .upload-area:hover, .upload-area.dragover {
        border-color: #4a9eff;
        background: rgba(74, 158, 255, 0.05);
    }
    .upload-area input { display: none; }
    .upload-area .icon { font-size: 36px; margin-bottom: 12px; }
    .upload-area p { color: #888; font-size: 14px; }
    .upload-area .filename { color: #4a9eff; font-weight: 500; margin-top: 8px; }

    /* Settings grid */
    .settings-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }
    .field label {
        display: block;
        font-size: 12px;
        color: #888;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .field input, .field select {
        width: 100%;
        padding: 10px 12px;
        background: #0d0d0d;
        border: 1px solid #333;
        border-radius: 6px;
        color: #fff;
        font-size: 14px;
    }
    .field input:focus, .field select:focus {
        outline: none;
        border-color: #4a9eff;
    }

    /* Checkboxes */
    .checkbox-row {
        display: flex;
        gap: 24px;
        margin-top: 16px;
    }
    .checkbox-row label {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        cursor: pointer;
    }
    .checkbox-row input[type="checkbox"] {
        width: 16px;
        height: 16px;
        accent-color: #4a9eff;
    }

    /* Buttons */
    .btn {
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.2s;
    }
    .btn-primary {
        background: #4a9eff;
        color: #fff;
        width: 100%;
        font-size: 16px;
        padding: 14px;
    }
    .btn-primary:hover { background: #3a8eef; }
    .btn-primary:disabled { background: #333; color: #666; cursor: not-allowed; }
    .btn-download {
        background: #2a2a2a;
        color: #4a9eff;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .btn-download:hover { background: #333; }

    /* Progress */
    .progress-bar {
        width: 100%;
        height: 4px;
        background: #2a2a2a;
        border-radius: 2px;
        overflow: hidden;
        margin-top: 16px;
    }
    .progress-bar .fill {
        height: 100%;
        background: #4a9eff;
        width: 0%;
        transition: width 0.3s;
    }
    .status-text { margin-top: 12px; font-size: 13px; color: #888; }

    /* Labels section */
    .person-label {
        display: grid;
        grid-template-columns: auto 1fr 1fr;
        gap: 12px;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid #222;
    }
    .person-label:last-child { border-bottom: none; }
    .person-info {
        font-size: 12px;
        color: #666;
        min-width: 120px;
    }
    .person-label input {
        padding: 8px 12px;
        background: #0d0d0d;
        border: 1px solid #333;
        border-radius: 6px;
        color: #fff;
        font-size: 14px;
    }
    .person-label input:focus { outline: none; border-color: #4a9eff; }

    /* Downloads */
    .downloads { display: flex; flex-wrap: wrap; gap: 8px; }

    /* Phases */
    .phase { display: none; }
    .phase.active { display: block; }

    /* Spinner */
    .spinner {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid #333;
        border-top-color: #4a9eff;
        border-radius: 50%;
        animation: spin 0.6s linear infinite;
        margin-right: 8px;
        vertical-align: middle;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="container">
    <h1>Lower Third Generator</h1>
    <p class="subtitle">Face detection &amp; automatic lower third placement for broadcast post-production</p>

    <!-- PHASE 1: Upload & Settings -->
    <div id="phase-upload" class="phase active">
        <div class="card">
            <h2>Video</h2>
            <div class="upload-area" id="dropZone" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="video/*">
                <div class="icon">&#9655;</div>
                <p>Drop proxy video here or click to browse</p>
                <p class="filename" id="fileName"></p>
            </div>
        </div>

        <div class="card">
            <h2>Settings</h2>
            <div class="settings-grid">
                <div class="field">
                    <label>Sample interval (frames)</label>
                    <input type="number" id="interval" value="12" min="1" max="100">
                </div>
                <div class="field">
                    <label>Detection threshold</label>
                    <input type="number" id="threshold" value="0.5" min="0.1" max="1.0" step="0.05">
                </div>
                <div class="field">
                    <label>Cluster sensitivity (eps)</label>
                    <input type="number" id="clusterEps" value="0.65" min="0.1" max="2.0" step="0.05">
                </div>
                <div class="field">
                    <label>Lower third duration (sec)</label>
                    <input type="number" id="duration" value="5.0" min="1.0" max="30.0" step="0.5">
                </div>
            </div>
            <div class="checkbox-row">
                <label><input type="checkbox" id="optRender"> Render ProRes 4444 preview</label>
                <label><input type="checkbox" id="optAvid"> Export AVID files (SubCap + Markers)</label>
            </div>
        </div>

        <button class="btn btn-primary" id="btnDetect" disabled onclick="startDetection()">
            Upload &amp; Detect Faces
        </button>
    </div>

    <!-- PHASE 2: Detecting -->
    <div id="phase-detecting" class="phase">
        <div class="card">
            <h2><span class="spinner"></span>Processing...</h2>
            <div class="progress-bar"><div class="fill" id="progressFill"></div></div>
            <p class="status-text" id="statusText">Uploading video...</p>
        </div>
    </div>

    <!-- PHASE 3: Label -->
    <div id="phase-label" class="phase">
        <div class="card">
            <h2>Label each person</h2>
            <p class="subtitle" style="margin-bottom:16px">Enter name and title/function for each detected person</p>
            <div id="labelList"></div>
        </div>
        <button class="btn btn-primary" onclick="submitLabels()">Generate Lower Thirds</button>
    </div>

    <!-- PHASE 4: Generating -->
    <div id="phase-generating" class="phase">
        <div class="card">
            <h2><span class="spinner"></span>Generating outputs...</h2>
            <div class="progress-bar"><div class="fill" id="progressFill2"></div></div>
            <p class="status-text" id="statusText2">Building timecode segments...</p>
        </div>
    </div>

    <!-- PHASE 5: Done -->
    <div id="phase-done" class="phase">
        <div class="card">
            <h2>Done</h2>
            <p class="status-text" id="summaryText" style="margin-bottom:16px"></p>
            <div class="downloads" id="downloadList"></div>
        </div>
        <button class="btn btn-primary" onclick="location.reload()" style="margin-top:16px; background:#2a2a2a;">
            Start new detection
        </button>
    </div>
</div>

<script>
const $ = id => document.getElementById(id);
let selectedFile = null;
let currentJobId = null;

// File selection
$('fileInput').addEventListener('change', e => {
    if (e.target.files.length) {
        selectedFile = e.target.files[0];
        $('fileName').textContent = selectedFile.name + ' (' + (selectedFile.size / 1024 / 1024).toFixed(1) + ' MB)';
        $('btnDetect').disabled = false;
    }
});

// Drag & drop
const dz = $('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        selectedFile = e.dataTransfer.files[0];
        $('fileName').textContent = selectedFile.name + ' (' + (selectedFile.size / 1024 / 1024).toFixed(1) + ' MB)';
        $('btnDetect').disabled = false;
    }
});

function showPhase(name) {
    document.querySelectorAll('.phase').forEach(p => p.classList.remove('active'));
    $('phase-' + name).classList.add('active');
}

async function startDetection() {
    showPhase('detecting');

    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('settings', JSON.stringify({
        interval: parseInt($('interval').value),
        threshold: parseFloat($('threshold').value),
        cluster_eps: parseFloat($('clusterEps').value),
        duration: parseFloat($('duration').value),
        render: $('optRender').checked,
        avid: $('optAvid').checked,
    }));

    try {
        const resp = await fetch('/api/detect', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.error) { alert(data.error); showPhase('upload'); return; }
        currentJobId = data.job_id;
        pollDetection();
    } catch (e) {
        alert('Upload failed: ' + e.message);
        showPhase('upload');
    }
}

async function pollDetection() {
    const resp = await fetch('/api/status/' + currentJobId);
    const data = await resp.json();

    $('progressFill').style.width = data.progress + '%';
    $('statusText').textContent = data.status;

    if (data.phase === 'labeling') {
        showLabelPhase(data.clusters);
    } else if (data.phase === 'error') {
        alert('Error: ' + data.status);
        showPhase('upload');
    } else {
        setTimeout(pollDetection, 500);
    }
}

function showLabelPhase(clusters) {
    const list = $('labelList');
    list.innerHTML = '';
    clusters.forEach((c, i) => {
        const div = document.createElement('div');
        div.className = 'person-label';
        div.innerHTML = `
            <div class="person-info">Person ${i}<br>${c.detections} detections<br>${c.tc_first} – ${c.tc_last}</div>
            <input type="text" placeholder="Name" id="name_${i}" value="">
            <input type="text" placeholder="Title / Function" id="title_${i}" value="">
        `;
        list.appendChild(div);
    });
    showPhase('label');
}

async function submitLabels() {
    showPhase('generating');

    const labels = [];
    const items = document.querySelectorAll('.person-label');
    items.forEach((_, i) => {
        labels.push({
            name: $('name_' + i).value || 'Person ' + i,
            title: $('title_' + i).value || '',
        });
    });

    const resp = await fetch('/api/generate/' + currentJobId, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ labels }),
    });
    const data = await resp.json();
    if (data.error) { alert(data.error); showPhase('label'); return; }

    pollGenerate();
}

async function pollGenerate() {
    const resp = await fetch('/api/status/' + currentJobId);
    const data = await resp.json();

    $('progressFill2').style.width = data.progress + '%';
    $('statusText2').textContent = data.status;

    if (data.phase === 'done') {
        showDonePhase(data.outputs, data.segments);
    } else if (data.phase === 'error') {
        alert('Error: ' + data.status);
        showPhase('label');
    } else {
        setTimeout(pollGenerate, 500);
    }
}

function showDonePhase(outputs, segments) {
    $('summaryText').textContent = segments.length + ' lower third(s) generated';
    const list = $('downloadList');
    list.innerHTML = '';
    outputs.forEach(o => {
        const btn = document.createElement('a');
        btn.className = 'btn btn-download';
        btn.href = '/api/download/' + currentJobId + '/' + o.filename;
        btn.download = o.filename;
        btn.textContent = o.label;
        list.appendChild(btn);
    });
    showPhase('done');
}
</script>
</body>
</html>
"""


# ─── API ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/detect", methods=["POST"])
def api_detect():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]
    settings = json.loads(request.form.get("settings", "{}"))

    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    video.save(video_path)

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "phase": "detecting",
        "progress": 0,
        "status": "Initializing...",
        "video_path": video_path,
        "settings": settings,
        "pipeline": None,
        "clusters": [],
        "outputs": [],
        "segments": [],
    }

    thread = threading.Thread(target=run_detection, args=(job_id,))
    thread.start()

    return jsonify({"job_id": job_id})


def run_detection(job_id):
    job = jobs[job_id]
    settings = job["settings"]

    try:
        pipeline = FaceDetectorPipeline(
            video_path=job["video_path"],
            sample_interval=settings.get("interval", 12),
            detection_threshold=settings.get("threshold", 0.5),
            cluster_eps=settings.get("cluster_eps", 0.65),
            lower_third_duration=settings.get("duration", 5.0),
            render=settings.get("render", False),
            avid=settings.get("avid", False),
        )

        job["status"] = "Detecting faces..."
        job["progress"] = 10
        pipeline._extract_and_detect()

        job["status"] = "Clustering faces..."
        job["progress"] = 70
        pipeline._cluster_faces()

        job["progress"] = 90
        job["status"] = f"Found {len(pipeline.clusters)} person(s)"
        job["pipeline"] = pipeline

        # Prepare cluster info for labeling UI
        cluster_info = []
        for c in pipeline.clusters:
            frames = c.frame_numbers
            cluster_info.append({
                "cluster_id": c.cluster_id,
                "detections": len(c.detections),
                "tc_first": frame_to_timecode(frames[0], pipeline.fps),
                "tc_last": frame_to_timecode(frames[-1], pipeline.fps),
            })

        job["clusters"] = cluster_info
        job["phase"] = "labeling"
        job["progress"] = 100

    except Exception as e:
        job["phase"] = "error"
        job["status"] = str(e)


@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "phase": job["phase"],
        "progress": job["progress"],
        "status": job["status"],
        "clusters": job.get("clusters", []),
        "outputs": job.get("outputs", []),
        "segments": job.get("segments", []),
    })


@app.route("/api/generate/<job_id>", methods=["POST"])
def api_generate(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    data = request.json
    labels = data.get("labels", [])
    pipeline = job["pipeline"]

    # Apply labels to clusters
    for i, cluster in enumerate(pipeline.clusters):
        if i < len(labels):
            cluster.name = labels[i].get("name", f"Person_{i}")
            cluster.title = labels[i].get("title", "")

    job["phase"] = "generating"
    job["progress"] = 0
    job["status"] = "Building segments..."

    thread = threading.Thread(target=run_generate, args=(job_id,))
    thread.start()

    return jsonify({"ok": True})


def run_generate(job_id):
    job = jobs[job_id]
    pipeline = job["pipeline"]

    try:
        job["status"] = "Building timecode segments..."
        job["progress"] = 20
        pipeline._build_segments()

        job["status"] = "Writing CSV..."
        job["progress"] = 40
        csv_path = pipeline._write_csv()

        outputs = [{"filename": os.path.basename(csv_path), "label": "CSV"}]

        if pipeline.avid:
            job["status"] = "Writing AVID files..."
            job["progress"] = 50
            subcap_path = pipeline._write_avid_subcap()
            marker_path = pipeline._write_avid_markers()
            outputs.append({"filename": os.path.basename(subcap_path), "label": "AVID SubCap"})
            outputs.append({"filename": os.path.basename(marker_path), "label": "AVID Markers"})

        if pipeline.render:
            job["status"] = "Rendering ProRes 4444..."
            job["progress"] = 60
            render_path = pipeline._render_video()
            outputs.append({"filename": os.path.basename(render_path), "label": "ProRes 4444 MOV"})

        segments_info = [
            {"name": s.name, "title": s.title, "tc_in": s.tc_in, "tc_out": s.tc_out}
            for s in pipeline.segments
        ]

        job["outputs"] = outputs
        job["segments"] = segments_info
        job["phase"] = "done"
        job["progress"] = 100
        job["status"] = "Complete"

    except Exception as e:
        job["phase"] = "error"
        job["status"] = str(e)


@app.route("/api/download/<job_id>/<filename>")
def api_download(job_id, filename):
    filename = secure_filename(filename)
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)


if __name__ == "__main__":
    print("\n  Lower Third Generator — Web UI")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
