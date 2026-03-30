#!/usr/bin/env python3
"""
Lower Third Generator — Hugging Face Spaces (Gradio)

Face detection & automatic lower third placement for broadcast post-production.
Upload proxy video → Detect faces → Label persons → Download CSV / AVID / ProRes 4444
"""

import csv
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class FaceDetection:
    frame_number: int
    embedding: np.ndarray
    bbox: tuple
    confidence: float


@dataclass
class PersonCluster:
    cluster_id: int
    detections: list = field(default_factory=list)
    name: str = ""
    title: str = ""

    @property
    def frame_numbers(self):
        return sorted(set(d.frame_number for d in self.detections))


@dataclass
class TimecodeSegment:
    name: str
    title: str
    tc_in: str
    tc_out: str
    frame_in: int = 0
    frame_out: int = 0


def frame_to_timecode(frame_number, fps):
    total_frames = int(frame_number)
    ff = total_frames % round(fps)
    total_seconds = total_frames // round(fps)
    ss = total_seconds % 60
    mm = (total_seconds // 60) % 60
    hh = total_seconds // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


# ─── Load model once at startup ─────────────────────────────────────────────

print("Loading InsightFace model...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("Model ready.")


# ─── Pipeline state (per session via gr.State) ──────────────────────────────

def detect_faces(video_path, interval, threshold, cluster_eps, progress=gr.Progress()):
    """Step 1: Detect and cluster faces. Returns state + preview images."""
    if video_path is None:
        raise gr.Error("Please upload a video first.")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Detect
    detections = []
    frame_number = 0
    progress(0, desc="Detecting faces...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % interval == 0:
            faces = face_app.get(frame)
            for face in faces:
                if face.det_score < threshold or face.embedding is None:
                    continue
                detections.append(FaceDetection(
                    frame_number=frame_number,
                    embedding=face.embedding,
                    bbox=tuple(face.bbox.astype(int)),
                    confidence=float(face.det_score),
                ))
            if total_frames > 0:
                progress(frame_number / total_frames, desc=f"Detecting... {len(detections)} faces found")
        frame_number += 1

    cap.release()

    if not detections:
        raise gr.Error("No faces detected. Try lowering the detection threshold.")

    # Cluster
    progress(0.9, desc="Clustering faces...")
    embeddings = np.array([d.embedding for d in detections])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms

    labels = DBSCAN(eps=cluster_eps, min_samples=2, metric="cosine").fit_predict(embeddings_norm)
    unique_labels = sorted(set(labels) - {-1})

    clusters = []
    for label in unique_labels:
        cluster = PersonCluster(cluster_id=label)
        for i, det in enumerate(detections):
            if labels[i] == label:
                cluster.detections.append(det)
        clusters.append(cluster)

    if not clusters:
        raise gr.Error("No person clusters found. Try adjusting cluster sensitivity.")

    # Generate preview crops
    cap = cv2.VideoCapture(video_path)
    previews = []
    preview_info = []

    for cluster in clusters:
        mid_det = cluster.detections[len(cluster.detections) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_det.frame_number)
        ret, frame = cap.read()
        if ret:
            x1, y1, x2, y2 = mid_det.bbox
            pad = 50
            y1, x1 = max(0, y1 - pad), max(0, x1 - pad)
            y2, x2 = min(frame.shape[0], y2 + pad), min(frame.shape[1], x2 + pad)
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            previews.append(Image.fromarray(crop))
        else:
            previews.append(Image.new("RGB", (200, 200), (40, 40, 40)))

        frames = cluster.frame_numbers
        tc_first = frame_to_timecode(frames[0], fps)
        tc_last = frame_to_timecode(frames[-1], fps)
        preview_info.append(f"Person {cluster.cluster_id} | {len(cluster.detections)} detections | {tc_first} – {tc_last}")

    cap.release()
    progress(1.0, desc=f"Found {len(clusters)} person(s)")

    # Store state
    state = {
        "video_path": video_path,
        "fps": fps,
        "w": w,
        "h": h,
        "total_frames": total_frames,
        "clusters": clusters,
    }

    # Build gallery items
    gallery_items = [(img, info) for img, info in zip(previews, preview_info)]

    # Build label inputs info
    info_text = f"**{len(clusters)} person(s) detected.** Enter name and title for each person below."

    return state, gallery_items, info_text


def generate_outputs(
    state, duration, render_prores, export_avid,
    *name_title_pairs,
    progress=gr.Progress(),
):
    """Step 2: Apply labels, build segments, generate outputs."""
    if state is None:
        raise gr.Error("Please detect faces first.")

    clusters = state["clusters"]
    fps = state["fps"]
    w = state["w"]
    h = state["h"]
    total_frames = state["total_frames"]
    video_path = state["video_path"]
    interval = 12

    # Apply labels
    for i, cluster in enumerate(clusters):
        name_idx = i * 2
        title_idx = i * 2 + 1
        if name_idx < len(name_title_pairs):
            cluster.name = name_title_pairs[name_idx] or f"Person_{i}"
        if title_idx < len(name_title_pairs):
            cluster.title = name_title_pairs[title_idx] or ""

    # Build segments
    progress(0.1, desc="Building segments...")
    segments = []
    max_gap = interval * 3

    for cluster in clusters:
        if not cluster.name:
            continue
        frames = cluster.frame_numbers
        if not frames:
            continue

        seg_groups = []
        current = [frames[0]]
        for i in range(1, len(frames)):
            if frames[i] - frames[i - 1] <= max_gap:
                current.append(frames[i])
            else:
                seg_groups.append(current)
                current = [frames[i]]
        seg_groups.append(current)

        for seg_frames in seg_groups:
            tc_in_frame = max(seg_frames[0] - 1, 0)
            lt_duration_frames = int(duration * fps)
            seg_end_frame = seg_frames[-1] + 1
            tc_out_frame = min(tc_in_frame + lt_duration_frames, seg_end_frame)

            segments.append(TimecodeSegment(
                name=cluster.name,
                title=cluster.title,
                tc_in=frame_to_timecode(tc_in_frame, fps),
                tc_out=frame_to_timecode(tc_out_frame, fps),
                frame_in=tc_in_frame,
                frame_out=tc_out_frame,
            ))

    segments.sort(key=lambda s: s.tc_in)

    base = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = tempfile.mkdtemp()
    output_files = []

    # CSV
    progress(0.2, desc="Writing CSV...")
    csv_path = os.path.join(output_dir, f"{base}_lower_thirds.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "title", "tc_in", "tc_out"])
        for seg in segments:
            writer.writerow([seg.name, seg.title, seg.tc_in, seg.tc_out])
    output_files.append(csv_path)

    # AVID
    if export_avid:
        progress(0.3, desc="Writing AVID files...")
        subcap_path = os.path.join(output_dir, f"{base}_lower_thirds.subcap.txt")
        with open(subcap_path, "w") as f:
            f.write(f"@ Generated from {os.path.basename(video_path)}\n\n<begin subtitles>\n")
            for seg in segments:
                f.write(f"{seg.tc_in} {seg.tc_out}\n{seg.name}\n")
                if seg.title:
                    f.write(f"{seg.title}\n")
                f.write("\n")
            f.write("<end subtitles>\n")
        output_files.append(subcap_path)

        marker_path = os.path.join(output_dir, f"{base}_lower_thirds.markers.txt")
        with open(marker_path, "w") as f:
            f.write("Color\tName\tComment\tTC1\tTC2\n")
            for seg in segments:
                comment = f"{seg.name} | {seg.title}" if seg.title else seg.name
                f.write(f"Cyan\t{seg.name}\t{comment}\t{seg.tc_in}\t{seg.tc_out}\n")
        output_files.append(marker_path)

    # ProRes 4444
    if render_prores:
        progress(0.4, desc="Rendering ProRes 4444...")
        render_path = os.path.join(output_dir, f"{base}_lower_thirds.mov")

        margin_x = int(w * 0.10)
        margin_y = int(h * 0.10)
        name_font_size = max(int(h * 0.035), 16)
        title_font_size = max(int(h * 0.025), 12)

        font_paths_bold = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        font_paths_reg = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

        name_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        for fp in font_paths_bold:
            if os.path.isfile(fp):
                name_font = ImageFont.truetype(fp, name_font_size)
                break
        for fp in font_paths_reg:
            if os.path.isfile(fp):
                title_font = ImageFont.truetype(fp, title_font_size)
                break

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-c:v", "prores_ks", "-profile:v", "4444",
            "-pix_fmt", "yuva444p10le", render_path,
        ]
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

        cap = cv2.VideoCapture(video_path)
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            active = [s for s in segments if s.frame_in <= frame_number <= s.frame_out]
            pil_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

            if active:
                draw = ImageDraw.Draw(pil_img, "RGBA")
                for i, seg in enumerate(active):
                    line_height = name_font_size + title_font_size + int(h * 0.015)
                    text_y = h - margin_y - line_height - (i * (line_height + int(h * 0.02)))
                    text_x = margin_x

                    name_bbox = draw.textbbox((0, 0), seg.name, font=name_font)
                    title_bbox = draw.textbbox((0, 0), seg.title, font=title_font) if seg.title else (0, 0, 0, 0)
                    bar_w = max(name_bbox[2] - name_bbox[0], title_bbox[2] - title_bbox[0]) + int(w * 0.04)
                    bar_h = line_height + int(h * 0.015)
                    bar_x = text_x - int(w * 0.015)
                    bar_y = text_y - int(h * 0.008)

                    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(0, 0, 0, 160))
                    draw.text((text_x, text_y), seg.name, font=name_font, fill=(255, 255, 255, 255))
                    if seg.title:
                        title_y = text_y + name_font_size + int(h * 0.005)
                        draw.text((text_x, title_y), seg.title, font=title_font, fill=(200, 200, 200, 255))

            ffmpeg_proc.stdin.write(np.array(pil_img).tobytes())
            frame_number += 1

            if frame_number % 250 == 0 and total_frames > 0:
                pct = 0.4 + (frame_number / total_frames) * 0.55
                progress(pct, desc=f"Rendering: {frame_number / total_frames * 100:.0f}%")

        cap.release()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        output_files.append(render_path)

    progress(1.0, desc="Done!")

    # Summary table
    summary = "| Name | Title | TC In | TC Out |\n|---|---|---|---|\n"
    for seg in segments:
        summary += f"| {seg.name} | {seg.title} | {seg.tc_in} | {seg.tc_out} |\n"

    return output_files, summary


# ─── Gradio UI ───────────────────────────────────────────────────────────────

MAX_PERSONS = 10  # max supported label fields

with gr.Blocks(
    title="Lower Third Generator",
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
    ),
    css="""
        .container { max-width: 960px; margin: auto; }
        footer { display: none !important; }
    """,
) as demo:

    state = gr.State(None)

    gr.Markdown("# Lower Third Generator\nFace detection & automatic lower third placement for broadcast post-production.")

    # ── Step 1: Upload & Detect ──────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="Upload proxy video (H264)")
        with gr.Column(scale=1):
            interval = gr.Slider(1, 48, value=12, step=1, label="Sample interval (frames)")
            threshold = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Detection threshold")
            cluster_eps = gr.Slider(0.1, 2.0, value=0.65, step=0.05, label="Cluster sensitivity (eps)")

    btn_detect = gr.Button("Detect Faces", variant="primary", size="lg")

    # ── Step 2: Preview & Label ──────────────────────────────────────
    info_text = gr.Markdown(visible=False)
    gallery = gr.Gallery(label="Detected Persons", columns=4, height=250, visible=False)

    # Dynamic label inputs (name + title per person, up to MAX_PERSONS)
    name_inputs = []
    title_inputs = []
    label_rows = []

    for i in range(MAX_PERSONS):
        with gr.Row(visible=False) as row:
            name_inp = gr.Textbox(label=f"Person {i} — Name", placeholder="Name")
            title_inp = gr.Textbox(label=f"Person {i} — Title", placeholder="Title / Function")
        name_inputs.append(name_inp)
        title_inputs.append(title_inp)
        label_rows.append(row)

    # ── Step 3: Generate ─────────────────────────────────────────────
    with gr.Row(visible=False) as generate_row:
        with gr.Column():
            duration = gr.Slider(1.0, 30.0, value=5.0, step=0.5, label="Lower third duration (sec)")
        with gr.Column():
            render_prores = gr.Checkbox(value=True, label="Render ProRes 4444 (alpha)")
            export_avid = gr.Checkbox(value=True, label="Export AVID files (SubCap + Markers)")

    btn_generate = gr.Button("Generate Lower Thirds", variant="primary", size="lg", visible=False)

    # ── Step 4: Results ──────────────────────────────────────────────
    summary_output = gr.Markdown(visible=False)
    file_output = gr.Files(label="Download outputs", visible=False)

    # ── Wiring ───────────────────────────────────────────────────────

    def on_detect(video, interval_val, threshold_val, eps_val):
        new_state, gallery_items, info = detect_faces(
            video, int(interval_val), threshold_val, eps_val
        )
        n_clusters = len(new_state["clusters"])

        # Show/hide label rows based on number of clusters
        row_updates = []
        for i in range(MAX_PERSONS):
            row_updates.append(gr.update(visible=(i < n_clusters)))

        return (
            [new_state]
            + [gr.update(value=info, visible=True)]       # info_text
            + [gr.update(value=gallery_items, visible=True)]  # gallery
            + row_updates                                     # label rows
            + [gr.update(visible=True)]                       # generate_row
            + [gr.update(visible=True)]                       # btn_generate
            + [gr.update(visible=False)]                      # summary
            + [gr.update(visible=False)]                      # files
        )

    btn_detect.click(
        fn=on_detect,
        inputs=[video_input, interval, threshold, cluster_eps],
        outputs=(
            [state]
            + [info_text]
            + [gallery]
            + label_rows
            + [generate_row]
            + [btn_generate]
            + [summary_output]
            + [file_output]
        ),
    )

    # Interleave name/title inputs for *args unpacking
    all_label_inputs = []
    for n, t in zip(name_inputs, title_inputs):
        all_label_inputs.append(n)
        all_label_inputs.append(t)

    def on_generate(current_state, dur, render, avid, *name_title_pairs):
        files, summary = generate_outputs(
            current_state, dur, render, avid, *name_title_pairs
        )
        return (
            gr.update(value=summary, visible=True),
            gr.update(value=files, visible=True),
        )

    btn_generate.click(
        fn=on_generate,
        inputs=[state, duration, render_prores, export_avid] + all_label_inputs,
        outputs=[summary_output, file_output],
    )


if __name__ == "__main__":
    demo.launch()
