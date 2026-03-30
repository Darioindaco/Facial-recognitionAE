#!/usr/bin/env python3
"""
Face Detection & Lower Third CSV Generator for Broadcast Post-Production.

Pipeline:
1. Extract frames from proxy video (H264)
2. Detect faces using InsightFace (fully local)
3. Cluster faces per person across shots
4. Label each person with name and title
5. Output CSV: name, title, tc_in, tc_out
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class FaceDetection:
    """A single face detection in a specific frame."""
    frame_number: int
    embedding: np.ndarray
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float


@dataclass
class PersonCluster:
    """A cluster of face detections belonging to one person."""
    cluster_id: int
    detections: list = field(default_factory=list)
    name: str = ""
    title: str = ""

    @property
    def frame_numbers(self) -> list:
        return sorted(set(d.frame_number for d in self.detections))


@dataclass
class TimecodeSegment:
    """A continuous timecode range where a person is visible."""
    name: str
    title: str
    tc_in: str
    tc_out: str


# ─── Timecode utilities ─────────────────────────────────────────────────────

def frame_to_timecode(frame_number: int, fps: float) -> str:
    """Convert frame number to SMPTE timecode HH:MM:SS:FF."""
    total_frames = int(frame_number)
    ff = total_frames % round(fps)
    total_seconds = total_frames // round(fps)
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


# ─── Core pipeline ──────────────────────────────────────────────────────────

class FaceDetectorPipeline:
    def __init__(
        self,
        video_path: str,
        sample_interval: int = 12,
        detection_threshold: float = 0.5,
        cluster_eps: float = 0.65,
        cluster_min_samples: int = 2,
        min_segment_frames: int = 1,
        lower_third_duration: float = 5.0,
    ):
        self.video_path = video_path
        self.sample_interval = sample_interval
        self.detection_threshold = detection_threshold
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.min_segment_frames = min_segment_frames
        self.lower_third_duration = lower_third_duration

        self.detections: list[FaceDetection] = []
        self.clusters: list[PersonCluster] = []
        self.segments: list[TimecodeSegment] = []

        self.fps: float = 25.0
        self.total_frames: int = 0

        # Initialize InsightFace
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def run(self) -> str:
        """Run the full pipeline. Returns path to output CSV."""
        print(f"\n{'='*60}")
        print(f"  Face Detection Lower Third Generator")
        print(f"{'='*60}")

        self._extract_and_detect()
        self._cluster_faces()
        self._label_clusters()
        self._build_segments()
        output_path = self._write_csv()

        print(f"\n{'='*60}")
        print(f"  Done! CSV written to: {output_path}")
        print(f"{'='*60}\n")
        return output_path

    # ── Step 1: Frame extraction + face detection ────────────────────────

    def _extract_and_detect(self):
        """Extract frames from video and detect faces."""
        print(f"\n[1/4] Extracting frames and detecting faces...")
        print(f"  Video: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            sys.exit(f"Error: Cannot open video file: {self.video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = self.total_frames / self.fps if self.fps > 0 else 0

        print(f"  FPS: {self.fps:.2f} | Frames: {self.total_frames} | Duration: {duration_s:.1f}s")
        print(f"  Sampling every {self.sample_interval} frames ({self.fps / self.sample_interval:.1f} samples/sec)")

        frame_number = 0
        sampled = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % self.sample_interval == 0:
                faces = self.face_app.get(frame)
                for face in faces:
                    if face.det_score < self.detection_threshold:
                        continue
                    if face.embedding is None:
                        continue
                    det = FaceDetection(
                        frame_number=frame_number,
                        embedding=face.embedding,
                        bbox=tuple(face.bbox.astype(int)),
                        confidence=float(face.det_score),
                    )
                    self.detections.append(det)
                sampled += 1

                # Progress
                if sampled % 50 == 0:
                    pct = (frame_number / self.total_frames * 100) if self.total_frames > 0 else 0
                    print(f"  Progress: {pct:.0f}% ({len(self.detections)} faces found)")

            frame_number += 1

        cap.release()
        print(f"  Sampled {sampled} frames, found {len(self.detections)} face detections.")

    # ── Step 2: Cluster faces by person ──────────────────────────────────

    def _cluster_faces(self):
        """Cluster face embeddings to group detections by person."""
        print(f"\n[2/4] Clustering faces by person...")

        if not self.detections:
            print("  No faces detected. Nothing to cluster.")
            return

        embeddings = np.array([d.embedding for d in self.detections])

        # Normalize embeddings for cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms

        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
            metric="cosine",
        ).fit(embeddings_norm)

        labels = clustering.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        for label in sorted(unique_labels):
            cluster = PersonCluster(cluster_id=label)
            for i, det in enumerate(self.detections):
                if labels[i] == label:
                    cluster.detections.append(det)
            self.clusters.append(cluster)

        noise_count = sum(1 for l in labels if l == -1)
        print(f"  Found {len(self.clusters)} unique persons ({noise_count} noise detections discarded).")

        for c in self.clusters:
            frames = c.frame_numbers
            print(f"  - Person {c.cluster_id}: {len(c.detections)} detections across {len(frames)} unique frames")

    # ── Step 3: Label each person ────────────────────────────────────────

    def _label_clusters(self):
        """Ask user to label each person cluster with name and title."""
        print(f"\n[3/4] Label each person (name + title for lower third)")
        print(f"  {len(self.clusters)} person(s) to label.\n")

        if not self.clusters:
            return

        for cluster in self.clusters:
            frames = cluster.frame_numbers
            tc_first = frame_to_timecode(frames[0], self.fps)
            tc_last = frame_to_timecode(frames[-1], self.fps)

            print(f"  Person {cluster.cluster_id}")
            print(f"    Appearances: {len(cluster.detections)} detections")
            print(f"    Time range: {tc_first} - {tc_last}")

            name = input(f"    Name:  ").strip()
            title = input(f"    Title: ").strip()

            if not name:
                name = f"Person_{cluster.cluster_id}"
            cluster.name = name
            cluster.title = title
            print()

    # ── Step 4: Build timecode segments ──────────────────────────────────

    def _build_segments(self):
        """Convert clusters into timecode segments for CSV output.

        Groups consecutive frame detections into continuous segments.
        Uses the first appearance of each person as the lower third trigger,
        with a configurable duration (default 5 seconds).
        """
        print(f"\n[4/4] Building timecode segments...")

        for cluster in self.clusters:
            if not cluster.name:
                continue

            frames = cluster.frame_numbers
            if len(frames) < self.min_segment_frames:
                continue

            # Group into continuous segments (allow gap of up to 2x sample interval)
            max_gap = self.sample_interval * 3
            segments_frames = []
            current_segment = [frames[0]]

            for i in range(1, len(frames)):
                if frames[i] - frames[i - 1] <= max_gap:
                    current_segment.append(frames[i])
                else:
                    segments_frames.append(current_segment)
                    current_segment = [frames[i]]
            segments_frames.append(current_segment)

            # For each segment, generate a lower third at the start
            for seg_frames in segments_frames:
                tc_in_frame = seg_frames[0]
                # Lower third duration in frames
                lt_duration_frames = int(self.lower_third_duration * self.fps)
                tc_out_frame = tc_in_frame + lt_duration_frames

                segment = TimecodeSegment(
                    name=cluster.name,
                    title=cluster.title,
                    tc_in=frame_to_timecode(tc_in_frame, self.fps),
                    tc_out=frame_to_timecode(tc_out_frame, self.fps),
                )
                self.segments.append(segment)

        # Sort by tc_in timecode
        self.segments.sort(key=lambda s: s.tc_in)
        print(f"  Generated {len(self.segments)} lower third segment(s).")

    # ── Step 5: Write CSV ────────────────────────────────────────────────

    def _write_csv(self) -> str:
        """Write segments to CSV file."""
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        output_path = os.path.join("output", f"{base}_lower_thirds.csv")
        os.makedirs("output", exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "title", "tc_in", "tc_out"])
            for seg in self.segments:
                writer.writerow([seg.name, seg.title, seg.tc_in, seg.tc_out])

        print(f"\n  CSV Output ({len(self.segments)} entries):")
        for seg in self.segments:
            print(f"    {seg.name}, {seg.title}, {seg.tc_in}, {seg.tc_out}")

        return output_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect faces in video and generate lower third CSV for After Effects.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python face_detector.py input_video.mp4
  python face_detector.py input_video.mp4 --interval 6 --duration 4.0

Output CSV format (read by After Effects ExtendScript):
  name, title, tc_in, tc_out
  Jan Janssen, Directeur, 01:02:14:08, 01:02:19:08
        """,
    )
    parser.add_argument("video", help="Path to input proxy video (H264)")
    parser.add_argument(
        "--interval", type=int, default=12,
        help="Sample every N frames (default: 12, ~2 samples/sec at 25fps)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Face detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--cluster-eps", type=float, default=0.65,
        help="DBSCAN epsilon for clustering — lower = stricter matching (default: 0.65)",
    )
    parser.add_argument(
        "--cluster-min", type=int, default=2,
        help="Minimum detections to form a cluster (default: 2)",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Lower third display duration in seconds (default: 5.0)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"Error: Video file not found: {args.video}")

    pipeline = FaceDetectorPipeline(
        video_path=args.video,
        sample_interval=args.interval,
        detection_threshold=args.threshold,
        cluster_eps=args.cluster_eps,
        cluster_min_samples=args.cluster_min,
        lower_third_duration=args.duration,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
