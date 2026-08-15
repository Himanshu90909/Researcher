"""
Video Scene Detection Module
Implements temporal scene segmentation using histogram differencing,
shot boundary detection, and scene clustering.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os


@dataclass
class Scene:
    """Represents a detected scene in a video."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    keyframe_path: Optional[str] = None
    description: Optional[str] = None
    similarity_score: float = 0.0


class SceneDetector:
    """
    Detects scene boundaries in video using histogram differencing.
    Implements adaptive thresholding for robust shot boundary detection.
    """

    def __init__(self, threshold: float = 30.0, min_scene_length: int = 15):
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.histogram_bins = 256

    def compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute normalized color histogram for a frame."""
        hist = cv2.calcHist([frame], [0, 1, 2], None,
                           [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def histogram_difference(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compute difference between two histograms using Chi-Square."""
        return cv2.compareHist(hist1.astype(np.float32),
                               hist2.astype(np.float32),
                               cv2.HISTCMP_CHISQR_ALT)

    def detect_scenes(self, video_path: str) -> List[Scene]:
        """
        Detect scene boundaries in a video file.

        Args:
            video_path: Path to the input video file.

        Returns:
            List of detected Scene objects.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        prev_hist = None
        scene_boundaries = [0]
        frame_idx = 0
        differences = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 3 == 0:  # Sample every 3rd frame for efficiency
                hist = self.compute_histogram(frame)
                if prev_hist is not None:
                    diff = self.histogram_difference(prev_hist, hist)
                    differences.append((frame_idx, diff))
                    if diff > self.threshold and (frame_idx - scene_boundaries[-1]) > self.min_scene_length:
                        scene_boundaries.append(frame_idx)
                prev_hist = hist

            frame_idx += 1

        scene_boundaries.append(total_frames)
        cap.release()

        # Create Scene objects
        scenes = []
        for i in range(len(scene_boundaries) - 1):
            start, end = scene_boundaries[i], scene_boundaries[i + 1]
            scenes.append(Scene(
                start_frame=start,
                end_frame=end,
                start_time=start / fps,
                end_time=end / fps,
            ))

        return scenes

    def adaptive_threshold(self, differences: List[Tuple[int, float]]) -> float:
        """Compute adaptive threshold using rolling mean + k*std."""
        values = [d[1] for d in differences]
        if len(values) < 2:
            return self.threshold
        mean = np.mean(values)
        std = np.std(values)
        return mean + 2 * std

    def detect_with_adaptive_threshold(self, video_path: str) -> List[Scene]:
        """Detect scenes using adaptive thresholding."""
        # First pass: collect all differences
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        prev_hist = None
        all_diffs = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 3 == 0:
                hist = self.compute_histogram(frame)
                if prev_hist is not None:
                    diff = self.histogram_difference(prev_hist, hist)
                    all_diffs.append((frame_idx, diff))
                prev_hist = hist
            frame_idx += 1

        cap.release()

        # Adaptive threshold
        threshold = self.adaptive_threshold(all_diffs)
        total_frames = frame_idx
        boundaries = [0]
        for frame_idx, diff in all_diffs:
            if diff > threshold and (frame_idx - boundaries[-1]) > self.min_scene_length:
                boundaries.append(frame_idx)
        boundaries.append(total_frames)

        scenes = []
        for i in range(len(boundaries) - 1):
            scenes.append(Scene(
                start_frame=boundaries[i],
                end_frame=boundaries[i + 1],
                start_time=boundaries[i] / fps,
                end_time=boundaries[i + 1] / fps,
            ))
        return scenes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect scenes in a video")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="scenes/", help="Output directory")
    parser.add_argument("--threshold", type=float, default=30.0, help="Detection threshold")
    args = parser.parse_args()

    detector = SceneDetector(threshold=args.threshold)
    scenes = detector.detect_scenes(args.input)
    print(f"Detected {len(scenes)} scenes:")
    for i, scene in enumerate(scenes):
        print(f"  Scene {i+1}: {scene.start_time:.2f}s - {scene.end_time:.2f}s")
