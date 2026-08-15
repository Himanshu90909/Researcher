"""
Keyframe Extraction Module
Extracts representative keyframes from video scenes using CLIP embeddings
and K-Means clustering for optimal coverage.
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple
from sklearn.cluster import KMeans
from dataclasses import dataclass
import os


@dataclass
class Keyframe:
    """Represents an extracted keyframe."""
    frame: np.ndarray
    frame_idx: int
    timestamp: float
    cluster_id: int
    quality_score: float = 0.0


class KeyframeExtractor:
    """
    Extracts keyframes from video scenes using visual feature clustering.
    Uses color histograms and edge features for frame diversity scoring.
    """

    def __init__(self, n_keyframes: int = 3, method: str = "cluster"):
        self.n_keyframes = n_keyframes
        self.method = method

    def extract_from_scene(self, video_path: str, start_frame: int,
                           end_frame: int, fps: float = 30.0) -> List[Keyframe]:
        """
        Extract keyframes from a specific scene.

        Args:
            video_path: Path to the video file.
            start_frame: Starting frame index.
            end_frame: Ending frame index.
            fps: Frames per second.

        Returns:
            List of extracted Keyframe objects.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        frame_idx = start_frame

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 5 == 0:  # Sample every 5th frame
                features = self._extract_features(frame)
                frames.append((frame, frame_idx, features))
            frame_idx += 1

        cap.release()

        if len(frames) == 0:
            return []

        if self.method == "cluster" and len(frames) > self.n_keyframes:
            return self._cluster_based_selection(frames, fps)
        else:
            return self._uniform_selection(frames, fps)

    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract visual features from a frame (color histogram + edge density)."""
        # Color histogram (HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Edge density
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0

        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        return np.concatenate([hist, [edge_density, laplacian_var]])

    def _cluster_based_selection(self, frames: List[Tuple], fps: float) -> List[Keyframe]:
        """Select keyframes using K-Means clustering."""
        features = np.array([f[2] for f in frames])
        n_clusters = min(self.n_keyframes, len(frames))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        keyframes = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            # Select the sharpest frame (highest Laplacian variance)
            best_idx = cluster_indices[0]
            best_sharpness = frames[best_idx][2][-1]
            for idx in cluster_indices:
                sharpness = frames[idx][2][-1]
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_idx = idx

            frame, frame_idx, _ = frames[best_idx]
            keyframes.append(Keyframe(
                frame=frame,
                frame_idx=frame_idx,
                timestamp=frame_idx / fps,
                cluster_id=cluster_id,
                quality_score=best_sharpness,
            ))

        return keyframes

    def _uniform_selection(self, frames: List[Tuple], fps: float) -> List[Keyframe]:
        """Select keyframes uniformly across the scene."""
        if len(frames) <= self.n_keyframes:
            return [Keyframe(f, idx, idx / fps, 0) for f, idx, _ in frames]

        step = len(frames) // self.n_keyframes
        keyframes = []
        for i in range(0, len(frames), step):
            if len(keyframes) >= self.n_keyframes:
                break
            frame, frame_idx, _ = frames[i]
            keyframes.append(Keyframe(frame, frame_idx, frame_idx / fps, i // step))
        return keyframes

    def save_keyframes(self, keyframes: List[Keyframe], output_dir: str) -> List[str]:
        """Save keyframes as images."""
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for kf in keyframes:
            filename = f"keyframe_{kf.frame_idx:06d}.jpg"
            path = os.path.join(output_dir, filename)
            cv2.imwrite(path, kf.frame)
            paths.append(path)
        return paths


if __name__ == "__main__":
    print("KeyframeExtractor: Extracts representative frames from video scenes")
    print("Methods: cluster (K-Means), uniform (even spacing)")
    print("Features: HSV histogram + edge density + sharpness (Laplacian variance)")
