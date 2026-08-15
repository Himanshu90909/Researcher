"""
Automated Video Quality Assessment
Implements temporal, spatial, and perceptual quality metrics
for evaluating AI-generated/enhanced video content.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VideoQualityReport:
    """Comprehensive video quality report."""
    spatial_score: float
    temporal_score: float
    perceptual_score: float
    motion_smoothness: float
    color_consistency: float
    sharpness_score: float
    overall_score: float
    details: Dict


class SpatialQuality:
    """Spatial quality metrics (per-frame)."""
    
    @staticmethod
    def sharpness_score(frame: np.ndarray) -> float:
        """Compute sharpness using Laplacian variance."""
        if frame.ndim == 3:
            frame = frame.mean(axis=2)
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        convolved = _convolve2d(frame, laplacian)
        return float(convolved.var())
    
    @staticmethod
    def color_distribution(frame: np.ndarray) -> Dict:
        """Analyze color distribution."""
        if frame.ndim == 2:
            frame = frame[:, :, np.newaxis]
        
        channel_stats = {}
        for c in range(frame.shape[2]):
            ch = frame[:, :, c]
            channel_stats[f"ch{c}"] = {
                "mean": float(ch.mean()),
                "std": float(ch.std()),
                "dynamic_range": float(ch.max() - ch.min()),
            }
        
        # Colorfulness metric (Hasler & Susstrunk, 2003)
        if frame.shape[2] >= 3:
            rg = frame[:,:,0] - frame[:,:,1]
            yb = 0.5 * (frame[:,:,0] + frame[:,:,1]) - frame[:,:,2]
            colorfulness = np.sqrt(rg.std()**2 + yb.std()**2) + \
                          0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)
            channel_stats["colorfulness"] = float(colorfulness)
        
        return channel_stats
    
    @staticmethod
    def noise_level(frame: np.ndarray) -> float:
        """Estimate noise level using local variance."""
        if frame.ndim == 3:
            frame = frame.mean(axis=2)
        # Use median filter as signal estimate
        from scipy.ndimage import median_filter
        smooth = median_filter(frame, size=3)
        noise = frame - smooth
        return float(noise.std())


class TemporalQuality:
    """Temporal quality metrics (across frames)."""
    
    @staticmethod
    def motion_consistency(frames: List[np.ndarray]) -> float:
        """Check motion consistency across frames."""
        if len(frames) < 3:
            return 1.0
        
        diffs = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            diffs.append(diff)
        
        # Consistency = low variance in frame-to-frame differences
        return float(1.0 / (1.0 + np.std(diffs)))
    
    @staticmethod
    def flicker_detection(frames: List[np.ndarray]) -> float:
        """Detect flickering (rapid brightness changes)."""
        if len(frames) < 2:
            return 0.0
        
        brightness = [f.mean() for f in frames]
        brightness_diff = np.diff(brightness)
        
        # Count rapid changes
        rapid_changes = np.sum(np.abs(brightness_diff) > 2 * np.std(brightness_diff))
        return float(rapid_changes / max(1, len(brightness_diff)))
    
    @staticmethod
    def temporal_stability(frames: List[np.ndarray]) -> float:
        """Measure temporal stability of features."""
        if len(frames) < 2:
            return 1.0
        
        # Feature stability via histogram correlation
        correlations = []
        for i in range(1, len(frames)):
            hist1 = np.histogram(frames[i].flatten(), bins=64)[0]
            hist2 = np.histogram(frames[i-1].flatten(), bins=64)[0]
            corr = np.corrcoef(hist1, hist2)[0, 1]
            correlations.append(corr)
        
        return float(np.mean(correlations))
    
    @staticmethod
    def optical_flow_smoothness(frames: List[np.ndarray]) -> float:
        """Estimate optical flow smoothness (proxy)."""
        if len(frames) < 3:
            return 1.0
        
        flow_magnitudes = []
        for i in range(2, len(frames)):
            flow1 = np.abs(frames[i-1].astype(float) - frames[i-2].astype(float))
            flow2 = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
            flow_diff = np.abs(flow2 - flow1)
            flow_magnitudes.append(np.mean(flow_diff))
        
        # Lower mean flow difference = smoother
        return float(1.0 / (1.0 + np.mean(flow_magnitudes)))


class PerceptualQuality:
    """Perceptual quality metrics."""
    
    @staticmethod
    def niqe_proxy(frame: np.ndarray) -> float:
        """NIQE (Natural Image Quality Evaluator) proxy."""
        if frame.ndim == 3:
            frame = frame.mean(axis=2)
        
        # Local variance distribution
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(frame, size=7)
        local_var = uniform_filter((frame - local_mean)**2, size=7)
        
        # Natural images have specific variance distribution
        var_dist = np.histogram(local_var.flatten(), bins=20)[0]
        var_dist = var_dist / (var_dist.sum() + 1e-8)
        
        # Entropy as quality proxy
        entropy = -np.sum(var_dist * np.log(var_dist + 1e-8))
        return float(entropy)
    
    @staticmethod
    def aesthetic_score(frame: np.ndarray) -> float:
        """Simple aesthetic scoring based on composition rules."""
        if frame.ndim == 3:
            gray = frame.mean(axis=2)
        else:
            gray = frame
        
        h, w = gray.shape
        
        # Rule of thirds: check energy distribution
        thirds_h = [gray[:h//3], gray[h//3:2*h//3], gray[2*h//3:]]
        thirds_w = [gray[:, :w//3], gray[:, w//3:2*w//3], gray[:, 2*w//3:]]
        
        energy_h = [t.var() for t in thirds_h]
        energy_w = [t.var() for t in thirds_w]
        
        # Good composition: balanced energy across thirds
        balance_h = 1.0 / (1.0 + np.std(energy_h) / (np.mean(energy_h) + 1e-8))
        balance_w = 1.0 / (1.0 + np.std(energy_w) / (np.mean(energy_w) + 1e-8))
        
        # Symmetry
        left = gray[:, :w//2]
        right = gray[:, w//2:][:, ::-1]
        symmetry = 1.0 / (1.0 + np.mean(np.abs(left - right)))
        
        return float((balance_h + balance_w + symmetry) / 3)


class VideoQualityAssessment:
    """
    Comprehensive video quality assessment pipeline.
    Combines spatial, temporal, and perceptual metrics.
    """
    
    def __init__(self):
        self.spatial = SpatialQuality()
        self.temporal = TemporalQuality()
        self.perceptual = PerceptualQuality()
    
    def assess(self, frames: List[np.ndarray]) -> VideoQualityReport:
        """Run full quality assessment on a sequence of frames."""
        # Spatial metrics (average across frames)
        sharpness_scores = [self.spatial.sharpness_score(f) for f in frames]
        noise_scores = [self.spatial.noise_level(f) for f in frames]
        
        # Temporal metrics
        motion_consistency = self.temporal.motion_consistency(frames)
        flicker = self.temporal.flicker_detection(frames)
        temporal_stability = self.temporal.temporal_stability(frames)
        flow_smoothness = self.temporal.optical_flow_smoothness(frames)
        
        # Perceptual metrics
        perceptual_scores = [self.perceptual.niqe_proxy(f) for f in frames]
        aesthetic_scores = [self.perceptual.aesthetic_score(f) for f in frames]
        
        # Color consistency
        color_distributions = [self.spatial.color_distribution(f) for f in frames]
        color_consistency = self._compute_color_consistency(color_distributions)
        
        # Normalize scores to [0, 1]
        spatial_score = np.mean(sharpness_scores) / (np.mean(sharpness_scores) + np.mean(noise_scores) + 1e-8)
        temporal_score = (motion_consistency + temporal_stability + flow_smoothness) / 3
        perceptual_score = np.mean(perceptual_scores) / (np.max(perceptual_scores) + 1e-8)
        
        overall = (spatial_score + temporal_score + perceptual_score) / 3
        
        return VideoQualityReport(
            spatial_score=float(spatial_score),
            temporal_score=float(temporal_score),
            perceptual_score=float(perceptual_score),
            motion_smoothness=float(motion_consistency),
            color_consistency=float(color_consistency),
            sharpness_score=float(np.mean(sharpness_scores)),
            overall_score=float(overall),
            details={
                "flicker": float(flicker),
                "temporal_stability": float(temporal_stability),
                "flow_smoothness": float(flow_smoothness),
                "avg_noise": float(np.mean(noise_scores)),
                "avg_aesthetic": float(np.mean(aesthetic_scores)),
                "n_frames": len(frames),
            }
        )
    
    def _compute_color_consistency(self, distributions: List[Dict]) -> float:
        """Compute color consistency across frames."""
        if len(distributions) < 2:
            return 1.0
        
        means = []
        for d in distributions:
            if "ch0" in d:
                means.append([d["ch0"]["mean"], d.get("ch1", {}).get("mean", 0), d.get("ch2", {}).get("mean", 0)])
        
        if not means:
            return 1.0
        
        means = np.array(means)
        # Consistency = 1 - normalized variance
        consistency = 1.0 / (1.0 + np.mean(np.std(means, axis=0)))
        return float(consistency)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution."""
    from scipy.ndimage import convolve
    return convolve(image, kernel)


if __name__ == "__main__":
    print("Video Quality Assessment")
    print("=" * 50)
    
    # Generate test frames
    frames = [np.random.rand(64, 64, 3) for _ in range(10)]
    
    vqa = VideoQualityAssessment()
    report = vqa.assess(frames)
    
    print(f"\nQuality Report:")
    print(f"  Spatial score: {report.spatial_score:.4f}")
    print(f"  Temporal score: {report.temporal_score:.4f}")
    print(f"  Perceptual score: {report.perceptual_score:.4f}")
    print(f"  Motion smoothness: {report.motion_smoothness:.4f}")
    print(f"  Color consistency: {report.color_consistency:.4f}")
    print(f"  Sharpness: {report.sharpness_score:.4f}")
    print(f"  Overall: {report.overall_score:.4f}")
    print(f"\n  Details: flicker={report.details['flicker']:.4f}, "
          f"stability={report.details['temporal_stability']:.4f}")
    
    print("\nMetrics: Sharpness (Laplacian), NIQE proxy, Aesthetic (rule of thirds)")
    print("         Motion consistency, Flicker detection, Optical flow smoothness")
