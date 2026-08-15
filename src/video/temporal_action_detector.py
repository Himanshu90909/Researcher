"""
Temporal Action Detection Module
Detects and localizes actions/events in video using temporal proposals
and classification.
Based on Singh et al. (2018) and proposal-based action detection methods.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    WALKING = "walking"
    RUNNING = "running"
    COOKING = "cooking"
    SPORTS = "sports"
    INTERVIEW = "interview"
    MUSIC = "music"
    GAMING = "gaming"
    TRAVEL = "travel"
    UNKNOWN = "unknown"


@dataclass
class ActionProposal:
    """A temporal action proposal."""
    start_time: float
    end_time: float
    action_type: ActionType
    confidence: float
    features: Optional[np.ndarray] = None


@dataclass 
class TemporalSegment:
    """A temporal segment of video."""
    start_time: float
    end_time: float
    features: np.ndarray
    scene_id: int


class FeatureExtractor:
    """Extracts temporal features from video segments."""
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
        # Simulated feature dictionary for common action types
        self.action_prototypes = {
            ActionType.WALKING: np.random.randn(feature_dim) * 0.5,
            ActionType.RUNNING: np.random.randn(feature_dim) * 0.5,
            ActionType.COOKING: np.random.randn(feature_dim) * 0.5,
            ActionType.SPORTS: np.random.randn(feature_dim) * 0.5,
            ActionType.INTERVIEW: np.random.randn(feature_dim) * 0.5,
            ActionType.MUSIC: np.random.randn(feature_dim) * 0.5,
            ActionType.GAMING: np.random.randn(feature_dim) * 0.5,
            ActionType.TRAVEL: np.random.randn(feature_dim) * 0.5,
        }
    
    def extract_temporal_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract temporal features from a sequence of frames."""
        # Simulated: would use 3D CNN or I3D in production
        features = np.random.randn(self.feature_dim) * 0.3
        
        # Compute optical flow proxy (frame differencing)
        if len(frames) > 1:
            diffs = [np.mean(np.abs(frames[i] - frames[i-1])) for i in range(1, len(frames))]
            motion_intensity = np.mean(diffs)
            features[0] = motion_intensity  # Motion feature
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features


class TemporalActionDetector:
    """
    Temporal Action Detection pipeline.
    1. Generate temporal proposals (sliding window)
    2. Extract features for each proposal
    3. Classify action type
    4. Refine boundaries with temporal IoU
    """
    
    def __init__(self, feature_dim: int = 512, proposal_length: float = 2.0,
                 stride: float = 0.5, confidence_threshold: float = 0.5):
        self.feature_extractor = FeatureExtractor(feature_dim)
        self.proposal_length = proposal_length
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.feature_dim = feature_dim
    
    def generate_proposals(self, duration: float) -> List[Tuple[float, float]]:
        """Generate temporal proposals using sliding window."""
        proposals = []
        start = 0.0
        while start + self.proposal_length <= duration:
            proposals.append((start, start + self.proposal_length))
            start += self.stride
        return proposals
    
    def classify_action(self, features: np.ndarray) -> Tuple[ActionType, float]:
        """Classify action type from features using prototype matching."""
        best_type = ActionType.UNKNOWN
        best_score = -1
        
        for action_type, prototype in self.feature_extractor.action_prototypes.items():
            # Cosine similarity
            score = np.dot(features, prototype) / (
                np.linalg.norm(features) * np.linalg.norm(prototype) + 1e-8
            )
            if score > best_score:
                best_score = score
                best_type = action_type
        
        # Convert similarity to confidence (sigmoid)
        confidence = 1 / (1 + np.exp(-5 * (best_score - 0.5)))
        return best_type, float(confidence)
    
    def detect(self, video_duration: float, 
               frames_provider: Optional[callable] = None) -> List[ActionProposal]:
        """
        Detect actions in a video.
        
        Args:
            video_duration: Total video duration in seconds
            frames_provider: Optional callback to get frames for a time range
            
        Returns:
            List of detected ActionProposal objects
        """
        proposals = self.generate_proposals(video_duration)
        detections = []
        
        for start, end in proposals:
            # Get frames for this segment (simulated)
            if frames_provider:
                frames = frames_provider(start, end)
            else:
                # Simulated frames
                n_frames = int((end - start) * 30)  # 30 fps
                frames = [np.random.rand(224, 224, 3) for _ in range(n_frames)]
            
            # Extract features
            features = self.feature_extractor.extract_temporal_features(frames)
            
            # Classify
            action_type, confidence = self.classify_action(features)
            
            if confidence >= self.confidence_threshold:
                detections.append(ActionProposal(
                    start_time=start,
                    end_time=end,
                    action_type=action_type,
                    confidence=confidence,
                    features=features,
                ))
        
        # Non-maximum suppression
        detections = self._nms(detections)
        return detections
    
    def _nms(self, detections: List[ActionProposal], 
             iou_threshold: float = 0.3) -> List[ActionProposal]:
        """Non-maximum suppression to remove overlapping detections."""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: -x.confidence)
        kept = []
        
        for det in detections:
            should_keep = True
            for kept_det in kept:
                iou = self._temporal_iou(det, kept_det)
                if iou > iou_threshold and det.action_type == kept_det.action_type:
                    should_keep = False
                    break
            if should_keep:
                kept.append(det)
        
        return kept
    
    def _temporal_iou(self, a: ActionProposal, b: ActionProposal) -> float:
        """Compute temporal Intersection over Union."""
        intersection = max(0, min(a.end_time, b.end_time) - max(a.start_time, b.start_time))
        union = (a.end_time - a.start_time) + (b.end_time - b.start_time) - intersection
        return intersection / (union + 1e-8)
    
    def summarize_detections(self, detections: List[ActionProposal]) -> Dict:
        """Summarize detection results."""
        action_counts = {}
        for det in detections:
            action = det.action_type.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        total_duration = sum(d.end_time - d.start_time for d in detections)
        
        return {
            "total_detections": len(detections),
            "action_distribution": action_counts,
            "total_action_duration": total_duration,
            "avg_confidence": np.mean([d.confidence for d in detections]) if detections else 0,
        }


if __name__ == "__main__":
    detector = TemporalActionDetector(
        proposal_length=3.0,
        stride=1.0,
        confidence_threshold=0.4
    )
    
    print("Temporal Action Detection")
    print(f"  Proposal length: {detector.proposal_length}s, Stride: {detector.stride}s")
    print(f"  Confidence threshold: {detector.confidence_threshold}")
    
    # Detect in a 30-second video
    detections = detector.detect(video_duration=30.0)
    print(f"\nDetected {len(detections)} actions:")
    
    for i, det in enumerate(detections):
        print(f"  [{i+1}] {det.action_type.value}: "
              f"{det.start_time:.1f}s - {det.end_time:.1f}s "
              f"(confidence: {det.confidence:.3f})")
    
    summary = detector.summarize_detections(detections)
    print(f"\nSummary: {summary}")
