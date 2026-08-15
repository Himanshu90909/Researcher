"""
Scene Understanding & SLAM for AR/VR
Implements Simultaneous Localization and Mapping, scene parsing,
and 3D scene graph construction for AR applications.

Based on:
- Meta: 'SceneScript' (Avetisyan et al. 2024) - Structured 3D scene understanding
- ORB-SLAM3 (Campos et al. 2021) - Visual-inertial SLAM
- Meta: 'NeuralRecon' (Sun et al. 2021) - Real-time 3D reconstruction
- Apple: 'ARKit' scene understanding

Key concepts:
1. SLAM: Track camera pose + build map simultaneously
2. Scene Parsing: Detect objects, walls, floor, ceiling in 3D
3. Scene Graph: Hierarchical representation of scene structure
4. Plane Detection: Find planar surfaces for AR placement
5. Semantic Segmentation: Label pixels/points with semantic classes
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class SemanticClass(Enum):
    WALL = "wall"
    FLOOR = "floor"
    CEILING = "ceiling"
    TABLE = "table"
    CHAIR = "chair"
    SOFA = "sofa"
    BED = "bed"
    WINDOW = "window"
    DOOR = "door"
    TV = "tv"
    LAMP = "lamp"
    PLANT = "plant"
    PERSON = "person"
    UNKNOWN = "unknown"


@dataclass
class Plane3D:
    """Detected 3D plane (for AR placement)."""
    center: np.ndarray     # (3,) plane center
    normal: np.ndarray     # (3,) plane normal
    extent: np.ndarray     # (2,) width, height
    semantic_class: SemanticClass = SemanticClass.UNKNOWN
    confidence: float = 0.5
    plane_id: int = 0

    def point_distance(self, point: np.ndarray) -> float:
        """Distance from point to plane."""
        return float(np.dot(point - self.center, self.normal))

    def contains_point(self, point: np.ndarray, threshold: float = 0.05) -> bool:
        """Check if point lies on plane."""
        return abs(self.point_distance(point)) < threshold


@dataclass
class Landmark:
    """SLAM landmark (3D point with observations)."""
    position: np.ndarray       # (3,) 3D position
    descriptor: np.ndarray    # Feature descriptor
    observations: List[int]   # Frame indices where observed
    n_observations: int = 0
    is_valid: bool = True


@dataclass
class Keyframe:
    """SLAM keyframe."""
    pose: np.ndarray          # (4,4) camera pose
    timestamp: float
    features: np.ndarray      # (N, 2) 2D feature points
    descriptors: np.ndarray   # (N, D) feature descriptors
    landmarks: List[int]      # Associated landmark indices


class SLAM:
    """
    Simultaneous Localization and Mapping.
    Tracks camera trajectory while building a 3D map.
    
    Pipeline:
    1. Feature extraction (ORB/SIFT-like)
    2. Feature matching between frames
    3. Pose estimation (PnP)
    4. Local bundle adjustment (optimize poses + landmarks)
    5. Map management (keyframe culling, landmark merging)
    """
    def __init__(self, n_features: int = 500):
        self.n_features = n_features
        self.landmarks: List[Landmark] = []
        self.keyframes: List[Keyframe] = []
        self.trajectory: List[np.ndarray] = []  # Camera poses
        self.current_pose: np.ndarray = np.eye(4)
        self.frame_count: int = 0
    
    def process_frame(self, image: np.ndarray, timestamp: float) -> Dict:
        """Process a new frame through SLAM."""
        # Extract features (simulated)
        features = np.random.rand(self.n_features, 2) * np.array(image.shape[:2][::-1])
        descriptors = np.random.randn(self.n_features, 32)
        
        # Pose estimation (simulated - incremental motion)
        motion = np.eye(4)
        motion[:3, 3] = np.random.randn(3) * 0.01  # Small random motion
        self.current_pose = self.current_pose @ motion
        
        # Create keyframe every 10 frames
        is_keyframe = (self.frame_count % 10 == 0)
        if is_keyframe:
            kf = Keyframe(
                pose=self.current_pose.copy(),
                timestamp=timestamp,
                features=features,
                descriptors=descriptors,
                landmarks=[],
            )
            self.keyframes.append(kf)
            
            # Add landmarks
            for i in range(0, len(features), 5):  # Subsample
                point_3d = self._triangulate(features[i], self.current_pose)
                lm = Landmark(
                    position=point_3d,
                    descriptor=descriptors[i],
                    observations=[self.frame_count],
                    n_observations=1,
                )
                self.landmarks.append(lm)
                kf.landmarks.append(len(self.landmarks) - 1)
        
        self.trajectory.append(self.current_pose.copy())
        self.frame_count += 1
        
        return {
            'pose': self.current_pose.copy(),
            'is_keyframe': is_keyframe,
            'n_keyframes': len(self.keyframes),
            'n_landmarks': len(self.landmarks),
        }
    
    def _triangulate(self, feature_2d: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Triangulate a 3D point from 2D observation."""
        # Simulated triangulation
        depth = np.random.uniform(1.0, 5.0)
        direction = np.array([feature_2d[0], feature_2d[1], depth]) / depth
        world_point = pose[:3, :3] @ direction + pose[:3, 3]
        return world_point
    
    def local_bundle_adjustment(self):
        """Optimize poses and landmarks (simulated)."""
        # In practice: g2o or Ceres optimization
        n_poses = len(self.keyframes)
        n_points = len(self.landmarks)
        
        # Simulated optimization: perturb slightly
        for lm in self.landmarks:
            lm.position += np.random.randn(3) * 0.001
        
        return {
            'n_poses': n_poses,
            'n_points': n_points,
            'iterations': 10,
            'cost': np.random.uniform(0.01, 0.1),
        }
    
    def get_map(self) -> Dict:
        """Get the current map."""
        points = np.array([lm.position for lm in self.landmarks]) if self.landmarks else np.empty((0, 3))
        poses = np.array([kf.pose for kf in self.keyframes]) if self.keyframes else np.empty((0, 4, 4))
        
        return {
            'points': points,
            'poses': poses,
            'n_points': len(self.landmarks),
            'n_keyframes': len(self.keyframes),
            'trajectory_length': len(self.trajectory),
        }


class PlaneDetector:
    """
    Detect planar surfaces in 3D point clouds for AR object placement.
    Uses RANSAC for plane fitting.
    """
    def __init__(self, ransac_iterations: int = 100, 
                 distance_threshold: float = 0.02,
                 min_inliers: int = 50):
        self.ransac_iterations = ransac_iterations
        self.distance_threshold = distance_threshold
        self.min_inliers = min_inliers
    
    def detect_planes(self, point_cloud: np.ndarray, 
                      colors: Optional[np.ndarray] = None) -> List[Plane3D]:
        """Detect planes in point cloud using RANSAC."""
        planes = []
        remaining_points = point_cloud.copy()
        plane_id = 0
        
        while len(remaining_points) > self.min_inliers:
            plane, inliers = self._ransac_plane(remaining_points)
            if plane is None:
                break
            
            # Classify plane
            plane.semantic_class = self._classify_plane(plane)
            plane.plane_id = plane_id
            planes.append(plane)
            
            # Remove inliers
            remaining_points = remaining_points[~inliers]
            plane_id += 1
        
        return planes
    
    def _ransac_plane(self, points: np.ndarray) -> Tuple[Optional[Plane3D], np.ndarray]:
        """RANSAC plane fitting."""
        best_plane = None
        best_inliers = np.zeros(len(points), dtype=bool)
        best_n_inliers = 0
        
        for _ in range(self.ransac_iterations):
            # Sample 3 random points
            indices = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[indices]
            
            # Fit plane
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            center = np.mean([p1, p2, p3], axis=0)
            
            # Count inliers
            distances = np.abs((points - center) @ normal)
            inliers = distances < self.distance_threshold
            n_inliers = np.sum(inliers)
            
            if n_inliers > best_n_inliers:
                best_n_inliers = n_inliers
                best_inliers = inliers
                inlier_points = points[inliers]
                center = inlier_points.mean(axis=0)
                
                # Plane extent
                if len(inlier_points) > 2:
                    centered = inlier_points - center
                    # Project to plane
                    u, s, vh = np.linalg.svd(centered[:, :2])
                    extent = np.array([s[0] * 2, s[1] * 2]) if len(s) >= 2 else np.array([1.0, 1.0])
                else:
                    extent = np.array([1.0, 1.0])
                
                best_plane = Plane3D(
                    center=center,
                    normal=normal,
                    extent=extent,
                    confidence=n_inliers / len(points),
                )
        
        if best_n_inliers < self.min_inliers:
            return None, np.zeros(len(points), dtype=bool)
        
        return best_plane, best_inliers
    
    def _classify_plane(self, plane: Plane3D) -> SemanticClass:
        """Classify plane by normal direction."""
        normal = plane.normal
        up = np.array([0, 1, 0])
        
        angle = np.arccos(np.clip(np.dot(normal, up), -1, 1))
        
        if angle < np.radians(20):  # Normal pointing up
            return SemanticClass.FLOOR
        elif angle > np.radians(160):  # Normal pointing down
            return SemanticClass.CEILING
        else:  # Vertical
            return SemanticClass.WALL


class SceneGraph:
    """
    3D Scene Graph: hierarchical representation of scene structure.
    
    Scene -> Room -> Objects -> Parts
    
    Used by Meta's SceneScript for structured 3D understanding.
    """
    def __init__(self):
        self.nodes: Dict[int, Dict] = {}
        self.edges: List[Tuple[int, int, str]] = []  # (from, to, relation)
        self.next_id: int = 0
    
    def add_object(self, position: np.ndarray, size: np.ndarray,
                   semantic_class: SemanticClass) -> int:
        """Add an object to the scene graph."""
        node_id = self.next_id
        self.next_id += 1
        self.nodes[node_id] = {
            'type': 'object',
            'position': position,
            'size': size,
            'semantic_class': semantic_class,
            'confidence': np.random.uniform(0.7, 1.0),
        }
        return node_id
    
    def add_room(self, position: np.ndarray, size: np.ndarray,
                 room_type: str = "living_room") -> int:
        """Add a room node."""
        node_id = self.next_id
        self.next_id += 1
        self.nodes[node_id] = {
            'type': 'room',
            'position': position,
            'size': size,
            'room_type': room_type,
        }
        return node_id
    
    def add_relation(self, from_id: int, to_id: int, relation: str):
        """Add a relation between nodes."""
        self.edges.append((from_id, to_id, relation))
    
    def get_objects_by_class(self, semantic_class: SemanticClass) -> List[Dict]:
        """Get all objects of a specific class."""
        return [
            {**node, 'id': nid}
            for nid, node in self.nodes.items()
            if node.get('semantic_class') == semantic_class
        ]
    
    def get_relations(self, node_id: int) -> List[Tuple[int, str]]:
        """Get all relations for a node."""
        relations = []
        for f, t, r in self.edges:
            if f == node_id:
                relations.append((t, r))
            elif t == node_id:
                relations.append((f, f"inverse_{r}"))
        return relations
    
    def to_dict(self) -> Dict:
        """Serialize scene graph."""
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'n_objects': sum(1 for n in self.nodes.values() if n['type'] == 'object'),
            'n_rooms': sum(1 for n in self.nodes.values() if n['type'] == 'room'),
            'n_relations': len(self.edges),
        }


class SemanticSegmentation3D:
    """
    3D semantic segmentation for scene understanding.
    Assigns semantic labels to 3D points.
    """
    def __init__(self, n_classes: int = 14):
        self.n_classes = n_classes
        self.classes = list(SemanticClass)[:n_classes]
    
    def segment(self, point_cloud: np.ndarray, 
                colors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Segment point cloud into semantic classes.
        Returns: (N,) array of class indices
        """
        n_points = len(point_cloud)
        labels = np.zeros(n_points, dtype=int)
        
        # Height-based classification (simulated)
        heights = point_cloud[:, 1]
        
        for i, (h, p) in enumerate(zip(heights, point_cloud)):
            if h < -0.5:
                labels[i] = SemanticClass.FLOOR.value
            elif h > 2.0:
                labels[i] = SemanticClass.CEILING.value
            else:
                # Random semantic class for walls/objects
                labels[i] = np.random.choice([
                    SemanticClass.WALL.value,
                    SemanticClass.TABLE.value,
                    SemanticClass.CHAIR.value,
                    SemanticClass.SOFA.value,
                ])
        
        return labels
    
    def get_segmentation_stats(self, labels: np.ndarray) -> Dict:
        """Get statistics about segmentation."""
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = {}
        for u, c in zip(unique, counts):
            class_counts[self.classes[u].value] = int(c)
        return class_counts


class ARSceneUnderstanding:
    """
    Full AR scene understanding pipeline.
    Combines SLAM + plane detection + scene graph + segmentation.
    """
    def __init__(self):
        self.slam = SLAM(n_features=200)
        self.plane_detector = PlaneDetector()
        self.scene_graph = SceneGraph()
        self.semantic_seg = SemanticSegmentation3D()
    
    def process_scene(self, image: np.ndarray, 
                      point_cloud: Optional[np.ndarray] = None,
                      timestamp: float = 0.0) -> Dict:
        """
        Process a frame for full scene understanding.
        
        1. SLAM: Track camera and build map
        2. Detect planes for AR placement
        3. Build scene graph
        4. Semantic segmentation
        """
        # SLAM
        slam_result = self.slam.process_frame(image, timestamp)
        
        # Get map
        map_data = self.slam.get_map()
        
        # Plane detection
        planes = []
        if map_data['n_points'] > 0:
            points = map_data['points']
            planes = self.plane_detector.detect_planes(points)
        
        # Add planes to scene graph
        room_id = self.scene_graph.add_room(
            position=np.array([0, 0, 0]),
            size=np.array([5, 3, 5]),
        )
        for plane in planes:
            obj_id = self.scene_graph.add_object(
                position=plane.center,
                size=np.array([plane.extent[0], 0.01, plane.extent[1]]),
                semantic_class=plane.semantic_class,
            )
            self.scene_graph.add_relation(room_id, obj_id, "contains")
        
        # Semantic segmentation
        seg_labels = np.array([])
        if map_data['n_points'] > 0:
            seg_labels = self.semantic_seg.segment(map_data['points'])
        
        return {
            'slam': slam_result,
            'map': map_data,
            'planes': planes,
            'n_planes': len(planes),
            'scene_graph': self.scene_graph.to_dict(),
            'segmentation': seg_labels.tolist(),
        }
