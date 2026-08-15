"""
3D Gaussian Splatting for Real-Time Radiance Field Rendering
Implements explicit 3D Gaussian representation for real-time novel view synthesis.

Based on:
- Kerbl et al. (2023) '3D Gaussian Splatting for Real-Time Radiance Field Rendering'
- Zip-NeRF (Barron et al. 2023) - importance sampling

Key concepts:
1. Represent scene as set of 3D Gaussians (position, covariance, color, opacity)
2. Project 3D Gaussians to 2D screen space
3. Alpha-blend splats in sorted order for each pixel

Advantages over NeRF:
- Real-time rendering (>100 FPS) vs NeRF (minutes per frame)
- Explicit representation (no MLP queries)
- Editable (move, merge, split Gaussians)
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Gaussian3D:
    """A single 3D Gaussian splat."""
    position: np.ndarray      # (3,) center in 3D
    scale: np.ndarray         # (3,) scaling factors
    rotation: np.ndarray      # (4,) quaternion (w, x, y, z)
    color: np.ndarray         # (3,) spherical harmonic coefficients (simplified to RGB)
    opacity: float            # (1,) alpha value [0, 1]

    def get_covariance_3d(self) -> np.ndarray:
        """Compute 3D covariance matrix from scale and rotation."""
        # Rotation matrix from quaternion
        R = self._quat_to_matrix(self.rotation)
        S = np.diag(self.scale)
        covariance = R @ S @ S.T @ R.T  # Sigma = R * S * S^T * R^T
        return covariance

    def _quat_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])


@dataclass
class GaussianSplattingConfig:
    """Configuration for Gaussian Splatting."""
    n_gaussians: int = 1000
    max_splat_size: float = 0.1
    min_opacity: float = 0.01
    densification_interval: int = 100
    densification_gradient: float = 0.0002
    pruning_opacity: float = 0.005
    split_factor: float = 1.6


class ProjectionUtils:
    """Utilities for projecting 3D Gaussians to 2D screen space."""
    
    @staticmethod
    def project_gaussian(gaussian: Gaussian3D, 
                        view_matrix: np.ndarray,
                        proj_matrix: np.ndarray,
                        viewport: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Project a 3D Gaussian to 2D screen space.
        
        Returns:
            mean_2d: (2,) screen-space center
            cov_2d: (2,2) 2D covariance
            opacity: projected opacity
        """
        # Transform center to view space
        center_view = view_matrix @ np.append(gaussian.position, 1.0)
        
        # Transform to clip space
        center_clip = proj_matrix @ center_view
        w = center_clip[3]
        if abs(w) < 1e-6:
            return np.zeros(2), np.eye(2), 0.0
        
        # Screen space center
        center_ndc = center_clip[:3] / w
        mean_2d = np.array([
            (center_ndc[0] + 1) * 0.5 * viewport[0],
            (1 - center_ndc[1]) * 0.5 * viewport[1],
        ])
        
        # Project covariance to 2D (EWA approximation)
        cov_3d = gaussian.get_covariance_3d()
        # Jacobian of projection (simplified)
        view_center = center_view[:3]
        depth = -view_center[2]
        if abs(depth) < 1e-6:
            depth = 1e-6
        
        J = np.array([
            [1/depth, 0, -view_center[0]/(depth**2)],
            [0, 1/depth, -view_center[1]/(depth**2)],
        ])
        
        W = view_matrix[:3, :3]
        cov_view = W @ cov_3d @ W.T
        cov_2d = J @ cov_view[:3, :3] @ J.T
        
        return mean_2d, cov_2d, gaussian.opacity

    @staticmethod
    def evaluate_gaussian_2d(point: np.ndarray, mean: np.ndarray, 
                             cov: np.ndarray) -> float:
        """Evaluate 2D Gaussian at a point."""
        diff = point - mean
        inv_cov = np.linalg.inv(cov + np.eye(2) * 1e-6)
        exponent = -0.5 * diff @ inv_cov @ diff
        return float(np.exp(exponent))


class GaussianSplattingRenderer:
    """
    Real-time Gaussian Splatting renderer.
    Renders scenes represented as collections of 3D Gaussians.
    """
    def __init__(self, config: GaussianSplattingConfig):
        self.config = config
        self.gaussians: List[Gaussian3D] = []
        self.projection = ProjectionUtils()
        
    def initialize_random(self, bounds: Tuple[np.ndarray, np.ndarray]):
        """Initialize random Gaussians within bounds."""
        min_bound, max_bound = bounds
        for _ in range(self.config.n_gaussians):
            pos = min_bound + np.random.rand(3) * (max_bound - min_bound)
            rot = np.random.randn(4)
            rot = rot / np.linalg.norm(rot)
            self.gaussians.append(Gaussian3D(
                position=pos,
                scale=np.random.rand(3) * 0.05 + 0.01,
                rotation=rot,
                color=np.random.rand(3),
                opacity=np.random.rand() * 0.5 + 0.5,
            ))

    def render(self, view_matrix: np.ndarray, proj_matrix: np.ndarray,
               viewport: Tuple[int, int]) -> np.ndarray:
        """
        Render scene from given camera.
        
        Args:
            view_matrix: (4,4) view matrix
            proj_matrix: (4,4) projection matrix
            viewport: (width, height)
            
        Returns:
            image: (H, W, 3) rendered image
        """
        H, W = viewport[1], viewport[0]
        image = np.zeros((H, W, 3))
        
        # Sort Gaussians by depth (back-to-front)
        depths = []
        for g in self.gaussians:
            view_pos = view_matrix @ np.append(g.position, 1.0)
            depths.append(-view_pos[2])
        
        sorted_indices = np.argsort(depths)[::-1]
        
        # Splat each Gaussian
        for idx in sorted_indices:
            g = self.gaussians[idx]
            mean_2d, cov_2d, opacity = self.projection.project_gaussian(
                g, view_matrix, proj_matrix, viewport
            )
            
            if opacity < self.config.min_opacity:
                continue
            
            # Bounding box of splat
            radius = 3 * np.sqrt(np.maximum(np.diag(cov_2d), 1e-6))
            x_min = max(0, int(mean_2d[0] - radius[0]))
            x_max = min(W, int(mean_2d[0] + radius[0]))
            y_min = max(0, int(mean_2d[1] - radius[1]))
            y_max = min(H, int(mean_2d[1] + radius[1]))
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            # Evaluate Gaussian at each pixel in bounding box
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    point = np.array([x, y], dtype=float)
                    weight = self.projection.evaluate_gaussian_2d(point, mean_2d, cov_2d)
                    alpha = weight * opacity
                    alpha = np.clip(alpha, 0, 1)
                    
                    # Alpha blending (over operator)
                    image[y, x] = image[y, x] * (1 - alpha) + g.color * alpha
        
        return np.clip(image, 0, 1)

    def densify(self, gradient_threshold: float = 0.0002):
        """Split large Gaussians and prune transparent ones."""
        new_gaussians = []
        for g in self.gaussians:
            # Prune low opacity
            if g.opacity < self.config.pruning_opacity:
                continue
            
            # Split large Gaussians
            scale_norm = np.linalg.norm(g.scale)
            if scale_norm > self.config.max_splat_size:
                # Split into 2
                scale_new = g.scale / self.config.split_factor
                offset = g.scale * 0.5
                for sign in [1, -1]:
                    new_g = Gaussian3D(
                        position=g.position + np.array([sign * offset[0], 0, 0]),
                        scale=scale_new,
                        rotation=g.rotation.copy(),
                        color=g.color.copy(),
                        opacity=g.opacity,
                    )
                    new_gaussians.append(new_g)
            else:
                new_gaussians.append(g)
        
        self.gaussians = new_gaussians
        return len(self.gaussians)

    def get_stats(self) -> Dict:
        """Get scene statistics."""
        return {
            'n_gaussians': len(self.gaussians),
            'avg_opacity': np.mean([g.opacity for g in self.gaussians]) if self.gaussians else 0,
            'avg_scale': np.mean([np.linalg.norm(g.scale) for g in self.gaussians]) if self.gaussians else 0,
        }


def create_view_matrix(position: np.ndarray, look_at: np.ndarray, 
                       up: np.ndarray = np.array([0, 1, 0])) -> np.ndarray:
    """Create view matrix (camera looking at target)."""
    forward = look_at - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, forward)
    
    view = np.eye(4)
    view[:3, 0] = right
    view[:3, 1] = new_up
    view[:3, 2] = -forward
    view[:3, 3] = -position
    return view


def create_proj_matrix(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Create perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fov) / 2)
    proj = np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0],
    ])
    return proj


if __name__ == "__main__":
    print("3D Gaussian Splatting (Real-Time Radiance Field Rendering)")
    print("=" * 60)

    config = GaussianSplattingConfig(n_gaussians=50)
    renderer = GaussianSplattingRenderer(config)

    # Initialize scene
    bounds = (np.array([-1, -1, -1]), np.array([1, 1, 1]))
    renderer.initialize_random(bounds)
    print(f"Initialized {len(renderer.gaussians)} Gaussians")

    # Camera setup
    view = create_view_matrix(np.array([0, 0, 3]), np.array([0, 0, 0]))
    proj = create_proj_matrix(60, 1.0, 0.1, 100)
    
    # Render
    image = renderer.render(view, proj, (32, 32))
    print(f"Rendered: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")

    # Densify
    n_after = renderer.densify()
    print(f"After densification: {n_after} Gaussians")
    
    stats = renderer.get_stats()
    print(f"Stats: {stats}")

    print("\nKey advantages over NeRF:")
    print("  Real-time rendering (>100 FPS)")
    print("  Explicit representation (no MLP)")
    print("  Editable (split, merge, move Gaussians)")
    print("\nReferences:")
    print("  Kerbl et al. (2023) - 3D Gaussian Splatting for Real-Time Rendering")
    print("  Barron et al. (2023) - Zip-NeRF: Importance Sampling")
