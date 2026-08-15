"""
Neural Radiance Fields (NeRF) for 3D Scene Reconstruction
Implements volumetric rendering from posed images using MLP-based
radiance field representation.

Based on:
- Mildenhall et al. (2020) 'NeRF: Representing Scenes as Neural Radiance Fields'
- Barron et al. (2022) 'Mip-NeRF 360'
- Müller et al. (2022) 'Instant NGP' (hash encoding)

Key concepts:
1. Encode 3D point + viewing direction -> density + color via MLP
2. Sample rays from camera, query points along each ray
3. Volume rendering: accumulate colors weighted by density (alpha compositing)
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CameraPose:
    """6DoF camera pose (position + orientation)."""
    position: np.ndarray      # (3,) xyz
    rotation: np.ndarray     # (3,3) rotation matrix
    focal_length: float
    width: int
    height: int

    def get_rays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate camera rays for all pixels."""
        # Pixel coordinates
        i, j = np.meshgrid(
            np.arange(self.width), np.arange(self.height), indexing='xy'
        )
        # Camera coordinates -> ray directions
        dirs = np.stack([
            (i - self.width / 2) / self.focal_length,
            -(j - self.height / 2) / self.focal_length,
            -np.ones_like(i),
        ], axis=-1)
        # World coordinates
        rays_d = dirs @ self.rotation.T  # (H, W, 3)
        rays_o = np.broadcast_to(self.position, rays_d.shape)  # (H, W, 3)
        return rays_o, rays_d


@dataclass
class NeRFConfig:
    """Configuration for NeRF model."""
    pos_encoding_dim: int = 60   # 3D position encoding
    dir_encoding_dim: int = 24   # View direction encoding
    hidden_dim: int = 256
    n_layers: int = 8
    n_samples: int = 64           # Samples per ray (coarse)
    n_samples_fine: int = 128     # Samples per ray (fine)
    near: float = 0.1
    far: float = 10.0
    lerp_sigma: float = 0.0      # Depth smoothing


class PositionalEncoding:
    """
    Positional encoding for high-frequency detail capture.
    gamma(x) = [x, sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]
    """
    def __init__(self, n_freqs: int = 10):
        self.n_freqs = n_freqs
        self.freq_bands = 2.0 ** np.arange(n_freqs)  # (L,)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input coordinates."""
        # x: (..., C)
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(np.sin(freq * np.pi * x))
            encoded.append(np.cos(freq * np.pi * x))
        return np.concatenate(encoded, axis=-1)


class NeRFLayer:
    """Single MLP layer with optional skip connection."""
    def __init__(self, in_dim: int, out_dim: int, skip: bool = False):
        self.weights = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.bias = np.zeros(out_dim)
        self.skip = skip

    def forward(self, x: np.ndarray, skip_input: Optional[np.ndarray] = None) -> np.ndarray:
        h = x @ self.weights + self.bias
        if skip_input is not None:
            h = np.concatenate([h, skip_input], axis=-1)
        return np.maximum(0, h)  # ReLU


class NeRF:
    """
    Neural Radiance Field model.
    
    Architecture:
    - Position MLP (8 layers, 256 hidden) -> density + feature
    - Direction MLP (1 layer) -> RGB color
    
    sigma(x) = MLP_pos(x)[0]  (volume density)
    c(x, d) = MLP_dir(MLP_pos(x)[1:], d)  (view-dependent color)
    
    Volume rendering:
    C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i
    where T_i = exp(-sum_{j<i} sigma_j * delta_j)
    """
    def __init__(self, config: NeRFConfig):
        self.config = config
        self.pos_encoder = PositionalEncoding(n_freqs=config.pos_encoding_dim // 6)
        self.dir_encoder = PositionalEncoding(n_freqs=config.dir_encoding_dim // 3)

        # Position MLP (with skip connection at layer 4)
        pos_dim = 3 + 2 * 3 * (config.pos_encoding_dim // 6)  # 3 + encoded
        self.pos_layers = []
        dims = [pos_dim] + [config.hidden_dim] * config.n_layers
        for i in range(config.n_layers):
            in_d = dims[i] + (pos_dim if i == 4 else 0)  # Skip at layer 4
            self.pos_layers.append(NeRFLayer(in_d, dims[i+1] if i < config.n_layers-1 else config.hidden_dim))

        # Density head
        self.density_layer = NeRFLayer(config.hidden_dim, 1)

        # Direction MLP
        feat_dim = config.hidden_dim
        dir_dim = 3 + 2 * 3 * (config.dir_encoding_dim // 3)
        self.dir_layers = [
            NeRFLayer(feat_dim + dir_dim, config.hidden_dim // 2),
            NeRFLayer(config.hidden_dim // 2, 3),
        ]

    def query(self, points: np.ndarray, directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query the NeRF at 3D points with viewing directions.
        
        Args:
            points: (N, 3) 3D coordinates
            directions: (N, 3) viewing directions
            
        Returns:
            densities: (N,) volume density sigma
            colors: (N, 3) RGB color
        """
        # Encode position
        pos_encoded = self.pos_encoder.encode(points)  # (N, pos_dim)

        # Position MLP with skip connection
        h = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            skip = pos_encoded if i == 4 else None
            h = layer.forward(h, skip_input=skip if skip is not None else None)

        # Density (from position MLP)
        density_raw = self.density_layer.forward(h)
        density = np.maximum(0, density_raw.squeeze(-1))  # ReLU on density

        # Color (from direction MLP)
        dir_encoded = self.dir_encoder.encode(directions)
        h_dir = np.concatenate([h, dir_encoded], axis=-1)
        for layer in self.dir_layers:
            h_dir = layer.forward(h_dir)
        color = 1.0 / (1.0 + np.exp(-np.clip(h_dir, -500, 500)))  # (N, 3) in [0, 1]

        return density, color

    def render_rays(self, rays_o: np.ndarray, rays_d: np.ndarray) -> Dict:
        """
        Volume render rays using NeRF.
        
        Args:
            rays_o: (N_rays, 3) ray origins
            rays_d: (N_rays, 3) ray directions
            
        Returns:
            Dict with rendered RGB, depth, and weights
        """
        N_rays = rays_o.shape[0]
        n_samples = self.config.n_samples

        # Sample points along rays (stratified sampling)
        t_vals = np.linspace(self.config.near, self.config.far, n_samples)
        t_vals = t_vals + np.random.uniform(0, 1, (N_rays, n_samples)) * (self.config.far - self.config.near) / n_samples
        # (N_rays, n_samples)

        # 3D points: o + t * d
        points = rays_o[:, None, :] + t_vals[..., None] * rays_d[:, None, :]  # (N_rays, n_samples, 3)
        directions = np.broadcast_to(rays_d[:, None, :], points.shape)

        # Query NeRF
        points_flat = points.reshape(-1, 3)
        dirs_flat = directions.reshape(-1, 3)
        densities, colors = self.query(points_flat, dirs_flat)

        # Reshape
        densities = densities.reshape(N_rays, n_samples)  # (N_rays, n_samples)
        colors = colors.reshape(N_rays, n_samples, 3)

        # Volume rendering (alpha compositing)
        # delta_i = t_{i+1} - t_i (distance between adjacent samples)
        deltas = np.diff(t_vals, prepend=t_vals[:, :1], axis=1)  # (N_rays, n_samples)

        # alpha_i = 1 - exp(-sigma_i * delta_i)
        alpha = 1.0 - np.exp(-densities * deltas)

        # T_i = exp(-sum_{j<i} sigma_j * delta_j) = prod_{j<i} (1 - alpha_j)
        T = np.cumprod(1.0 - alpha + 1e-10, axis=1)
        T = np.concatenate([np.ones((N_rays, 1)), T[:, :-1]], axis=1)  # T_1 = 1

        # Weights w_i = T_i * alpha_i
        weights = T * alpha  # (N_rays, n_samples)

        # Rendered color C(r) = sum_i w_i * c_i
        rgb = np.sum(weights[..., None] * colors, axis=1)  # (N_rays, 3)

        # Depth = sum_i w_i * t_i
        depth = np.sum(weights * t_vals, axis=1)  # (N_rays,)

        # Accumulated opacity
        acc = np.sum(weights, axis=1)  # (N_rays,)

        return {
            'rgb': rgb,
            'depth': depth,
            'weights': weights,
            'accumulated_opacity': acc,
        }

    def render_image(self, camera: CameraPose) -> Dict:
        """Render a full image from a camera pose."""
        rays_o, rays_d = camera.get_rays()
        H, W = camera.height, camera.width

        # Flatten rays
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)

        # Render in chunks (memory efficient)
        chunk_size = 1024
        rgbs, depths = [], []
        for i in range(0, len(rays_o_flat), chunk_size):
            chunk_o = rays_o_flat[i:i+chunk_size]
            chunk_d = rays_d_flat[i:i+chunk_size]
            result = self.render_rays(chunk_o, chunk_d)
            rgbs.append(result['rgb'])
            depths.append(result['depth'])

        rgb = np.concatenate(rgbs).reshape(H, W, 3)
        depth = np.concatenate(depths).reshape(H, W)

        return {
            'image': np.clip(rgb, 0, 1),
            'depth_map': depth,
            'camera': camera,
        }


class NeRFTrainer:
    """Training pipeline for NeRF models."""
    def __init__(self, model: NeRF, lr: float = 5e-4):
        self.model = model
        self.lr = lr

    def compute_loss(self, rendered: np.ndarray, target: np.ndarray) -> Dict:
        """MSE loss between rendered and target images."""
        mse = np.mean((rendered - target) ** 2)
        psnr = -10 * np.log10(mse + 1e-8)
        return {'mse': float(mse), 'psnr': float(psnr)}

    def train_step(self, camera: CameraPose, target_image: np.ndarray) -> Dict:
        """Single training step."""
        result = self.model.render_image(camera)
        losses = self.compute_loss(result['image'], target_image)
        # Simulated gradient update
        return losses


if __name__ == "__main__":
    print("Neural Radiance Fields (NeRF)")
    print("=" * 60)
    print("Volumetric rendering from posed 2D images -> 3D scene")

    config = NeRFConfig(
        pos_encoding_dim=60, dir_encoding_dim=24,
        hidden_dim=128, n_layers=6, n_samples=32
    )
    nerf = NeRF(config)

    # Define camera
    camera = CameraPose(
        position=np.array([0, 0, 3]),
        rotation=np.eye(3),
        focal_length=100,
        width=32, height=32
    )

    # Render
    result = nerf.render_image(camera)
    print(f"\nRendered image: {result['image'].shape}")
    print(f"Depth map: {result['depth_map'].shape}")
    print(f"RGB range: [{result['image'].min():.3f}, {result['image'].max():.3f}]")
    print(f"Depth range: [{result['depth_map'].min():.3f}, {result['depth_map'].max():.3f}]")

    # Train on a simulated target
    target = np.random.rand(32, 32, 3)
    trainer = NeRFTrainer(nerf)
    for epoch in range(5):
        loss = trainer.train_step(camera, target)
        print(f"Epoch {epoch+1}: MSE={loss['mse']:.6f}, PSNR={loss['psnr']:.2f} dB")

    print("\nKey equations:")
    print("  C(r) = sum_i T_i * alpha_i * c_i")
    print("  alpha_i = 1 - exp(-sigma_i * delta_i)")
    print("  T_i = prod_{j<i} (1 - alpha_j)")
    print("\nReferences:")
    print("  Mildenhall et al. (2020) - NeRF: Representing Scenes as Neural Radiance Fields")
    print("  Barron et al. (2022) - Mip-NeRF 360")
    print("  Müller et al. (2022) - Instant Neural Graphics Primitives (Instant NGP)")
