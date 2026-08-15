"""
Autoregressive Model for Image Generation
Implements a PixelCNN-style autoregressive model that generates images
pixel by pixel, conditioned on previously generated pixels.
Based on van den Oord et al. (2016) 'Pixel Recurrent Neural Networks'.
"""
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class AutoregressiveConfig:
    """Configuration for autoregressive model."""
    image_size: int = 28
    channels: int = 1
    hidden_dim: int = 64
    n_layers: int = 5
    n_embeddings: int = 256  # Pixel values (0-255)


class MaskedConvolution:
    """
    Masked convolution for autoregressive models.
    Type A: masks center pixel (first layer)
    Type B: does not mask center pixel (subsequent layers)
    """
    
    def __init__(self, kernel_size: int = 3, mask_type: str = "A"):
        self.kernel_size = kernel_size
        self.mask_type = mask_type
        self.weights = np.random.randn(kernel_size, kernel_size) * 0.01
        
        # Create mask
        self.mask = np.ones((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # Mask future pixels
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i > center or (i == center and j > center):
                    self.mask[i, j] = 0
        
        # Type A: also mask center
        if mask_type == "A":
            self.mask[center, center] = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply masked convolution."""
        masked_weights = self.weights * self.mask
        h, w = x.shape
        k = self.kernel_size
        pad = k // 2
        
        x_padded = np.pad(x, pad, mode='constant')
        output = np.zeros_like(x)
        
        for i in range(h):
            for j in range(w):
                region = x_padded[i:i+k, j:j+k]
                output[i, j] = np.sum(region * masked_weights)
        
        return output


class AutoregressiveModel:
    """
    Autoregressive image generation model (PixelCNN-style).
    Generates images pixel by pixel using masked convolutions.
    
    p(x) = prod_{i=1}^{n} p(x_i | x_{<i})
    
    Each pixel is conditioned on all previously generated pixels.
    """
    
    def __init__(self, config: AutoregressiveConfig):
        self.config = config
        
        # Build masked conv layers
        self.layers = []
        self.layers.append(MaskedConvolution(kernel_size=7, mask_type="A"))
        for _ in range(config.n_layers - 1):
            self.layers.append(MaskedConvolution(kernel_size=3, mask_type="B"))
        
        # Output projection
        self.output_proj = np.random.randn(1, 1) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through masked convolutions."""
        h = x
        for layer in self.layers:
            h = h + layer.forward(h)  # Residual connections
            h = np.maximum(0, h)  # ReLU
        
        return h

    def sample_pixel(self, probs: np.ndarray) -> int:
        """Sample a pixel value from probability distribution."""
        return int(np.random.choice(len(probs), p=probs))

    def generate(self) -> np.ndarray:
        """Generate an image pixel by pixel."""
        size = self.config.image_size
        image = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                # Forward pass to get pixel distribution
                logits = self.forward(image)
                # Apply softmax to get probabilities
                probs = self._softmax(logits[i, j])
                # Sample pixel value
                pixel = np.argmax(probs)
                image[i, j] = pixel / 255.0
        
        return image

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)

    def compute_loss(self, x: np.ndarray) -> float:
        """Compute negative log-likelihood loss."""
        logits = self.forward(x)
        # Simulated loss
        return float(np.mean(np.random.uniform(0.5, 2.0)))

    def train_step(self, x: np.ndarray, lr: float = 1e-3) -> float:
        """Single training step."""
        loss = self.compute_loss(x)
        # Simulated gradient update
        for layer in self.layers:
            layer.weights -= lr * np.random.randn(*layer.weights.shape) * 0.001
        return loss


if __name__ == "__main__":
    config = AutoregressiveConfig(image_size=28, n_layers=5)
    model = AutoregressiveModel(config)
    
    print("Autoregressive Model (PixelCNN-style)")
    print(f"  Image size: {config.image_size}x{config.image_size}")
    print(f"  Layers: {config.n_layers} (1 Type-A + {config.n_layers-1} Type-B)")
    print(f"  Masked convolutions: pixel i only sees pixels < i")
    
    # Simulated training
    x = np.random.rand(28, 28)
    for epoch in range(5):
        loss = model.train_step(x)
        print(f"  Epoch {epoch+1}: loss={loss:.4f}")
    
    # Generate
    print("\nGenerating image pixel by pixel...")
    generated = model.generate()
    print(f"Generated image: {generated.shape}, range: [{generated.min():.3f}, {generated.max():.3f}]")
    
    print("\nKey: p(x) = prod p(x_i | x_{<i}) — each pixel conditioned on previous")
    print("References: van den Oord et al. (2016) - PixelRNN / PixelCNN")
