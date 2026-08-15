"""
Variational Autoencoder (VAE) for Image Generation
Implements encoder-decoder architecture with reparameterization trick
for generative image modeling.
Based on Kingma & Welling (2014) 'Auto-Encoding Variational Bayes'.
"""
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


@dataclass
class VAEConfig:
    """Configuration for VAE."""
    input_dim: int = 784  # 28x28
    hidden_dim: int = 512
    latent_dim: int = 64
    learning_rate: float = 1e-3
    beta: float = 1.0  # For beta-VAE ( Higgins et al. 2017)


class VAE:
    """
    Variational Autoencoder for image generation.
    
    Architecture:
    - Encoder: x -> h -> (mu, log_var) -> z (reparameterization)
    - Decoder: z -> h -> x_recon
    
    Loss = Reconstruction Loss + KL Divergence
    KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    """

    def __init__(self, config: VAEConfig):
        self.config = config
        
        # Encoder weights
        self.enc_w1 = np.random.randn(config.input_dim, config.hidden_dim) * 0.01
        self.enc_b1 = np.zeros(config.hidden_dim)
        self.enc_w_mu = np.random.randn(config.hidden_dim, config.latent_dim) * 0.01
        self.enc_b_mu = np.zeros(config.latent_dim)
        self.enc_w_logvar = np.random.randn(config.hidden_dim, config.latent_dim) * 0.01
        self.enc_b_logvar = np.zeros(config.latent_dim)
        
        # Decoder weights
        self.dec_w1 = np.random.randn(config.latent_dim, config.hidden_dim) * 0.01
        self.dec_b1 = np.zeros(config.hidden_dim)
        self.dec_w2 = np.random.randn(config.hidden_dim, config.input_dim) * 0.01
        self.dec_b2 = np.zeros(config.input_dim)

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode input to latent space parameters (mu, log_var)."""
        h = np.maximum(0, x @ self.enc_w1 + self.enc_b1)
        mu = h @ self.enc_w_mu + self.enc_b_mu
        log_var = h @ self.enc_w_logvar + self.enc_b_logvar
        return mu, log_var

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        """Reparameterization trick: z = mu + std * epsilon."""
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vector to reconstruction."""
        h = np.maximum(0, z @ self.dec_w1 + self.dec_b1)
        recon = _sigmoid(h @ self.dec_w2 + self.dec_b2)
        return recon

    def forward(self, x: np.ndarray) -> Dict:
        """Full forward pass."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "log_var": log_var, "z": z}

    def loss(self, x: np.ndarray, outputs: Dict) -> Dict:
        """Compute VAE loss (reconstruction + KL)."""
        recon = outputs["recon"]
        mu = outputs["mu"]
        log_var = outputs["log_var"]
        
        # Reconstruction loss (BCE)
        recon_loss = np.mean(
            -x * np.log(recon + 1e-8) - (1 - x) * np.log(1 - recon + 1e-8),
            axis=-1
        )
        
        # KL divergence
        kl_loss = -0.5 * np.mean(
            1 + log_var - mu**2 - np.exp(log_var),
            axis=-1
        )
        
        total_loss = np.mean(recon_loss) + self.config.beta * np.mean(kl_loss)
        
        return {
            "total_loss": float(total_loss),
            "recon_loss": float(np.mean(recon_loss)),
            "kl_loss": float(np.mean(kl_loss)),
        }

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate samples from the latent space."""
        z = np.random.randn(n_samples, self.config.latent_dim)
        return self.decode(z)

    def interpolate(self, x1: np.ndarray, x2: np.ndarray, n_steps: int = 10) -> np.ndarray:
        """Interpolate between two samples in latent space."""
        mu1, _ = self.encode(x1.reshape(1, -1))
        mu2, _ = self.encode(x2.reshape(1, -1))
        
        interpolations = []
        for alpha in np.linspace(0, 1, n_steps):
            z = alpha * mu1 + (1 - alpha) * mu2
            interpolations.append(self.decode(z))
        
        return np.array(interpolations).squeeze()


if __name__ == "__main__":
    config = VAEConfig(input_dim=784, hidden_dim=256, latent_dim=32)
    vae = VAE(config)
    
    # Simulated training
    print("VAE (Variational Autoencoder)")
    print(f"  Input dim: {config.input_dim}, Hidden: {config.hidden_dim}, Latent: {config.latent_dim}")
    print(f"  Beta: {config.beta} (beta-VAE for disentanglement)")
    
    x = np.random.rand(16, 784)
    
    for epoch in range(5):
        outputs = vae.forward(x)
        losses = vae.loss(x, outputs)
        print(f"  Epoch {epoch+1}: total={losses['total_loss']:.4f}, "
              f"recon={losses['recon_loss']:.4f}, kl={losses['kl_loss']:.4f}")
    
    # Generate
    samples = vae.generate(5)
    print(f"\nGenerated {len(samples)} samples: {samples.shape}")
    
    # Interpolate
    interp = vae.interpolate(x[0], x[1], n_steps=5)
    print(f"Interpolation: {interp.shape}")
    
    print("\nReferences:")
    print("  Kingma & Welling (2014) - Auto-Encoding Variational Bayes")
    print("  Higgins et al. (2017) - beta-VAE: Learning Basic Visual Concepts")
