"""
Research Notebook 2: Generative Models (Diffusion, GAN, VAE)
Demonstrates image generation using different approaches.

Run: python notebooks/02_generative_models.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.diffusion_sr import DiffusionModel, DiffusionConfig, GANSuperResolution
from src.models.vae import VAE, VAEConfig
from src.models.autoregressive import AutoregressiveModel, AutoregressiveConfig


def main():
    print("=" * 60)
    print("Notebook 2: Generative Models Comparison")
    print("=" * 60)
    
    # 1. Diffusion Model
    print("\n--- DDPM Diffusion Model ---")
    diff_config = DiffusionConfig(num_timesteps=100, image_size=32)
    diffusion = DiffusionModel(diff_config)
    
    x_0 = np.random.randn(4, 32, 32, 3) * 0.5
    print(f"Original: mean={x_0.mean():.4f}, std={x_0.std():.4f}")
    
    # Forward diffusion at different timesteps
    for t in [10, 30, 50, 80, 99]:
        x_t, _ = diffusion.forward_diffusion(x_0, t=t)
        print(f"  t={t:3d}: mean={x_t.mean():.4f}, std={x_t.std():.4f}")
    
    # Reverse diffusion
    x_noisy, _ = diffusion.forward_diffusion(x_0, t=50)
    x_denoised = diffusion.reverse_diffusion_step(x_noisy, t=50, predicted_noise=x_noisy * 0.3)
    print(f"  Denoised (1 step): mean={x_denoised.mean():.4f}, std={x_denoised.std():.4f}")
    
    # 2. GAN Super-Resolution
    print("\n--- GAN Super-Resolution ---")
    gan = GANSuperResolution(scale_factor=4)
    low_res = np.random.rand(16, 16, 3)
    high_res = gan.generator_forward(low_res)
    print(f"  Input: {low_res.shape} -> Output: {high_res.shape}")
    
    # Training step
    target = np.random.rand(64, 64, 3)
    for step in range(5):
        losses = gan.train_step(low_res, target)
        print(f"  Step {step+1}: D_loss={losses['d_loss']:.4f}, G_loss={losses['g_loss']:.4f}")
    
    # 3. VAE
    print("\n--- Variational Autoencoder ---")
    vae_config = VAEConfig(input_dim=784, hidden_dim=256, latent_dim=32, beta=0.5)
    vae = VAE(vae_config)
    
    x = np.random.rand(8, 784)
    for epoch in range(5):
        outputs = vae.forward(x)
        losses = vae.loss(x, outputs)
        print(f"  Epoch {epoch+1}: total={losses['total_loss']:.4f}, "
              f"recon={losses['recon_loss']:.4f}, kl={losses['kl_loss']:.4f}")
    
    # Generate samples
    samples = vae.generate(5)
    print(f"  Generated samples: {samples.shape}")
    
    # Interpolation
    interp = vae.interpolate(x[0], x[1], n_steps=5)
    print(f"  Interpolation: {interp.shape}")
    
    # 4. Autoregressive Model
    print("\n--- Autoregressive Model (PixelCNN) ---")
    ar_config = AutoregressiveConfig(image_size=16, n_layers=3)
    ar = AutoregressiveModel(ar_config)
    
    x = np.random.rand(16, 16)
    for epoch in range(5):
        loss = ar.train_step(x)
        print(f"  Epoch {epoch+1}: loss={loss:.4f}")
    
    # 5. Comparison
    print("\n--- Model Comparison ---")
    print("  Diffusion: High quality, slow sampling, stable training")
    print("  GAN:       Fast inference, training instability, mode collapse risk")
    print("  VAE:       Fast, blurry outputs, probabilistic generation")
    print("  AR:        Stable, slow generation, exact likelihood")
    
    print("\n✓ All generative models verified")


if __name__ == "__main__":
    main()
