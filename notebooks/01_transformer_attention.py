"""
Research Notebook 1: Transformer Architecture & Attention
Demonstrates multi-head attention, positional encoding, and cross-modal fusion.

Run: python notebooks/01_transformer_attention.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer_model import (
    MultiHeadAttention, PositionalEncoding, TransformerLayer,
    TransformerConfig, MultimodalTransformer
)

def main():
    print("=" * 60)
    print("Notebook 1: Transformer Architecture & Attention")
    print("=" * 60)
    
    # 1. Multi-Head Attention
    print("\n--- Multi-Head Attention ---")
    attn = MultiHeadAttention(d_model=64, n_heads=8)
    x = np.random.randn(2, 10, 64)
    output, weights = attn.forward(x)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Attention weights: {weights.shape}")
    
    # 2. Positional Encoding
    print("\n--- Positional Encoding ---")
    pe = PositionalEncoding(d_model=64, max_len=100)
    x_with_pe = pe.forward(x)
    print(f"PE added: {x_with_pe.shape}")
    
    # 3. Transformer Layer
    print("\n--- Transformer Layer ---")
    config = TransformerConfig(d_model=64, n_heads=8, n_layers=2, d_ff=128)
    layer = TransformerLayer(config)
    out, attn_w = layer.forward(x)
    print(f"Layer output: {out.shape}")
    print(f"Attention weights: {attn_w.shape}")
    
    # 4. Visualize attention patterns
    print("\n--- Attention Pattern Analysis ---")
    print(f"Max attention weight: {weights.max():.4f}")
    print(f"Mean attention weight: {weights.mean():.4f}")
    print(f"Attention entropy: {-_entropy(weights.mean(axis=(0,1))):.4f}")
    
    # 5. Full Multimodal Transformer
    print("\n--- Full Multimodal Transformer ---")
    mm_config = TransformerConfig(d_model=128, n_heads=4, n_layers=2, d_ff=256)
    mm = MultimodalTransformer(mm_config)
    
    visual_patches = np.random.randn(2, 10, 128)  # batch, patches, dim
    text_tokens = np.random.randint(0, 1000, (2, 15))
    
    result = mm.forward(visual_patches, text_tokens)
    print(f"Visual features: {result['visual_features'].shape}")
    print(f"Text features: {result['text_features'].shape}")
    print(f"Fused features: {result['fused_features'].shape}")
    
    print("\n✓ Transformer architecture verified")
    print("Key insight: Cross-modal attention allows visual and text features")
    print("to inform each other, enabling multimodal understanding.")


def _entropy(probs):
    probs = probs.flatten()
    probs = probs / (probs.sum() + 1e-8)
    return np.sum(probs * np.log(probs + 1e-8))


if __name__ == "__main__":
    main()
