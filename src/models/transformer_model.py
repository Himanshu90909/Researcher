"""
Multimodal Transformer Model
Implements a Transformer architecture for multimodal understanding
with cross-attention between visual and textual representations.
Based on the Transformer architecture (Vaswani et al., 2017).
"""
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for the multimodal Transformer."""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512
    vocab_size: int = 32000
    image_patch_size: int = 16
    image_size: int = 224


class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # Initialize weights (simplified)
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray,
                                     V: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scaled dot-product attention."""
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)

        if mask is not None:
            scores = scores + mask * -1e9

        attention_weights = self._softmax(scores, axis=-1)
        output = np.matmul(attention_weights, V)
        return output, attention_weights

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through multi-head attention."""
        batch_size, seq_len, _ = x.shape

        Q = np.matmul(x, self.W_q).reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.matmul(x, self.W_k).reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.matmul(x, self.W_v).reshape(batch_size, seq_len, self.n_heads, self.d_v).transpose(0, 2, 1, 3)

        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        output = np.matmul(attention_output, self.W_o)
        return output, attention_weights


class FeedForward:
    """Position-wise Feed-Forward Network (FFN)."""
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, np.matmul(x, self.W1) + self.b1) @ self.W2 + self.b2


class PositionalEncoding:
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 512):
        pe = np.zeros(max_len, d_model)
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe[np.newaxis, :, :]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x + self.pe[:, :x.shape[1], :]


class TransformerLayer:
    """Single Transformer encoder layer (self-attention + FFN + residual)."""
    def __init__(self, config: TransformerConfig):
        self.attention = MultiHeadAttention(config.d_model, config.n_heads)
        self.ffn = FeedForward(config.d_model, config.d_ff)
        self.dropout_rate = config.dropout

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Self-attention with residual
        attn_output, attn_weights = self.attention.forward(x, mask)
        x = self._layer_norm(x + attn_output)

        # FFN with residual
        ffn_output = self.ffn.forward(x)
        x = self._layer_norm(x + ffn_output)

        return x, attn_weights

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)


class MultimodalTransformer:
    """
    Multimodal Transformer for video understanding.
    Processes both visual patches and text tokens with cross-modal attention.

    Architecture:
    1. Visual Encoder: Patch embedding + positional encoding -> Transformer layers
    2. Text Encoder: Token embedding + positional encoding -> Transformer layers
    3. Cross-Modal Attention: Visual features attend to text and vice versa
    """

    def __init__(self, config: TransformerConfig):
        self.config = config
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)

        # Visual encoder
        self.visual_layers = [TransformerLayer(config) for _ in range(config.n_layers)]
        self.patch_embedding = np.random.randn(
            config.image_patch_size * config.image_patch_size * 3, config.d_model
        ) * 0.02

        # Text encoder
        self.text_layers = [TransformerLayer(config) for _ in range(config.n_layers)]
        self.token_embedding = np.random.randn(config.vocab_size, config.d_model) * 0.02

        # Cross-modal fusion
        self.cross_attention = MultiHeadAttention(config.d_model, config.n_heads)
        self.fusion_layer = TransformerLayer(config)

        # Output projection
        self.output_proj = np.random.randn(config.d_model, config.d_model) * 0.02

    def encode_visual(self, patches: np.ndarray) -> Tuple[np.ndarray, List]:
        """Encode visual patches."""
        # patches: (batch, n_patches, patch_dim)
        embedded = np.matmul(patches, self.patch_embedding)
        embedded = self.pos_encoding.forward(embedded)

        all_weights = []
        for layer in self.visual_layers:
            embedded, weights = layer.forward(embedded)
            all_weights.append(weights)

        return embedded, all_weights

    def encode_text(self, tokens: np.ndarray) -> Tuple[np.ndarray, List]:
        """Encode text tokens."""
        embedded = self.token_embedding[tokens]
        embedded = self.pos_encoding.forward(embedded)

        all_weights = []
        for layer in self.text_layers:
            embedded, weights = layer.forward(embedded)
            all_weights.append(weights)

        return embedded, all_weights

    def cross_modal_fusion(self, visual_features: np.ndarray,
                          text_features: np.ndarray) -> np.ndarray:
        """Fuse visual and text features using cross-attention."""
        # Visual features attend to text
        fused, _ = self.cross_attention.forward(
            visual_features,
        )
        fused, _ = self.fusion_layer.forward(fused)
        return np.matmul(fused, self.output_proj)

    def forward(self, visual_patches: np.ndarray, text_tokens: np.ndarray) -> Dict:
        """Full forward pass."""
        visual_features, visual_attn = self.encode_visual(visual_patches)
        text_features, text_attn = self.encode_text(text_tokens)
        fused = self.cross_modal_fusion(visual_features, text_features)

        return {
            "visual_features": visual_features,
            "text_features": text_features,
            "fused_features": fused,
            "visual_attention": visual_attn,
            "text_attention": text_attn,
        }


if __name__ == "__main__":
    print("MultimodalTransformer: Cross-attention between visual patches and text tokens")
    print(f"  d_model={512}, n_heads={8}, n_layers={6}")
    print("  Visual: patch embedding -> positional encoding -> 6 Transformer layers")
    print("  Text: token embedding -> positional encoding -> 6 Transformer layers")
    print("  Fusion: cross-modal attention -> output projection")
    print("\n  Based on Vaswani et al. (2017) 'Attention Is All You Need'")
