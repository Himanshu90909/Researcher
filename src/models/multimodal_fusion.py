"""
Multimodal Alignment & Fusion Module
Implements cross-modal alignment between visual, audio, and text
representations for unified multimodal understanding.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModalityFeatures:
    """Features from a single modality."""
    modality: str  # 'visual', 'audio', 'text'
    features: np.ndarray
    timestamps: Optional[List[float]] = None


class CrossModalAttention:
    """
    Cross-modal attention mechanism for aligning features
    from different modalities.
    """
    
    def __init__(self, dim: int = 512, n_heads: int = 8):
        self.dim = dim
        self.n_heads = n_heads
        self.d_k = dim // n_heads
        
        # Projection matrices for each modality pair
        self.projections = {
            ('visual', 'text'): np.random.randn(dim, dim) * 0.02,
            ('text', 'visual'): np.random.randn(dim, dim) * 0.02,
            ('audio', 'visual'): np.random.randn(dim, dim) * 0.02,
            ('audio', 'text'): np.random.randn(dim, dim) * 0.02,
            ('visual', 'audio'): np.random.randn(dim, dim) * 0.02,
            ('text', 'audio'): np.random.randn(dim, dim) * 0.02,
        }
    
    def attend(self, query_mod: str, key_mod: str, 
               query_feats: np.ndarray, key_feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cross-attention from query modality to key modality."""
        proj_key = (query_mod, key_mod)
        if proj_key not in self.projections:
            return query_feats, np.eye(query_feats.shape[0])
        
        W = self.projections[proj_key]
        Q = query_feats @ W
        K = key_feats
        V = key_feats
        
        # Multi-head attention
        batch_size, q_len, _ = Q.shape
        k_len = K.shape[1]
        Q_h = Q.reshape(batch_size, q_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K_h = K.reshape(batch_size, k_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V_h = V.reshape(batch_size, k_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        scores = (Q_h @ K_h.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-8)
        
        output = attn_weights @ V_h
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, q_len, self.dim)
        
        return output, attn_weights.mean(axis=1)


class ContrastiveAlignment:
    """
    Contrastive learning for cross-modal alignment.
    Uses InfoNCE loss to align visual and text representations
    in a shared embedding space.
    
    Based on CLIP (Radford et al. 2021) approach.
    """
    
    def __init__(self, dim: int = 512, temperature: float = 0.07):
        self.dim = dim
        self.temperature = temperature
    
    def info_nce_loss(self, visual_embeddings: np.ndarray, 
                      text_embeddings: np.ndarray) -> float:
        """
        InfoNCE loss for contrastive alignment.
        L = -log(exp(sim(v,t)/τ) / sum(exp(sim(v,t')/τ)))
        """
        # Normalize embeddings
        v = visual_embeddings / (np.linalg.norm(visual_embeddings, axis=1, keepdims=True) + 1e-8)
        t = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Similarity matrix
        logits = (v @ t.T) / self.temperature
        
        # Loss (symmetric)
        labels = np.arange(len(v))
        loss_v2t = -np.mean(np.log(
            np.exp(logits[labels, labels]) / 
            (np.sum(np.exp(logits), axis=1) + 1e-8)
        ))
        loss_t2v = -np.mean(np.log(
            np.exp(logits[labels, labels]) / 
            (np.sum(np.exp(logits), axis=0) + 1e-8)
        ))
        
        return float((loss_v2t + loss_t2v) / 2)
    
    def compute_alignment_score(self, visual_emb: np.ndarray, 
                                 text_emb: np.ndarray) -> float:
        """Compute alignment score between paired embeddings."""
        v = visual_emb / (np.linalg.norm(visual_emb) + 1e-8)
        t = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        return float(np.dot(v, t))


class MultimodalFusion:
    """
    Multimodal fusion strategies for combining features
    from multiple modalities.
    """
    
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.cross_attn = CrossModalAttention(dim)
        self.contrastive = ContrastiveAlignment(dim)
    
    def early_fusion(self, modality_features: List[np.ndarray]) -> np.ndarray:
        """Early fusion: concatenate features before processing."""
        return np.concatenate(modality_features, axis=-1)
    
    def late_fusion(self, modality_predictions: List[np.ndarray],
                    weights: Optional[List[float]] = None) -> np.ndarray:
        """Late fusion: weighted average of per-modality predictions."""
        if weights is None:
            weights = [1.0 / len(modality_predictions)] * len(modality_predictions)
        
        weighted = [w * pred for w, pred in zip(weights, modality_predictions)]
        return np.sum(weighted, axis=0)
    
    def attention_fusion(self, visual: np.ndarray, text: np.ndarray, 
                         audio: Optional[np.ndarray] = None) -> Dict:
        """Attention-based cross-modal fusion."""
        results = {}
        
        # Visual <-> Text
        v2t, v2t_weights = self.cross_attn.attend('visual', 'text', visual, text)
        t2v, t2v_weights = self.cross_attn.attend('text', 'visual', text, visual)
        results['visual_to_text'] = v2t
        results['text_to_visual'] = t2v
        results['v2t_weights'] = v2t_weights
        results['t2v_weights'] = t2v_weights
        
        # Audio fusion if provided
        if audio is not None:
            a2v, _ = self.cross_attn.attend('audio', 'visual', audio, visual)
            a2t, _ = self.cross_attn.attend('audio', 'text', audio, text)
            results['audio_to_visual'] = a2v
            results['audio_to_text'] = a2t
        
        # Combined representation (mean pool each to (batch, dim) then average)
        v2t_pooled = v2t.mean(axis=1)  # (batch, dim)
        t2v_pooled = t2v.mean(axis=1)  # (batch, dim)
        fused_pooled = (v2t_pooled + t2v_pooled) / 2
        if audio is not None:
            a2v_pooled = results['audio_to_visual'].mean(axis=1)
            a2t_pooled = results['audio_to_text'].mean(axis=1)
            fused_pooled = (fused_pooled + a2v_pooled + a2t_pooled) / 3
        # Broadcast back to visual seq length for output
        fused = np.broadcast_to(fused_pooled[:, None, :], v2t.shape).copy()
        
        results['fused'] = fused
        results['alignment_score'] = self.contrastive.compute_alignment_score(
            visual.mean(axis=1)[0], text.mean(axis=1)[0]
        )
        
        return results


if __name__ == "__main__":
    print("Multimodal Alignment & Fusion")
    print("=" * 50)
    
    fusion = MultimodalFusion(dim=256)
    
    # Simulated features
    visual = np.random.randn(2, 10, 256)
    text = np.random.randn(2, 8, 256)
    audio = np.random.randn(2, 12, 256)
    
    # Attention fusion
    results = fusion.attention_fusion(visual, text, audio)
    print(f"\nFusion results:")
    print(f"  Visual -> Text: {results['visual_to_text'].shape}")
    print(f"  Text -> Visual: {results['text_to_visual'].shape}")
    print(f"  Audio -> Visual: {results['audio_to_visual'].shape}")
    print(f"  Fused: {results['fused'].shape}")
    print(f"  Alignment score: {results['alignment_score']:.4f}")
    
    # Contrastive loss
    loss = fusion.contrastive.info_nce_loss(visual[:, 0, :], text[:, 0, :])
    print(f"  InfoNCE loss: {loss:.4f}")
    
    print("\nFusion strategies: Early (concat), Late (weighted avg), Attention (cross-modal)")
    print("Reference: Radford et al. (2021) - CLIP: Contrastive Language-Image Pre-training")
