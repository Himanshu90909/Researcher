# Multimodal Transformer Model — Logic & Architecture

> File: `src/models/transformer_model.py` (215 lines)
> Based on Vaswani et al. (2017) — *Attention Is All You Need*

---

## Overview

The `MultimodalTransformer` implements a dual-encoder architecture for video understanding. It processes **visual patches** (image regions) and **text tokens** (words) through separate Transformer encoder stacks, then fuses them via **cross-modal attention** — allowing visual features to attend to relevant text and vice versa.

This is the same ViT + BERT + cross-attention pattern used in Meta's FLAVA and ImageBind for multimodal understanding.

```
Image patches ──→ patch embed ──→ +pos enc ──→ 6× [attention → FFN] ──→ visual features ─┐
                                                                                         ├─→ cross-attention ──→ fusion layer ──→ output
Text tokens ───→ token embed ──→ +pos enc ──→ 6× [attention → FFN] ──→ text features ─────┘
```

---

## 1. TransformerConfig (Blueprint)

```python
@dataclass
class TransformerConfig:
    d_model: int = 512          # Embedding dimensionality
    n_heads: int = 8            # Parallel attention heads
    n_layers: int = 6           # Stacked encoder layers per modality
    d_ff: int = 2048            # Feed-forward hidden dimension
    dropout: float = 0.1       # Regularization rate
    max_seq_len: int = 512     # Maximum sequence length
    vocab_size: int = 32000    # Text vocabulary size
    image_patch_size: int = 16 # Patch size for image tokenization
    image_size: int = 224      # Input image resolution
```

**Design choices:**
- `d_model=512` with `n_heads=8` → each head has `d_k=64` dimensions
- `n_layers=6` per encoder → 12 total Transformer layers (6 visual + 6 text)
- `d_ff=2048` = 4× expansion in the feed-forward network (standard ratio from the original paper)
- `image_patch_size=16` → each 224×224 image produces 196 patches (14×14 grid)

---

## 2. MultiHeadAttention (Core Engine)

**Reference equation:** `Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V`

### Why multiple heads?

A single attention head must choose what to focus on — but language and vision have multiple simultaneous relationships (syntax, semantics, coreference, spatial proximity). With 8 heads, each can specialize on different patterns. Head 1 might learn subject→verb attention, Head 2 might learn pronoun→antecedent, Head 3 might capture long-range dependencies. The `W_o` projection then combines all perspectives.

### Step-by-step data flow

#### Step 1 — Linear projections (Q, K, V)

```python
Q = np.matmul(x, self.W_q)  # (batch, seq, 512) → (batch, seq, 512)
K = np.matmul(x, self.W_k)
V = np.matmul(x, self.W_v)
```

The same input `x` is projected three ways:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I pass along?"

Each projection is a learned linear map (`512 × 512` weight matrix).

#### Step 2 — Reshape to multi-head

```python
Q = Q.reshape(batch, seq, n_heads, d_k).transpose(0, 2, 1, 3)
# (batch, seq, 512) → (batch, seq, 8, 64) → (batch, 8, seq, 64)
```

The 512-dim vector is split into 8 chunks of 64. Each head operates independently on its own 64-dim subspace. This is the key trick — different heads work on different slices of the representation.

#### Step 3 — Scaled dot-product attention

```python
scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)
# (batch, 8, seq, 64) @ (batch, 8, 64, seq) → (batch, 8, seq, seq)

attention_weights = softmax(scores, axis=-1)  # Row-wise normalization
output = np.matmul(attention_weights, V)
# (batch, 8, seq, seq) @ (batch, 8, seq, 64) → (batch, 8, seq, 64)
```

- `Q @ Kᵀ` produces a `seq × seq` similarity matrix — how much each token relates to every other
- `/ √d_k` prevents large dot products from pushing softmax into saturation (vanishing gradients)
- `softmax` converts scores to probabilities (each row sums to 1)
- Multiply by `V` to get the weighted sum of values

#### Step 4 — Concatenate heads and project

```python
attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
# (batch, 8, seq, 64) → (batch, seq, 8, 64) → (batch, seq, 512)

output = np.matmul(attention_output, self.W_o)
# Final linear projection mixes information across heads
```

The 8 heads are stitched back into 512 dimensions, then `W_o` combines insights across heads.

### Masking (optional)

```python
if mask is not None:
    scores = scores + mask * -1e9
```

Masking sets disallowed positions to `-∞` before softmax, producing zero attention weights. Used for causal masking (autoregressive generation) and padding.

---

## 3. FeedForward (Position-wise FFN)

```python
def forward(self, x):
    return np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2
```

**Architecture:** Two linear layers with a ReLU activation in between.
- Expands from `512 → 2048 → 512`
- Applied independently to each position (token) in the sequence
- Gives the model non-linear transformation capacity that pure attention cannot do alone

**Why it matters:** Attention mixes information *across* tokens. The FFN transforms each token's representation *within* itself. Every transformer layer alternates: attention (mix) → FFN (transform).

---

## 4. PositionalEncoding (Injecting Order)

```python
pe[:, 0::2] = np.sin(position * div_term)   # Even indices: sine
pe[:, 1::2] = np.cos(position * div_term)   # Odd indices: cosine
```

**The problem:** Self-attention is permutation-invariant — it has no concept of token order. "The cat sat" and "sat cat The" produce identical attention scores.

**The solution:** Sinusoidal positional encoding injects position information using alternating sin/cos at geometrically increasing frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- **Low frequencies** (small `i`): capture long-range position
- **High frequencies** (large `i`): capture local position
- Added to embeddings at the start of both encoders

**Why sin/cos?** The sinusoidal form allows the model to learn relative positions, since `PE(pos+k)` can be expressed as a linear function of `PE(pos)`.

---

## 5. TransformerLayer (One Block of the Stack)

```python
def forward(self, x):
    # Self-attention with residual connection
    attn_output, attn_weights = self.attention.forward(x, mask)
    x = self._layer_norm(x + attn_output)

    # Feed-forward with residual connection
    ffn_output = self.ffn.forward(x)
    x = self._layer_norm(x + ffn_output)

    return x, attn_weights
```

This is the classic "post-LN" Transformer block:

```
        ┌───────────────────────────┐
   x ──→│  Multi-Head Attention     │──→ + ──→ LayerNorm ──→ x'
        └───────────────────────────┘              │
        ┌───────────────────────────┐               │
   x' ─→│  Feed-Forward Network     │──→ + ──→ LayerNorm ──→ x''
        └───────────────────────────┘
```

**Residual connections** (`x + attn_output`): Let gradients flow through deep stacks, preventing the vanishing gradient problem. Without these, stacking 6+ layers would be difficult.

**Layer normalization** (`(x - mean) / (std + ε)`): Stabilizes training by normalizing activations to zero mean and unit variance. Applied after each sub-layer.

**6 of these layers are stacked** in each encoder (visual and text).

---

## 6. MultimodalTransformer (Full Model)

### 6.1 Visual Encoder

```python
def encode_visual(self, patches):
    # patches: (batch, n_patches, 768) — each 16×16×3 patch flattened
    embedded = np.matmul(patches, self.patch_embedding)  # → (batch, n_patches, 512)
    embedded = self.pos_encoding.forward(embedded)        # Add positional encoding

    for layer in self.visual_layers:  # 6 TransformerLayers
        embedded, weights = layer.forward(embedded)

    return embedded, all_weights
```

**Pipeline:** Raw image patches → linear projection to 512-dim → positional encoding → 6 Transformer layers → visual features

This is the ViT (Vision Transformer) pattern from Dosovitskiy et al. (2021).

### 6.2 Text Encoder

```python
def encode_text(self, tokens):
    embedded = self.token_embedding[tokens]  # Lookup: token ID → 512-dim vector
    embedded = self.pos_encoding.forward(embedded)

    for layer in self.text_layers:  # 6 TransformerLayers
        embedded, weights = layer.forward(embedded)

    return embedded, all_weights
```

**Pipeline:** Token IDs → embedding lookup → positional encoding → 6 Transformer layers → text features

This is the BERT pattern from Devlin et al. (2019).

### 6.3 Cross-Modal Fusion

```python
def cross_modal_fusion(self, visual_features, text_features):
    # Visual features attend to text features
    fused, _ = self.cross_attention.forward(visual_features)
    fused, _ = self.fusion_layer.forward(fused)
    return np.matmul(fused, self.output_proj)
```

This is the crucial bridge between modalities. The cross-attention allows each visual patch to ask: *"Which words in the text are relevant to me?"* and pull in that information. A final Transformer layer + linear projection produces the fused output.

**Example:** If the text says "a dog running in the park" and the image shows a park scene, cross-attention lets the visual patch corresponding to the dog attend to the word "dog," the patch showing grass attend to "park," etc.

### 6.4 Full Forward Pass

```python
def forward(self, visual_patches, text_tokens):
    visual_features, visual_attn = self.encode_visual(visual_patches)
    text_features, text_attn = self.encode_text(text_tokens)
    fused = self.cross_modal_fusion(visual_features, text_features)

    return {
        "visual_features": visual_features,
        "text_features": text_features,
        "fused_features": fused,
        "visual_attention": visual_attn,  # Per-layer attention weights
        "text_attention": text_attn,
    }
```

Returns features from all stages plus attention weights from every layer — useful for interpretability and visualization.

---

## Complete Data Flow

```
INPUT:
  Image (224×224) → 196 patches (16×16×3 = 768-dim each)
  Text → token IDs (e.g., [101, 2023, 2003, ...])

VISUAL ENCODER:
  patches (batch, 196, 768)
    → @ patch_embedding → (batch, 196, 512)
    → + positional encoding
    → TransformerLayer × 6:
        ├─ MultiHeadAttention (8 heads × 64 dims)
        ├─ Add & LayerNorm
        ├─ FeedForward (512 → 2048 → 512)
        └─ Add & LayerNorm
    → visual_features (batch, 196, 512)

TEXT ENCODER:
  tokens (batch, seq_len)
    → token_embedding lookup → (batch, seq_len, 512)
    → + positional encoding
    → TransformerLayer × 6:
        ├─ MultiHeadAttention (8 heads × 64 dims)
        ├─ Add & LayerNorm
        ├─ FeedForward (512 → 2048 → 512)
        └─ Add & LayerNorm
    → text_features (batch, seq_len, 512)

CROSS-MODAL FUSION:
  visual_features + text_features
    → Cross-Attention (visual queries text)
    → Fusion TransformerLayer
    → @ output_proj
    → fused_features (batch, 196, 512)

OUTPUT:
  {
    visual_features, text_features, fused_features,
    visual_attention (6 layers), text_attention (6 layers)
  }
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| d_model | 512 | Balance between expressiveness and compute (original Transformer used 512) |
| n_heads | 8 | Enough diversity in attention patterns without over-fragmenting dimensions |
| n_layers | 6 | Sufficient depth for complex representations; matches original encoder |
| d_ff | 2048 (4× d_model) | Standard expansion ratio from Vaswani et al. |
| Patch size | 16×16 | Standard ViT patch size; 196 tokens per 224×224 image |
| Normalization | Post-LN | Original Transformer style (LayerNorm after residual addition) |
| Encoding | Sinusoidal | Allows learning of relative positions; no learned parameters needed |
| Dual encoder | Separate visual + text | Each modality gets specialized representations before fusion |
| Fusion | Cross-attention | Allows fine-grained alignment between specific patches and words |

---

## Component Summary

| Class | Role | Key Method |
|---|---|---|
| `TransformerConfig` | Configuration dataclass | — |
| `MultiHeadAttention` | Core attention mechanism | `scaled_dot_product_attention()` |
| `FeedForward` | Per-token non-linear transform | `forward()` |
| `PositionalEncoding` | Injects position information | `forward()` |
| `TransformerLayer` | Self-attention + FFN block | `forward()` |
| `MultimodalTransformer` | Full dual-encoder + fusion | `forward()`, `encode_visual()`, `encode_text()`, `cross_modal_fusion()` |

---

## References

1. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Dosovitskiy, A. et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale (ViT).* ICLR.
3. Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
4. Radford, A. et al. (2021). *CLIP: Contrastive Language-Image Pre-training.* ICML.
5. Singh, A. et al. (2022). *FLAVA: A Foundational Language And Vision Alignment Model.* CVPR.

---

## Author

**Himanshu Suthar**  
B.Tech CSE (AI & ML), Lovely Professional University  
GitHub: [github.com/Himanshu90909](https://github.com/Himanshu90909)
