# Researcher: Multimodal AI Research Framework for Video Intelligence & AR/VR

A comprehensive research framework implementing state-of-the-art multimodal AI for video understanding, generative models, LLM agents, speech processing, and AR/VR video generation. Aligned with Meta's research areas including Make-A-Video, Neural Radiance Fields, SceneScript, and Codec Avatars.

> **Original contribution:** `src/research/casa` — *proof-carrying sparse attention*.
> Every pruned attention output ships a deterministic, machine-checkable bound on its
> distance from the exact output, plus an independent verifier and a tamper-evident
> ledger. 2-26% token retention at `eps = 0.01-0.05` with **0 bound violations in 1350
> certificates**, and ~90% of a RAG context dropped with the drift always inside its
> certificate. Theorems, proofs, honest negatives and full experiments:
> **[docs/CASA.md](docs/CASA.md)**.

## Architecture

```
Researcher/
├── src/
│   ├── research/
│   │   └── casa/                      # ★ ORIGINAL CONTRIBUTION (see docs/CASA.md)
│   │       ├── bounds.py              #   Theorem 1 + Theorem 2, certified pattern search
│   │       ├── certificate.py         #   proof-carrying output, independent verifier, ledger
│   │       ├── attention.py           #   certified sparse attention + heuristic baselines
│   │       ├── context.py             #   certified pruning of retrieved RAG context
│   │       └── torch_head.py          #   nn.Module wrapper with autograd through kept tokens
│   ├── models/
│   │   ├── transformer_model.py      # Multimodal Transformer with cross-attention
│   │   ├── diffusion_sr.py            # DDPM diffusion + ESRGAN super-resolution
│   │   ├── vae.py                     # Variational Autoencoder (Kingma & Welling 2014)
│   │   ├── autoregressive.py          # PixelCNN-style autoregressive model
│   │   ├── llm_finetune.py            # SFT, DPO, LoRA/PEFT post-training
│   │   ├── multimodal_fusion.py       # Cross-modal attention + contrastive alignment (CLIP)
│   │   ├── nerf.py                     # Neural Radiance Fields (NeRF) for 3D reconstruction
│   │   ├── gaussian_splatting.py      # 3D Gaussian Splatting for real-time rendering
│   │   └── text_to_3d.py              # Text-to-3D generation (DreamFusion SDS)
│   ├── video/
│   │   ├── scene_detector.py          # Shot boundary detection
│   │   ├── keyframe_extractor.py     # K-Means keyframe extraction
│   │   ├── temporal_action_detector.py # Temporal action localization
│   │   ├── quality_assessment.py      # Video quality metrics (spatial/temporal/perceptual)
│   │   ├── depth_estimation.py        # Monocular/stereo depth for AR compositing
│   │   ├── sixdof_video.py            # 6DoF immersive video with DIBR
│   │   └── scene_understanding.py    # SLAM + plane detection + 3D scene graphs
│   ├── speech/
│   │   ├── enhancer.py                # Speech enhancement (spectral gating + VAD)
│   │   └── voice_cloner.py            # Voice cloning & TTS synthesis
│   ├── agents/
│   │   ├── video_agent.py             # ReAct agent with tool-calling
│   │   └── rag_pipeline.py            # Retrieval-augmented generation
│   ├── evaluation/
│   │   └── benchmark.py               # LLM-as-a-Judge + visual/audio metrics
│   └── utils/
│       └── data_preprocessing.py     # Video & audio preprocessing utilities
├── notebooks/                         # Research notebooks (7)
├── tests/                             # Test suites (20+ tests, incl. 18 for CASA)
├── scripts/                           # Training, evaluation & CASA experiment scripts
├── configs/                           # YAML configurations
├── results/casa/                      # Committed CASA results + figures
├── docs/CASA.md                       # ★ CASA paper: theorems, proofs, experiments
└── RESEARCH_PAPER.md                  # Full research paper documentation
```

## Research Areas

### 1. Multimodal Transformer Architecture
- Multi-head self-attention (Vaswani et al. 2017)
- Cross-modal attention between visual & text features
- Sinusoidal positional encoding
- Vision Transformer patch embedding

### 2. Generative Models
- **DDPM Diffusion** (Ho et al. 2020): Forward/reverse diffusion for image SR
- **ESRGAN** (Wang et al. 2021): GAN-based 4x super-resolution
- **VAE** (Kingma & Welling 2014): Variational autoencoder with β-VAE
- **Autoregressive** (van den Oord et al. 2016): PixelCNN with masked convolutions

### 3. AR/VR Video Generation
- **Neural Radiance Fields** (Mildenhall et al. 2020): Volumetric rendering from posed images
- **3D Gaussian Splatting** (Kerbl et al. 2023): Real-time novel view synthesis
- **Text-to-3D** (Poole et al. 2022): Score Distillation Sampling for 3D generation
- **6DoF Video** (Bemana et al. 2020): Immersive light field video with DIBR
- **Depth Estimation** (Ranftl et al. 2020): Monocular + stereo depth for AR compositing
- **Scene Understanding** (Avetisyan et al. 2024): SLAM + plane detection + scene graphs

### 4. LLM Agents & RAG
- ReAct pattern: Reasoning + Acting with tool-calling
- RAG pipeline: Document indexing, retrieval, augmented generation
- Tools: Scene detection, keyframe extraction, summarization

### 5. Speech Processing
- Spectral gating noise reduction
- Voice Activity Detection (VAD)
- Voice cloning with pitch/rate/spectral transformation
- Text-to-Speech synthesis

### 6. LLM Post-Training
- **SFT**: Supervised fine-tuning on instruction-response pairs
- **DPO** (Rafailov et al. 2023): Direct Preference Optimization
- **LoRA** (Hu et al. 2021): Low-rank adaptation with <1% trainable params

### 7. Model Evaluation
- **LLM-as-a-Judge**: Automated evaluation across 5 dimensions
- **Visual Metrics**: PSNR, SSIM, LPIPS
- **Audio Metrics**: STOI, SNR, PESQ
- **Video Quality**: Spatial, temporal, perceptual assessment

## Meta Research Alignment

| Meta Research Area | Researcher Module |
|---|---|
| Make-A-Video | Text-to-3D, Diffusion Model |
| Immersive Light Field Video (FRL) | NeRF, 6DoF Video |
| SceneScript | Scene Understanding, Scene Graphs |
| Codec Avatars | Voice Cloner, 3D Gaussian Splatting |
| DINOv2 | Multimodal Transformer, Contrastive Alignment |
| Emu Video | Diffusion SR, Autoregressive Model |

### 7. Certified Adaptive Sparse Attention (CASA) — original research

Everything else in this repository reproduces published work. This module is the
one original contribution: **proof-carrying approximation**. Instead of claiming
"we dropped 90% of the tokens and it looked fine", CASA returns the output *plus a
machine-checkable certificate*: a deterministic upper bound on `‖out_sparse −
out_dense‖₂`, together with the witness needed to re-derive it.

```python
from src.research import casa

out, cert = casa.certified_sparse_attention(q, K, V, block_size=16, eps=0.05)
cert.valid                       # bound <= eps, and not a fallback
cert.retention                   # 0.242  -> 76% of tokens dropped
cert.error_bound                 # 0.0411 -> provable, for THIS query
casa.verify_certificate(cert, q, K, V)   # independent re-derivation: PASS
```

The results, with the honest ones included:

| | |
|---|---|
| Theorem 1 | `‖o_S − o‖ ≤ τ(D_S + D_D) ≤ 2τD` for renormalized pruning, any data, factor 2 is tight |
| Theorem 2 | `τ ≤ min(1, exp(Σ_{dropped} (log n_g + u_g) − m_S))`, so `τ` needs only block summaries |
| Guarantee | 1350 certificates, **0 bound violations**, 0 verifier failures (E1) |
| Sparsity | 2-26% of tokens retained at `eps = 0.01-0.05` on selective heads; a flat head correctly refuses to prune (E2) |
| Heuristic control | keeping 90% of the probability mass still moved the output by 0.19 (4× the tolerance) in 17/30 queries; CASA 0/30 (E4) |
| Retrieval | **~90% of RAG context tokens dropped**, gold chunk kept in 12/12 runs, drift always inside its certificate (E6) |
| Auditing | 7/7 tampering classes rejected; certificate bound to its inputs by digest; hash-chained ledger (E8) |
| Cost | not a FlashAttention competitor: same asymptotics, ~11× constant on a NumPy CPU head (E7) |
| Negative result | score-envelope-only certification is vacuous at d=64, for a scaling reason (E5) |

Full write-up with proofs, pseudocode, per-experiment tables and limitations:
**[docs/CASA.md](docs/CASA.md)**. Published version (responsive, print-friendly, figures
inlined): **https://casa-paper.vercel.app** — the source of that page is `site/index.html`,
deployed as a static site with no build step. Reproduce with
`python scripts/run_casa_experiments.py` (~16 s, CPU only) and
`python -m pytest tests/test_casa.py -q` (18 tests).

The point of the module is not that approximate attention can be bounded — that has
been done in the literature — but that the bound is *shipped as an artifact you can
audit*: witness included, verifier independent, minimality re-checked, tampering
detectable.

## Usage

```python
# Run tests
python tests/test_framework.py    # Core framework tests
python tests/test_extended.py     # Extended module tests
python tests/test_arvr.py         # AR/VR module tests
python -m pytest tests/test_casa.py -q   # CASA: 18 tests for the certified-attention paper

# Research notebooks
python notebooks/01_transformer_attention.py
python notebooks/02_generative_models.py
python notebooks/03_agents_rag.py
python notebooks/04_evaluation.py
python notebooks/05_speech_processing.py
python notebooks/06_llm_finetune.py
python notebooks/07_arvr_video.py

# Training & evaluation
python scripts/train.py --model diffusion --epochs 100
python scripts/evaluate.py --model all --judge gpt-4

# CASA experiments (8 studies -> results/casa/ + 3 figures)
python scripts/run_casa_experiments.py
```

## Key References

1. Vaswani et al. (2017) - Attention Is All You Need
2. Ho et al. (2020) - Denoising Diffusion Probabilistic Models
3. Mildenhall et al. (2020) - NeRF: Representing Scenes as Neural Radiance Fields
4. Kerbl et al. (2023) - 3D Gaussian Splatting for Real-Time Rendering
5. Poole et al. (2022) - DreamFusion: Text-to-3D using 2D Diffusion
6. Rafailov et al. (2023) - Direct Preference Optimization
7. Hu et al. (2021) - LoRA: Low-Rank Adaptation
8. Radford et al. (2021) - CLIP: Contrastive Language-Image Pre-training
9. Avetisyan et al. (2024) - SceneScript: Structured 3D Scene Understanding
10. Kingma & Welling (2014) - Auto-Encoding Variational Bayes

## Author

**Himanshu Suthar**  
B.Tech CSE (AI & ML), Lovely Professional University  
GitHub: [github.com/Himanshu90909](https://github.com/Himanshu90909)  
Email: sutharindustry@gmail.com

## License

MIT License - See [LICENSE](LICENSE) for details.
