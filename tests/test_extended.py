"""
Extended tests for new Researcher modules.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_rag_pipeline():
    """Test RAG pipeline indexing and retrieval."""
    from src.agents.rag_pipeline import RAGPipeline
    rag = RAGPipeline()
    
    corpus = ["Video shows cooking", "Outdoor sports scene", "Music performance"]
    rag.index_documents(corpus)
    
    results = rag.retrieve("cooking video", top_k=2)
    assert len(results) <= 2
    assert all(hasattr(r, 'score') for r in results)
    
    gen = rag.generate("What is cooking?", top_k=2)
    assert "answer" in gen
    assert "retrieved" in gen
    print("✓ RAG pipeline: indexing, retrieval, generation")


def test_vae():
    """Test VAE forward pass and generation."""
    from src.models.vae import VAE, VAEConfig
    config = VAEConfig(input_dim=784, hidden_dim=128, latent_dim=32)
    vae = VAE(config)
    
    x = np.random.rand(4, 784)
    outputs = vae.forward(x)
    assert outputs["recon"].shape == (4, 784)
    assert outputs["mu"].shape == (4, 32)
    assert outputs["log_var"].shape == (4, 32)
    
    losses = vae.loss(x, outputs)
    assert losses["total_loss"] >= 0
    assert losses["recon_loss"] >= 0
    assert losses["kl_loss"] >= 0
    
    samples = vae.generate(3)
    assert samples.shape == (3, 784)
    print("✓ VAE: encode, decode, loss, generate")


def test_autoregressive():
    """Test autoregressive model."""
    from src.models.autoregressive import AutoregressiveModel, AutoregressiveConfig
    config = AutoregressiveConfig(image_size=8, n_layers=3)
    model = AutoregressiveModel(config)
    
    x = np.random.rand(8, 8)
    loss = model.train_step(x)
    assert loss >= 0
    
    generated = model.generate()
    assert generated.shape == (8, 8)
    assert np.all(generated >= 0) and np.all(generated <= 1)
    print("✓ Autoregressive: training, generation")


def test_temporal_action_detection():
    """Test temporal action detection."""
    from src.video.temporal_action_detector import TemporalActionDetector
    detector = TemporalActionDetector(
        proposal_length=2.0, stride=1.0, confidence_threshold=0.3
    )
    
    proposals = detector.generate_proposals(duration=10.0)
    assert len(proposals) > 0
    assert all(s < e for s, e in proposals)
    
    detections = detector.detect(video_duration=10.0)
    assert isinstance(detections, list)
    
    summary = detector.summarize_detections(detections)
    assert "total_detections" in summary
    assert "action_distribution" in summary
    print(f"✓ Temporal action detection: {len(detections)} detections in 10s video")


def test_voice_cloner():
    """Test voice cloning and TTS."""
    from src.speech.voice_cloner import VoiceCloner, TextToSpeech, VoiceProfile
    cloner = VoiceCloner(sample_rate=16000)
    
    audio = np.random.randn(16000) * 0.1
    profile = cloner.extract_voice_profile(audio)
    assert profile.pitch_mean > 0
    assert profile.speaking_rate > 0
    
    target = VoiceProfile(pitch_mean=200.0, speaking_rate=1.5)
    converted = cloner.convert_voice(audio, target)
    assert len(converted) > 0
    
    tts = TextToSpeech(sample_rate=16000)
    synthesized = tts.synthesize("Hello world", target)
    assert len(synthesized) > 0
    print("✓ Voice cloning: profile extraction, conversion, TTS")


def test_multimodal_fusion():
    """Test multimodal alignment and fusion."""
    from src.models.multimodal_fusion import MultimodalFusion, ContrastiveAlignment
    fusion = MultimodalFusion(dim=128)
    
    visual = np.random.randn(2, 10, 128)
    text = np.random.randn(2, 8, 128)
    audio = np.random.randn(2, 12, 128)
    
    results = fusion.attention_fusion(visual, text, audio)
    assert "fused" in results
    assert "alignment_score" in results
    assert results["fused"].shape == (2, 10, 128)
    
    # Contrastive loss
    loss = fusion.contrastive.info_nce_loss(visual[:, 0, :], text[:, 0, :])
    assert loss >= 0
    print(f"✓ Multimodal fusion: attention fusion, InfoNCE loss={loss:.4f}")


def test_video_quality_assessment():
    """Test video quality assessment."""
    from src.video.quality_assessment import VideoQualityAssessment
    vqa = VideoQualityAssessment()
    
    frames = [np.random.rand(32, 32, 3) for _ in range(5)]
    report = vqa.assess(frames)
    
    assert 0 <= report.spatial_score <= 1
    assert 0 <= report.temporal_score <= 1
    assert "n_frames" in report.details
    assert report.details["n_frames"] == 5
    print(f"✓ Video quality: spatial={report.spatial_score:.3f}, temporal={report.temporal_score:.3f}")


def test_data_preprocessing():
    """Test data preprocessing utilities."""
    from src.utils.data_preprocessing import AudioPreprocessor
    ap = AudioPreprocessor(sample_rate=16000, n_mels=40)
    
    audio = np.random.randn(16000) * 0.1
    mel_spec = ap.compute_mel_spectrogram(audio)
    assert mel_spec.shape[0] == 40  # n_mels
    print(f"✓ Data preprocessing: mel spectrogram {mel_spec.shape}")


def test_video_preprocessor():
    """Test video frame preprocessing."""
    from src.utils.data_preprocessing import VideoPreprocessor
    vp = VideoPreprocessor(target_size=(32, 32), normalize=True)
    
    frame = np.random.randint(0, 255, (64, 64, 3))
    resized = vp.resize_frame(frame)
    assert resized.shape == (32, 32, 3)
    
    normalized = vp.normalize_frame(resized)
    assert normalized.dtype == np.float32
    
    patches = vp.create_patches(resized, patch_size=16)
    assert patches.shape == (4, 16 * 16 * 3)  # 2x2 patches
    print(f"✓ Video preprocessing: resize, normalize, patches {patches.shape}")


if __name__ == "__main__":
    print("Running Extended Tests\n" + "=" * 40)
    test_rag_pipeline()
    test_vae()
    test_autoregressive()
    test_temporal_action_detection()
    test_voice_cloner()
    test_multimodal_fusion()
    test_video_quality_assessment()
    test_data_preprocessing()
    test_video_preprocessor()
    print("\n✅ All extended tests passed!")
