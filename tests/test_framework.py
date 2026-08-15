"""
Tests for the Researcher framework.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_transformer_attention():
    """Test multi-head attention computation."""
    from src.models.transformer_model import MultiHeadAttention
    attn = MultiHeadAttention(d_model=64, n_heads=8)
    x = np.random.randn(2, 10, 64)
    output, weights = attn.forward(x)
    assert output.shape == (2, 10, 64), f"Expected (2,10,64), got {output.shape}"
    assert weights.shape == (2, 8, 10, 10), f"Expected (2,8,10,10), got {weights.shape}"
    print("✓ Multi-head attention shapes correct")


def test_diffusion_forward():
    """Test forward diffusion process."""
    from src.models.diffusion_sr import DiffusionModel, DiffusionConfig
    config = DiffusionConfig(num_timesteps=100)
    model = DiffusionModel(config)
    x_0 = np.random.randn(4, 32, 32, 3) * 0.5
    x_t, noise = model.forward_diffusion(x_0, t=50)
    assert x_t.shape == x_0.shape, f"Shape mismatch: {x_t.shape} vs {x_0.shape}"
    assert noise.shape == x_0.shape
    print("✓ Forward diffusion produces correct shapes")


def test_gan_sr():
    """Test GAN super-resolution."""
    from src.models.diffusion_sr import GANSuperResolution
    gan = GANSuperResolution(scale_factor=4)
    low_res = np.random.rand(16, 16, 3)
    high_res = gan.generator_forward(low_res)
    assert high_res.shape == (64, 64, 3), f"Expected (64,64,3), got {high_res.shape}"
    print("✓ GAN super-resolution upscales correctly")


def test_llm_judge():
    """Test LLM-as-a-Judge evaluation."""
    from src.evaluation.benchmark import LLMasJudge
    judge = LLMasJudge(judge_model="gpt-4")
    scores = judge.evaluate(
        task="Summarize the video",
        response="The video shows a person cooking.",
    )
    assert "overall" in scores, "Missing overall score"
    assert all(1 <= scores[d] <= 10 for d in judge.dimensions), "Scores out of range"
    print(f"✓ LLM-as-a-Judge: overall={scores['overall']:.2f}")


def test_visual_metrics():
    """Test visual quality metrics."""
    from src.evaluation.benchmark import VisualMetrics
    target = np.random.rand(32, 32, 3)
    reference = np.random.rand(32, 32, 3)
    psnr = VisualMetrics.psnr(target, reference)
    ssim = VisualMetrics.ssim(target, reference)
    lpips = VisualMetrics.lpips_proxy(target, reference)
    assert psnr >= 0, f"PSNR should be non-negative, got {psnr}"
    assert -1 <= ssim <= 1, f"SSIM should be in [-1,1], got {ssim}"
    assert lpips >= 0, f"LPIPS should be non-negative, got {lpips}"
    print(f"✓ Visual metrics: PSNR={psnr:.2f}, SSIM={ssim:.4f}, LPIPS={lpips:.4f}")


def test_lora():
    """Test LoRA adapter creation."""
    from src.models.llm_finetune import LoRAAdapter, LoRAConfig
    lora = LoRAAdapter(LoRAConfig(r=8, lora_alpha=16))
    lora.create_adapter("q_proj", (768, 768))
    delta = lora.get_delta_weights("q_proj")
    assert delta.shape == (768, 768), f"Expected (768,768), got {delta.shape}"
    params = lora.count_parameters()
    assert params["trainable"] > 0
    print(f"✓ LoRA adapter: r=8, trainable_params={params['trainable']:,}")


def test_dpo_loss():
    """Test DPO loss computation."""
    from src.models.llm_finetune import DPOTrainer, DPOConfig
    dpo = DPOTrainer(DPOConfig(beta=0.1))
    loss = dpo.compute_dpo_loss(
        policy_chosen_logps=-1.0,
        policy_rejected_logps=-2.0,
        ref_chosen_logps=-1.5,
        ref_rejected_logps=-1.5,
    )
    assert loss >= 0, f"DPO loss should be non-negative, got {loss}"
    print(f"✓ DPO loss: {loss:.4f}")


def test_agent_planning():
    """Test video agent task planning."""
    from src.agents.video_agent import VideoAnalysisAgent
    agent = VideoAnalysisAgent()
    plan = agent.plan("summarize the video scenes")
    assert "scene_detection" in plan, "Should plan scene detection"
    assert "summarization" in plan, "Should plan summarization"
    print(f"✓ Agent planning: {plan}")


def test_speech_enhancer():
    """Test speech enhancer pipeline."""
    from src.speech.enhancer import SpeechEnhancer
    enhancer = SpeechEnhancer()
    audio = np.random.randn(16000) * 0.1
    segments = enhancer.voice_activity_detection(audio, threshold=0.001)
    assert isinstance(segments, list)
    print(f"✓ Speech VAD: detected {len(segments)} segments")


def test_benchmark_report():
    """Test benchmark report generation."""
    from src.evaluation.benchmark import MultimodalBenchmark
    bench = MultimodalBenchmark()
    report = bench.generate_report()
    assert "total_metrics" in report
    assert "results" in report
    print(f"✓ Benchmark report: {report['total_metrics']} metrics")


if __name__ == "__main__":
    print("Running Researcher Framework Tests\n" + "=" * 40)
    test_transformer_attention()
    test_diffusion_forward()
    test_gan_sr()
    test_llm_judge()
    test_visual_metrics()
    test_lora()
    test_dpo_loss()
    test_agent_planning()
    test_speech_enhancer()
    test_benchmark_report()
    print("\n✅ All tests passed!")
