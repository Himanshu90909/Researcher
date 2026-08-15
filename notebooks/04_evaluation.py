"""
Research Notebook 4: Model Evaluation & Benchmarking
Demonstrates LLM-as-a-Judge, visual metrics, and audio quality assessment.

Run: python notebooks/04_evaluation.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.benchmark import (
    MultimodalBenchmark, LLMasJudge, VisualMetrics, AudioMetrics
)


def main():
    print("=" * 60)
    print("Notebook 4: Model Evaluation & Benchmarking")
    print("=" * 60)
    
    # 1. LLM-as-a-Judge
    print("\n--- LLM-as-a-Judge ---")
    judge = LLMasJudge(judge_model="gpt-4")
    
    # Evaluate video summaries
    samples = [
        {
            "task": "Summarize this cooking video",
            "response": "The video shows a chef cooking Italian pasta with fresh ingredients in a modern kitchen, followed by friends enjoying the meal.",
            "reference": "A comprehensive cooking tutorial covering pasta preparation from ingredient selection through plating, with emphasis on technique and presentation.",
        },
        {
            "task": "Describe the main action in the scene",
            "response": "Someone is cutting vegetables on a cutting board.",
            "reference": "A chef demonstrates proper knife techniques for dicing onions and bell peppers with close-up shots.",
        },
        {
            "task": "What is the mood of the video?",
            "response": "Happy and energetic with upbeat background music.",
            "reference": "The video has a warm, inviting atmosphere with soft lighting and gentle acoustic music.",
        },
    ]
    
    for i, sample in enumerate(samples):
        scores = judge.evaluate(
            task=sample["task"],
            response=sample["response"],
            reference=sample["reference"],
        )
        print(f"\n  Sample {i+1}:")
        print(f"    Task: {sample['task']}")
        print(f"    Scores: {scores}")
    
    # Batch evaluation
    batch_results = judge.batch_evaluate(samples)
    avg = np.mean([r["overall"] for r in batch_results])
    print(f"\n  Batch average: {avg:.2f}/10")
    
    # 2. Visual Quality Metrics
    print("\n--- Visual Quality Metrics ---")
    # Create a reference and degraded version
    reference = np.random.rand(64, 64, 3) * 0.8 + 0.1
    degraded = reference + np.random.randn(64, 64, 3) * 0.05
    
    psnr = VisualMetrics.psnr(degraded, reference)
    ssim = VisualMetrics.ssim(degraded, reference)
    lpips = VisualMetrics.lpips_proxy(degraded, reference)
    
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    print(f"  LPIPS: {lpips:.4f}")
    
    # Test with more degradation
    for noise_level in [0.01, 0.05, 0.1, 0.2, 0.3]:
        noisy = reference + np.random.randn(64, 64, 3) * noise_level
        psnr_n = VisualMetrics.psnr(noisy, reference)
        ssim_n = VisualMetrics.ssim(noisy, reference)
        print(f"  Noise={noise_level:.2f}: PSNR={psnr_n:.2f}, SSIM={ssim_n:.4f}")
    
    # 3. Audio Quality Metrics
    print("\n--- Audio Quality Metrics ---")
    clean_audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
    
    for snr_target in [20, 10, 5, 0, -5]:
        noise_power = np.mean(clean_audio**2) / (10 ** (snr_target / 10))
        noisy_audio = clean_audio + np.random.randn(16000) * np.sqrt(noise_power)
        
        stoi = AudioMetrics.stoi_proxy(clean_audio, noisy_audio)
        snr_actual = AudioMetrics.snr_db(clean_audio, noisy_audio - clean_audio)
        print(f"  Target SNR={snr_target:3d} dB: STOI={stoi:.4f}, Actual SNR={snr_actual:.2f}")
    
    # 4. Full Benchmark Report
    print("\n--- Full Benchmark Report ---")
    bench = MultimodalBenchmark(judge_model="gpt-4")
    
    # Visual benchmark
    visual_results = bench.benchmark_visual(degraded, reference)
    print(f"  Visual: {len(visual_results)} metrics")
    
    # Audio benchmark
    enhanced_audio = clean_audio + np.random.randn(16000) * 0.01
    audio_results = bench.benchmark_audio(clean_audio, enhanced_audio)
    print(f"  Audio: {len(audio_results)} metrics")
    
    # Language benchmark
    lang_result = bench.benchmark_language(
        "Summarize video", "A cooking tutorial", "A comprehensive cooking guide"
    )
    print(f"  Language: score={lang_result.score:.2f}")
    
    # Generate report
    report = bench.generate_report()
    print(f"\n  Total metrics: {report['total_metrics']}")
    print(f"  Visual avg: {report['summary']['visual_avg']:.4f}")
    print(f"  Audio avg: {report['summary']['audio_avg']:.4f}")
    print(f"  Language avg: {report['summary']['language_avg']:.4f}")
    
    print("\n✓ Evaluation pipeline verified")
    print("Key: LLM-as-a-Judge provides human-like quality assessment")
    print("Combined with task-specific metrics for comprehensive evaluation")


if __name__ == "__main__":
    main()
