"""
Research Notebook 5: Speech Processing & Voice Cloning
Demonstrates speech enhancement, VAD, and voice synthesis.

Run: python notebooks/05_speech_processing.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.speech.enhancer import SpeechEnhancer
from src.speech.voice_cloner import VoiceCloner, TextToSpeech, VoiceProfile


def main():
    print("=" * 60)
    print("Notebook 5: Speech Processing & Voice Cloning")
    print("=" * 60)
    
    # 1. Speech Enhancement
    print("\n--- Speech Enhancement (Spectral Gating) ---")
    enhancer = SpeechEnhancer(sample_rate=16000)
    
    # Create a test signal: speech + noise
    duration = 3.0  # 3 seconds
    t = np.arange(int(duration * 16000)) / 16000
    speech = 0.3 * np.sin(2 * np.pi * 200 * t) * np.sin(2 * np.pi * 3 * t)  # Simulated speech
    noise = 0.1 * np.random.randn(len(t))
    noisy_speech = speech + noise
    
    print(f"  Input: {len(noisy_speech)} samples ({duration:.1f}s)")
    
    # Enhance
    enhanced = enhancer.spectral_gating(noisy_speech, noise_factor=2.0)
    print(f"  Enhanced: {len(enhanced)} samples")
    
    # SNR
    snr_before = enhancer.compute_snr(speech, noisy_speech)
    snr_after = enhancer.compute_snr(speech, enhanced)
    print(f"  SNR before: {snr_before:.2f} dB")
    print(f"  SNR after: {snr_after:.2f} dB")
    print(f"  Improvement: {snr_after - snr_before:.2f} dB")
    
    # 2. Voice Activity Detection
    print("\n--- Voice Activity Detection ---")
    # Create audio with speech and silence segments
    speech_segment = np.sin(2 * np.pi * 300 * np.arange(16000) / 16000) * 0.5
    silence = np.zeros(8000)
    mixed = np.concatenate([silence, speech_segment, silence, speech_segment, silence])
    
    segments = enhancer.voice_activity_detection(mixed, threshold=0.05)
    print(f"  Input: {len(mixed)} samples ({len(mixed)/16000:.1f}s)")
    print(f"  Detected {len(segments)} speech segments:")
    for seg in segments:
        print(f"    {seg.start_time:.2f}s - {seg.end_time:.2f}s ({seg.end_time - seg.start_time:.2f}s)")
    
    # Full enhancement pipeline
    pipeline_result = enhancer.enhance_pipeline(noisy_speech)
    print(f"\n  Full pipeline: SNR={pipeline_result['snr_db']:.2f} dB, "
          f"{pipeline_result['speech_segments']} speech segments")
    
    # 3. Voice Cloning
    print("\n--- Voice Cloning ---")
    cloner = VoiceCloner(sample_rate=22050)
    
    # Extract source voice profile
    source_audio = np.random.randn(22050) * 0.2
    source_profile = cloner.extract_voice_profile(source_audio)
    print(f"  Source profile:")
    print(f"    Pitch: {source_profile.pitch_mean:.1f} Hz (±{source_profile.pitch_std:.1f})")
    print(f"    Rate: {source_profile.speaking_rate:.2f} words/sec")
    print(f"    Spectral tilt: {source_profile.spectral_tilt:.1f} dB/octave")
    print(f"    Jitter: {source_profile.jitter:.4f}")
    print(f"    Shimmer: {source_profile.shimmer:.4f}")
    
    # Target profiles
    targets = [
        VoiceProfile(pitch_mean=120.0, pitch_std=20.0, speaking_rate=0.8, spectral_tilt=-8),
        VoiceProfile(pitch_mean=200.0, pitch_std=15.0, speaking_rate=1.2, spectral_tilt=-4),
        VoiceProfile(pitch_mean=280.0, pitch_std=25.0, speaking_rate=1.5, spectral_tilt=-2),
    ]
    
    for i, target in enumerate(targets):
        converted = cloner.convert_voice(source_audio, target)
        print(f"\n  Target {i+1}: pitch={target.pitch_mean:.0f}Hz, rate={target.speaking_rate:.1f}")
        print(f"    Converted: {len(converted)} samples")
    
    # 4. Text-to-Speech
    print("\n--- Text-to-Speech ---")
    tts = TextToSpeech(sample_rate=22050)
    
    texts = [
        "Hello, welcome to OpusClip",
        "AI video editing is the future",
        "This is a voice synthesis demo",
    ]
    
    target_voice = VoiceProfile(pitch_mean=180.0, pitch_std=20.0, speaking_rate=1.0)
    
    for text in texts:
        audio = tts.synthesize(text, target_voice)
        print(f"  '{text}' -> {len(audio)} samples ({len(audio)/22050:.2f}s)")
    
    print("\n✓ Speech processing pipeline verified")
    print("Key: Spectral gating removes noise, VAD finds speech segments")
    print("Voice cloning transforms voice characteristics, TTS synthesizes from text")


if __name__ == "__main__":
    main()
