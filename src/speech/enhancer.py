"""
Speech Enhancement Module
Implements noise reduction, voice activity detection (VAD),
and audio quality metrics for speech processing.
"""
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class AudioSegment:
    """Represents an audio segment."""
    audio: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    is_speech: bool = True


class SpeechEnhancer:
    """
    Speech enhancement using spectral gating for noise reduction.
    Implements noise profile estimation and spectral subtraction.
    """

    def __init__(self, sample_rate: int = 16000, frame_length: int = 2048,
                 hop_length: int = 512, noise_frames: int = 6):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.noise_frames = noise_frames

    def stft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Short-Time Fourier Transform."""
        frames = []
        for i in range(0, len(audio) - self.frame_length, self.hop_length):
            frame = audio[i:i + self.frame_length] * np.hanning(self.frame_length)
            frames.append(np.fft.rfft(frame))
        return np.array(frames), np.array(frames)

    def istft(self, stft_frames: np.ndarray) -> np.ndarray:
        """Inverse STFT."""
        output = np.zeros(len(stft_frames) * self.hop_length + self.frame_length)
        for i, frame in enumerate(stft_frames):
            start = i * self.hop_length
            output[start:start + self.frame_length] += np.fft.irfft(frame, self.frame_length) * np.hanning(self.frame_length)
        return output

    def estimate_noise_profile(self, stft_frames: np.ndarray) -> np.ndarray:
        """Estimate noise profile from first N frames."""
        noise_frames = stft_frames[:self.noise_frames]
        return np.mean(np.abs(noise_frames), axis=0)

    def spectral_gating(self, audio: np.ndarray, noise_factor: float = 2.0) -> np.ndarray:
        """
        Apply spectral gating noise reduction.

        Args:
            audio: Input audio signal
            noise_factor: Aggressiveness of noise reduction

        Returns:
            Enhanced audio signal
        """
        # Normalize
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # STFT
        stft_frames, _ = self.stft(audio)
        magnitudes = np.abs(stft_frames)
        phases = np.angle(stft_frames)

        # Estimate noise
        noise_profile = self.estimate_noise_profile(magnitudes)

        # Spectral gating
        mask = np.maximum(magnitudes - noise_factor * noise_profile, 0) / (magnitudes + 1e-10)

        # Apply mask
        enhanced_magnitudes = magnitudes * mask
        enhanced_stft = enhanced_magnitudes * np.exp(1j * phases)

        # ISTFT
        enhanced_audio = self.istft(enhanced_stft)

        return enhanced_audio

    def voice_activity_detection(self, audio: np.ndarray,
                                  threshold: float = 0.01,
                                  frame_length: int = 1024) -> List[AudioSegment]:
        """
        Detect speech segments using energy-based VAD.

        Args:
            audio: Input audio signal
            threshold: Energy threshold for speech detection
            frame_length: Frame length for energy computation

        Returns:
            List of AudioSegment objects
        """
        segments = []
        in_speech = False
        speech_start = 0

        for i in range(0, len(audio) - frame_length, frame_length):
            frame = audio[i:i + frame_length]
            energy = np.mean(frame ** 2)

            is_speech = energy > threshold

            if is_speech and not in_speech:
                in_speech = True
                speech_start = i
            elif not is_speech and in_speech:
                in_speech = False
                segments.append(AudioSegment(
                    audio=audio[speech_start:i],
                    sample_rate=self.sample_rate,
                    start_time=speech_start / self.sample_rate,
                    end_time=i / self.sample_rate,
                    is_speech=True,
                ))

        if in_speech:
            segments.append(AudioSegment(
                audio=audio[speech_start:],
                sample_rate=self.sample_rate,
                start_time=speech_start / self.sample_rate,
                end_time=len(audio) / self.sample_rate,
                is_speech=True,
            ))

        return segments

    def compute_snr(self, clean: np.ndarray, noisy: np.ndarray) -> float:
        """Compute Signal-to-Noise Ratio."""
        noise = noisy - clean
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)

    def compute_pesq_proxy(self, clean: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Compute a PESQ proxy metric (simplified).
        PESQ (Perceptual Evaluation of Speech Quality) ranges from -0.5 to 4.5.
        """
        # Simple proxy: correlation-based score
        min_len = min(len(clean), len(enhanced))
        correlation = np.corrcoef(clean[:min_len], enhanced[:min_len])[0, 1]
        return float(correlation * 4.5)  # Scale to PESQ range

    def enhance_pipeline(self, audio: np.ndarray) -> Dict:
        """Full enhancement pipeline."""
        enhanced = self.spectral_gating(audio)
        speech_segments = self.voice_activity_detection(enhanced)
        snr = self.compute_snr(audio, enhanced)

        return {
            "enhanced_audio": enhanced,
            "speech_segments": len(speech_segments),
            "total_speech_duration": sum(s.end_time - s.start_time for s in speech_segments),
            "snr_db": snr,
            "sample_rate": self.sample_rate,
        }


if __name__ == "__main__":
    print("SpeechEnhancer: Spectral gating noise reduction + VAD")
    print("Methods: STFT, noise estimation, spectral subtraction, energy-based VAD")
    print("Metrics: SNR, PESQ proxy")
