"""
Data processing utilities for video and audio preprocessing.
"""
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


class VideoPreprocessor:
    """Preprocess video frames for model input."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True):
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target size using bilinear interpolation."""
        h, w = frame.shape[:2]
        target_h, target_w = self.target_size
        
        # Simple bilinear resize
        y = np.linspace(0, h - 1, target_h)
        x = np.linspace(0, w - 1, target_w)
        
        if frame.ndim == 3:
            resized = np.zeros((target_h, target_w, frame.shape[2]))
            for c in range(frame.shape[2]):
                resized[:,:,c] = _bilinear_interp(frame[:,:,c], x, y)
        else:
            resized = _bilinear_interp(frame, x, y)
        
        return resized.astype(np.float32)
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame with ImageNet stats."""
        frame = frame / 255.0
        if self.normalize and frame.ndim == 3:
            frame = (frame - self.mean) / self.std
        return frame.astype(np.float32)
    
    def create_patches(self, frame: np.ndarray, patch_size: int = 16) -> np.ndarray:
        """Create patches for Vision Transformer input."""
        h, w = frame.shape[:2]
        n_h = h // patch_size
        n_w = w // patch_size
        
        patches = []
        for i in range(n_h):
            for j in range(n_w):
                patch = frame[i*patch_size:(i+1)*patch_size,
                             j*patch_size:(j+1)*patch_size]
                patches.append(patch.flatten())
        
        return np.array(patches)
    
    def process_video(self, frames: List[np.ndarray]) -> Dict:
        """Process entire video."""
        processed_frames = []
        patches_list = []
        
        for frame in frames:
            resized = self.resize_frame(frame)
            normalized = self.normalize_frame(resized)
            patches = self.create_patches(resized, patch_size=16)
            
            processed_frames.append(normalized)
            patches_list.append(patches)
        
        return {
            "frames": np.array(processed_frames),
            "patches": np.array(patches_list),
            "n_frames": len(frames),
            "shape": processed_frames[0].shape if processed_frames else None,
        }


class AudioPreprocessor:
    """Preprocess audio for model input."""
    
    def __init__(self, sample_rate: int = 16000, n_mels: int = 80,
                 n_fft: int = 1024, hop_length: int = 256):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel-scale spectrogram."""
        # STFT
        stft = self._stft(audio)
        magnitude = np.abs(stft)
        
        # Mel filterbank
        mel_filter = self._mel_filterbank()
        mel_spec = mel_filter @ magnitude
        
        # Log scale
        mel_spec = 10 * np.log10(mel_spec + 1e-10)
        
        return mel_spec
    
    def _stft(self, audio: np.ndarray) -> np.ndarray:
        """Compute STFT."""
        frames = []
        for i in range(0, len(audio) - self.n_fft, self.hop_length):
            frame = audio[i:i+self.n_fft] * np.hanning(self.n_fft)
            spectrum = np.fft.rfft(frame)
            frames.append(spectrum)
        return np.array(frames).T
    
    def _mel_filterbank(self) -> np.ndarray:
        """Create mel filterbank."""
        n_freqs = self.n_fft // 2 + 1
        mel_min = 0
        mel_max = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
        
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        bin_points = np.floor(hz_points / self.sample_rate * self.n_fft).astype(int)
        
        filterbank = np.zeros((self.n_mels, n_freqs))
        for m in range(self.n_mels):
            for k in range(bin_points[m], bin_points[m+1]):
                filterbank[m, k] = (k - bin_points[m]) / (bin_points[m+1] - bin_points[m] + 1e-8)
            for k in range(bin_points[m+1], bin_points[m+2]):
                filterbank[m, k] = (bin_points[m+2] - k) / (bin_points[m+2] - bin_points[m+1] + 1e-8)
        
        return filterbank
    
    def compute_mfccs(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Compute MFCCs from mel spectrogram."""
        mel_spec = self.compute_mel_spectrogram(audio)
        
        # DCT (Type II)
        from scipy.fft import dct
        mfccs = dct(mel_spec, type=2, axis=0, norm='ortho')[:n_mfcc]
        
        return mfccs
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def remove_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Remove silence from audio."""
        frames = []
        frame_length = int(0.025 * self.sample_rate)
        
        for i in range(0, len(audio) - frame_length, frame_length):
            frame = audio[i:i+frame_length]
            energy = np.mean(frame ** 2)
            if energy > threshold:
                frames.extend(frame)
        
        return np.array(frames) if frames else audio


def _bilinear_interp(image: np.ndarray, x_coords: np.ndarray, 
                     y_coords: np.ndarray) -> np.ndarray:
    """Bilinear interpolation."""
    h, w = image.shape[:2]
    
    x0 = np.floor(x_coords).astype(int)
    x1 = (x0 + 1) % w
    y0 = np.floor(y_coords).astype(int)
    y1 = (y0 + 1) % h
    
    x_frac = x_coords - x0
    y_frac = y_coords - y0
    
    x0 = np.clip(x0, 0, w-1)
    x1 = np.clip(x1, 0, w-1)
    y0 = np.clip(y0, 0, h-1)
    y1 = np.clip(y1, 0, h-1)
    
    result = np.zeros((len(y_coords), len(x_coords)))
    for i, (y0i, y1i, yfi) in enumerate(zip(y0, y1, y_frac)):
        for j, (x0j, x1j, xfj) in enumerate(zip(x0, x1, x_frac)):
            result[i, j] = (image[y0i, x0j] * (1 - yfi) * (1 - xfj) +
                           image[y0i, x1j] * (1 - yfi) * xfj +
                           image[y1i, x0j] * yfi * (1 - xfj) +
                           image[y1i, x1j] * yfi * xfj)
    
    return result


if __name__ == "__main__":
    print("Data Preprocessing Utilities")
    print("=" * 50)
    
    # Video preprocessing
    vp = VideoPreprocessor(target_size=(224, 224))
    frames = [np.random.randint(0, 255, (480, 640, 3)) for _ in range(5)]
    result = vp.process_video(frames)
    print(f"\nVideo preprocessing:")
    print(f"  Input: 5 frames of (480, 640, 3)")
    print(f"  Output frames: {result['frames'].shape}")
    print(f"  Output patches: {result['patches'].shape}")
    
    # Audio preprocessing
    ap = AudioPreprocessor(sample_rate=16000)
    audio = np.random.randn(16000) * 0.1
    mel_spec = ap.compute_mel_spectrogram(audio)
    mfccs = ap.compute_mfccs(audio)
    print(f"\nAudio preprocessing:")
    print(f"  Mel spectrogram: {mel_spec.shape}")
    print(f"  MFCCs: {mfccs.shape}")
