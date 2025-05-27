import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import os
import pickle
from tqdm import tqdm


class MITBIHLoader:

    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir

    def load_data(self):
        """Returns empty data - using synthetic data generation instead"""
        print("Using synthetic data generation...")
        return [], []


class FeatureExtractor:

    def __init__(self, fs=360):
        self.fs = fs

    def time_domain_features(self, heartbeat):
        """Extract time-domain features"""
        from scipy.stats import skew, kurtosis

        features = {
            'mean': np.mean(heartbeat),
            'std': np.std(heartbeat),
            'max': np.max(heartbeat),
            'min': np.min(heartbeat),
            'range': np.max(heartbeat) - np.min(heartbeat),
            'skewness': skew(heartbeat),
            'kurtosis': kurtosis(heartbeat),
            'rms': np.sqrt(np.mean(heartbeat**2))
        }
        return features

    def frequency_domain_features(self, heartbeat):
        """Extract frequency-domain features"""
        from scipy.fft import fft

        # FFT
        fft_vals = np.abs(fft(heartbeat))
        freqs = np.fft.fftfreq(len(heartbeat), 1 / self.fs)

        # Power spectral density features
        total_power = np.sum(fft_vals**2)

        features = {
            'spectral_centroid':
            np.sum(freqs[:len(freqs) // 2] * fft_vals[:len(freqs) // 2]) /
            np.sum(fft_vals[:len(freqs) // 2])
            if np.sum(fft_vals[:len(freqs) // 2]) > 0 else 0,
            'spectral_energy':
            total_power
        }
        return features

    def morphological_features(self, heartbeat):
        """Extract morphological features"""
        # R-peak is at center
        r_peak_idx = len(heartbeat) // 2

        features = {
            'r_amplitude': heartbeat[r_peak_idx],
            'q_amplitude': np.min(heartbeat[:r_peak_idx]),
            's_amplitude': np.min(heartbeat[r_peak_idx:]),
        }
        return features

    def extract_all_features(self, heartbeat):
        """Extract all features for a heartbeat"""
        features = {}
        features.update(self.time_domain_features(heartbeat))
        features.update(self.frequency_domain_features(heartbeat))
        features.update(self.morphological_features(heartbeat))
        return features
