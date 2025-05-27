import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
import os
import pickle
from tqdm import tqdm


class MITBIHLoader:

    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        # Start with fewer records for Replit (memory constraints)
        self.record_list = list(range(100, 110)) + list(range(
            200, 210))  # 20 records

    def download_data(self):
        """Download MIT-BIH database"""
        os.makedirs(self.data_dir, exist_ok=True)

        records = []
        annotations = []

        print("Downloading MIT-BIH Arrhythmia Database...")

        # First, download the entire database
        try:
            print("Downloading database files...")
            wfdb.dl_database('mitdb', dl_dir=self.data_dir)
        except Exception as e:
            print(f"Database download error: {e}")

        # Now read individual records
        for i, record_num in enumerate(
                tqdm(self.record_list, desc="Loading records")):
            try:
                # Read from downloaded files
                record_path = f'{self.data_dir}/mitdb/{record_num}'
                record = wfdb.rdrecord(record_path)
                annotation = wfdb.rdann(record_path, 'atr')

                records.append(record)
                annotations.append(annotation)

            except Exception as e:
                print(f"Failed to load record {record_num}: {e}")
                # Try alternative method
                try:
                    record = wfdb.rdrecord(f'mitdb/{record_num}')
                    annotation = wfdb.rdann(f'mitdb/{record_num}', 'atr')
                    records.append(record)
                    annotations.append(annotation)
                    print(
                        f"Successfully loaded record {record_num} using alternative method"
                    )
                except Exception as e2:
                    print(
                        f"Alternative method also failed for record {record_num}: {e2}"
                    )
                    continue

        # Save data
        with open(f'{self.data_dir}/records.pkl', 'wb') as f:
            pickle.dump(records, f)
        with open(f'{self.data_dir}/annotations.pkl', 'wb') as f:
            pickle.dump(annotations, f)

        print(f"Downloaded {len(records)} records successfully!")
        return records, annotations

    def load_data(self):
        """Load downloaded data"""
        try:
            with open(f'{self.data_dir}/records.pkl', 'rb') as f:
                records = pickle.load(f)
            with open(f'{self.data_dir}/annotations.pkl', 'rb') as f:
                annotations = pickle.load(f)
            print(f"Loaded {len(records)} records from cache")
            return records, annotations
        except FileNotFoundError:
            print("Data not found. Downloading...")
            return self.download_data()


class ECGPreprocessor:

    def __init__(self, fs=360):
        self.fs = fs

    def bandpass_filter(self, ecg_signal, lowcut=0.5, highcut=50):
        """Apply bandpass filter to remove baseline wander and high-freq noise"""
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)
        return filtered_signal

    def notch_filter(self, ecg_signal, notch_freq=60):
        """Remove power line interference"""
        nyquist = 0.5 * self.fs
        notch = notch_freq / nyquist

        b, a = butter(4, [notch - 0.01, notch + 0.01], btype='bandstop')
        filtered_signal = filtfilt(b, a, ecg_signal)
        return filtered_signal

    def normalize_signal(self, ecg_signal):
        """Z-score normalization"""
        return (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

    def preprocess(self, ecg_signal):
        """Complete preprocessing pipeline"""
        # Apply filters
        signal_filtered = self.bandpass_filter(ecg_signal)
        signal_filtered = self.notch_filter(signal_filtered)

        # Normalize
        signal_normalized = self.normalize_signal(signal_filtered)

        return signal_normalized


class ECGSegmentation:

    def __init__(self, fs=360):
        self.fs = fs

    def detect_r_peaks(self, ecg_signal, height_threshold=0.6):
        """Detect R-peaks in ECG signal"""
        # Find peaks with minimum distance and height
        peaks, _ = find_peaks(ecg_signal,
                              height=height_threshold * np.max(ecg_signal),
                              distance=int(0.6 *
                                           self.fs))  # Min 0.6s between peaks
        return peaks

    def extract_heartbeats(self,
                           ecg_signal,
                           r_peaks,
                           window_size=180):  # Smaller window for Replit
        """Extract individual heartbeat segments"""
        heartbeats = []
        half_window = window_size // 2

        for peak in r_peaks:
            if peak >= half_window and peak < len(ecg_signal) - half_window:
                heartbeat = ecg_signal[peak - half_window:peak + half_window]
                heartbeats.append(heartbeat)

        return np.array(heartbeats)


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


class LabelEncoder:

    def __init__(self):
        # AAMI standard classification
        self.aami_classes = {
            'N': ['N', 'L', 'R', 'e', 'j'],  # Normal
            'S': ['A', 'a', 'J', 'S'],  # Supraventricular
            'V': ['V', 'E'],  # Ventricular
            'F': ['F'],  # Fusion
            'Q': ['/', 'f', 'Q']  # Unknown/Unclassifiable
        }

    def encode_annotation(self, symbol):
        """Convert MIT-BIH annotation to AAMI class"""
        for aami_class, symbols in self.aami_classes.items():
            if symbol in symbols:
                return aami_class
        return 'Q'  # Unknown

    def create_labels(self, annotations, r_peaks):
        """Create labels for detected heartbeats"""
        labels = []

        for peak in r_peaks:
            # Find closest annotation
            closest_idx = np.argmin(np.abs(annotations.sample - peak))
            symbol = annotations.symbol[closest_idx]
            label = self.encode_annotation(symbol)
            labels.append(label)

        return labels
