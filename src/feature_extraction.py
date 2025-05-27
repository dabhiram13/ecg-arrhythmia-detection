import numpy as np
import pandas as pd
from data_preprocessing import FeatureExtractor
import pickle
import os
from tqdm import tqdm


def extract_all_features():
    """Generate synthetic ECG features"""
    print("Generating synthetic ECG data...")
    return create_sample_data()


def create_sample_data():
    """Create synthetic ECG data for demonstration when real data fails to download"""
    print("Creating synthetic ECG data for demonstration...")

    np.random.seed(42)  # For reproducible results

    # Generate synthetic ECG heartbeats
    n_samples = 1000
    heartbeat_length = 180

    all_heartbeats = []
    all_labels = []
    all_features = []

    # Generate different types of heartbeats
    label_types = ['N', 'V', 'S', 'F', 'Q']
    label_weights = [0.7, 0.15, 0.08, 0.04,
                     0.03]  # Normal beats are most common

    feature_extractor = FeatureExtractor()

    for i in tqdm(range(n_samples), desc="Generating synthetic data"):
        # Choose label
        label = np.random.choice(label_types, p=label_weights)

        # Generate synthetic heartbeat based on label
        t = np.linspace(0, 1, heartbeat_length)

        if label == 'N':  # Normal heartbeat
            heartbeat = np.sin(2 * np.pi * t) * np.exp(-((t - 0.5)**2) / 0.1)
            heartbeat += 0.3 * np.sin(6 * np.pi * t) * np.exp(-(
                (t - 0.5)**2) / 0.05)
        elif label == 'V':  # Ventricular (wider QRS)
            heartbeat = 0.8 * np.sin(2 * np.pi * t) * np.exp(-(
                (t - 0.5)**2) / 0.15)
            heartbeat += 0.4 * np.sin(4 * np.pi * t) * np.exp(-(
                (t - 0.5)**2) / 0.08)
        elif label == 'S':  # Supraventricular
            heartbeat = 1.2 * np.sin(2 * np.pi * t) * np.exp(-(
                (t - 0.4)**2) / 0.08)
            heartbeat += 0.2 * np.sin(8 * np.pi * t) * np.exp(-(
                (t - 0.4)**2) / 0.04)
        elif label == 'F':  # Fusion
            heartbeat = 0.6 * np.sin(2 * np.pi * t) * np.exp(-(
                (t - 0.5)**2) / 0.12)
            heartbeat += 0.6 * np.sin(3 * np.pi * t) * np.exp(-(
                (t - 0.6)**2) / 0.1)
        else:  # Q - Unknown/Artifact
            heartbeat = 0.3 * np.random.normal(0, 0.1, heartbeat_length)
            heartbeat += 0.4 * np.sin(2 * np.pi * t) * np.exp(-(
                (t - 0.5)**2) / 0.2)

        # Add noise
        noise = np.random.normal(0, 0.05, heartbeat_length)
        heartbeat += noise

        # Extract features
        features = feature_extractor.extract_all_features(heartbeat)

        all_heartbeats.append(heartbeat)
        all_labels.append(label)
        all_features.append(features)

    # Save processed data
    os.makedirs('data/processed', exist_ok=True)

    np.save('data/processed/heartbeats.npy', np.array(all_heartbeats))
    with open('data/processed/labels.pkl', 'wb') as f:
        pickle.dump(all_labels, f)

    # Save features as DataFrame
    df_features = pd.DataFrame(all_features)
    df_features.to_csv('data/processed/features.csv', index=False)

    print(f"Generated {len(all_heartbeats)} synthetic heartbeats")
    print(f"Label distribution: {pd.Series(all_labels).value_counts()}")
    print("✅ Synthetic data created successfully!")

    return all_heartbeats, all_labels, all_features


if __name__ == "__main__":
    extract_all_features()
