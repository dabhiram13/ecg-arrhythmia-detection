import os
import sys
import time
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.append('src')


def main():
    """Main execution pipeline optimized for Replit"""
    start_time = time.time()

    print("ECG Arrhythmia Detection System - Replit Edition")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")

    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/traditional', exist_ok=True)
    os.makedirs('models/deep_learning', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    try:
        # Clear any corrupted cache first
        print("\nClearing any corrupted cache...")
        if os.path.exists('data/processed/features.csv'):
            try:
                test_df = pd.read_csv('data/processed/features.csv')
                if len(test_df) == 0 or len(test_df.columns) == 0:
                    print("Removing corrupted files...")
                    os.remove('data/processed/features.csv')
                    if os.path.exists('data/processed/heartbeats.npy'):
                        os.remove('data/processed/heartbeats.npy')
                    if os.path.exists('data/processed/labels.pkl'):
                        os.remove('data/processed/labels.pkl')
            except:
                print("Removing corrupted files...")
                if os.path.exists('data/processed/features.csv'):
                    os.remove('data/processed/features.csv')
                if os.path.exists('data/processed/heartbeats.npy'):
                    os.remove('data/processed/heartbeats.npy')
                if os.path.exists('data/processed/labels.pkl'):
                    os.remove('data/processed/labels.pkl')

        # Step 1: Feature Extraction (synthetic data generation)
        print("\nStep 1: Feature Extraction...")
        from feature_extraction import extract_all_features
        extract_all_features()

        # Step 2: Traditional ML
        print("\nStep 2: Training Traditional ML Models...")
        from traditional_ml import train_traditional_models
        traditional_results = train_traditional_models()

        # Step 3: Deep Learning
        print("\nStep 3: Training Deep Learning Models...")
        from deep_learning import train_deep_models
        dl_results = train_deep_models()

        # Step 4: Final Report
        print("\n" + "=" * 50)
        print("FINAL RESULTS COMPARISON")
        print("=" * 50)

        print("\nTraditional ML Results:")
        for model, results in traditional_results.items():
            print(f"{model}: {results['accuracy']:.4f}")

        print("\nDeep Learning Results:")
        for model, results in dl_results.items():
            print(f"{model}: {results['test_accuracy']:.2f}%")

        # Step 5: Display Results
        print("\n" + "=" * 50)
        print("PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nGenerated Files:")
        print("📊 results/plots/ - Confusion matrices and training curves")
        print("🤖 models/ - Trained models")
        print("📈 results/ - Performance metrics")

        end_time = time.time()
        duration = (end_time - start_time) / 60

        print(f"\n⏱️ Total execution time: {duration:.1f} minutes")
        print("\n🎉 Check the results folder for visualizations!")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
