import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib

matplotlib.use('Agg')  # For Replit compatibility
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


def train_traditional_models():
    """Train traditional ML models"""
    print("Loading processed data...")

    # Load features and labels
    df_features = pd.read_csv('data/processed/features.csv')
    with open('data/processed/labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    print(f"Features shape: {df_features.shape}")
    print(f"Labels count: {len(labels)}")

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Handle missing values
    df_features = df_features.fillna(df_features.mean())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df_features,
        encoded_labels,
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models (smaller for Replit)
    models = {
        'Random Forest':
        RandomForestClassifier(
            n_estimators=50,  # Reduced for Replit
            max_depth=8,
            random_state=42,
            n_jobs=1  # Single core for Replit
        ),
        'SVM':
        SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    }

    results = {}
    os.makedirs('models/traditional', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train model
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test,
                                       y_pred,
                                       target_names=label_encoder.classes_)
        cm = confusion_matrix(y_test, y_pred)

        # Save model
        joblib.dump(
            model,
            f'models/traditional/{name.lower().replace(" ", "_")}_model.pkl')

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title(f'{name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(
            f'results/plots/{name.lower().replace(" ", "_")}_confusion_matrix.png',
            dpi=100)
        plt.close()

        results[name] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm
        }

        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"Classification Report:\n{report}")

    # Save results
    with open('results/traditional_ml_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Save preprocessing objects
    joblib.dump(label_encoder, 'models/traditional/label_encoder.pkl')
    joblib.dump(scaler, 'models/traditional/scaler.pkl')

    return results


if __name__ == "__main__":
    results = train_traditional_models()
