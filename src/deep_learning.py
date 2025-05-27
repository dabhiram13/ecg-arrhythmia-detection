import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib

matplotlib.use('Agg')  # For Replit compatibility
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class ECGDataset(Dataset):

    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CNN1D(nn.Module):

    def __init__(self,
                 input_size=180,
                 num_classes=5):  # Adjusted for smaller input
        super(CNN1D, self).__init__()

        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(1, 16, kernel_size=5,
                      padding=2),  # Reduced channels for Replit
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # Second conv block
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # Third conv block
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2))

        # Calculate flattened size
        self.flattened_size = self._get_flattened_size(input_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),  # Reduced size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes))

    def _get_flattened_size(self, input_size):
        with torch.no_grad():
            x = torch.randn(1, 1, input_size)
            x = self.conv_layers(x)
            return x.numel()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class LSTM(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 num_layers=2,
                 num_classes=5):  # Reduced size
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            dropout=0.2)

        self.classifier = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(32, num_classes))

    def forward(self, x):
        x = x.unsqueeze(2)  # Add feature dimension

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Take the last output
        out = self.classifier(out[:, -1, :])
        return out


def train_deep_model(model,
                     train_loader,
                     val_loader,
                     num_epochs=20):  # Reduced epochs
    """Train deep learning model"""
    device = torch.device('cpu')  # Force CPU for Replit
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_accuracies = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(
                device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(
                    device), batch_labels.to(device)
                outputs = model(batch_data)

                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        train_loss /= len(train_loader)
        val_accuracy = 100 * val_correct / val_total

        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

        if epoch % 5 == 0:
            print(
                f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%'
            )

    return train_losses, val_accuracies


def prepare_dl_data():
    """Prepare data for deep learning"""
    print("Loading data for deep learning...")

    # Load data
    heartbeats = np.load('data/processed/heartbeats.npy')
    with open('data/processed/labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        heartbeats,
        encoded_labels,
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels)

    # Further split training into train/validation
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      stratify=y_train)

    # Create datasets
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)

    # Create data loaders (smaller batch size for Replit)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader, label_encoder


def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    device = torch.device('cpu')
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(
                device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def train_deep_models():
    """Train deep learning models"""
    print("Preparing data for deep learning...")
    train_loader, val_loader, test_loader, label_encoder = prepare_dl_data()

    # Model parameters
    input_size = next(iter(train_loader))[0].shape[1]  # Get input size
    num_classes = len(label_encoder.classes_)

    print(f"Input size: {input_size}")
    print(f"Number of classes: {num_classes}")

    models = {
        'CNN': CNN1D(input_size=input_size, num_classes=num_classes),
        'LSTM': LSTM(num_classes=num_classes)
    }

    results = {}
    os.makedirs('models/deep_learning', exist_ok=True)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train model
        train_losses, val_accuracies = train_deep_model(model,
                                                        train_loader,
                                                        val_loader,
                                                        num_epochs=20)

        # Plot training curves
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title(f'{name} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies)
        plt.title(f'{name} - Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'results/plots/{name.lower()}_training_curves.png',
                    dpi=100)
        plt.close()

        # Evaluate on test set
        test_accuracy = evaluate_model(model, test_loader)

        # Save model
        torch.save(model.state_dict(),
                   f'models/deep_learning/{name.lower()}_model.pth')

        results[name] = {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'test_accuracy': test_accuracy,
            'best_val_accuracy': max(val_accuracies)
        }

        print(f"{name} - Best Validation Accuracy: {max(val_accuracies):.2f}%")
        print(f"{name} - Test Accuracy: {test_accuracy:.2f}%")

    # Save results
    with open('results/deep_learning_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":
    results = train_deep_models()
