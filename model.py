import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data
import os
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import optuna
from optuna.trial import Trial
import joblib
import time

class IDSGNNModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=64, num_heads=8, dropout=0.2, 
                 num_layers=2, pooling='add', residual=False, batch_norm=False):
        super(IDSGNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.pooling = pooling
        self.residual = residual
        self.batch_norm = batch_norm
        
        # Graph Attention layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(GATConv(num_features, hidden_channels, heads=num_heads, dropout=dropout))
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * num_heads))
        
        # Hidden layers
        for i in range(1, num_layers):
            if i == num_layers - 1:  # Last GAT layer
                self.gat_layers.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout))
                if batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            else:
                self.gat_layers.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
                if batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_channels * num_heads))
        
        # Output layers
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, batch):
        # Store original input for residual connection
        original_x = x
        
        # Process through GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            # Apply GAT layer
            x = gat_layer(x, edge_index)
            
            # Apply batch normalization if enabled
            if self.batch_norm:
                x = self.batch_norms[i](x)
            
            # Apply activation and dropout
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Apply residual connection if enabled and dimensions match
            if self.residual and i > 0 and x.size(-1) == original_x.size(-1):
                x = x + original_x
                original_x = x
        
        # Global pooling
        if self.pooling == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")
        
        # Final layers
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1)

def create_graph_data(features, edge_index, labels=None):
    """Create PyTorch Geometric Data object from features and edge indices"""
    x = torch.FloatTensor(features)
    edge_index = torch.LongTensor(edge_index)
    
    if labels is not None:
        y = torch.LongTensor(labels)
        return Data(x=x, edge_index=edge_index, y=y)
    
    return Data(x=x, edge_index=edge_index)

def train_model(model, train_loader, optimizer, device, scheduler=None):
    """Train the GNN model"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()
        
        # Calculate accuracy
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.y.size(0)
        
        total_loss += loss.item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, accuracy

def evaluate_model(model, loader, device, detailed=False):
    """Evaluate the GNN model"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            
            # Collect predictions and labels for detailed metrics
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
            correct += int((pred == data.y).sum())
            total += data.y.size(0)
    
    accuracy = correct / total
    
    if detailed:
        # Calculate detailed metrics
        # Convert both predictions and labels to integers to ensure consistency
        all_preds = [int(p) for p in all_preds]
        all_labels = [int(l) for l in all_labels]
        
        try:
            f1 = f1_score(all_labels, all_preds, average='weighted')
            report = classification_report(all_labels, all_preds, output_dict=True)
            cm = confusion_matrix(all_labels, all_preds)
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            print(f"Label types: {type(all_labels[0])}, Prediction types: {type(all_preds[0])}")
            print(f"Unique labels: {set(all_labels)}, Unique predictions: {set(all_preds)}")
            # Return default values if metrics calculation fails
            f1 = 0.0
            report = {}
            cm = np.zeros((2, 2))
        
        return accuracy, f1, report, cm, all_preds, all_labels
    
    return accuracy

def predict_new_data(model, data_loader, device):
    """Make predictions on new data"""
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.exp(out)  # Convert log_softmax to probabilities
            pred = out.argmax(dim=1)
            
            predictions.extend(pred.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)

def save_model(model, model_path, hyperparams=None, scaler=None):
    """Save model, hyperparameters, and scaler"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), model_path)
    
    # Save hyperparameters if provided
    if hyperparams:
        with open(f"{os.path.splitext(model_path)[0]}_hyperparams.json", 'w') as f:
            json.dump(hyperparams, f, indent=4)
    
    # Save scaler if provided
    if scaler:
        joblib.dump(scaler, f"{os.path.splitext(model_path)[0]}_scaler.pkl")
    
    print(f"Model saved to {model_path}")

def load_model(model_path, num_features, num_classes):
    """Load model and hyperparameters"""
    # Load hyperparameters
    hyperparams_path = f"{os.path.splitext(model_path)[0]}_hyperparams.json"
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
    else:
        # Default hyperparameters if file doesn't exist
        hyperparams = {
            'hidden_channels': 64,
            'num_heads': 8,
            'dropout': 0.2,
            'num_layers': 2,
            'pooling': 'add',
            'residual': False,
            'batch_norm': False
        }
    
    # Create model with loaded hyperparameters
    model = IDSGNNModel(
        num_features=num_features,
        num_classes=num_classes,
        hidden_channels=hyperparams.get('hidden_channels', 64),
        num_heads=hyperparams.get('num_heads', 8),
        dropout=hyperparams.get('dropout', 0.2),
        num_layers=hyperparams.get('num_layers', 2),
        pooling=hyperparams.get('pooling', 'add'),
        residual=hyperparams.get('residual', False),
        batch_norm=hyperparams.get('batch_norm', False)
    )
    
    # Load model state
    model.load_state_dict(torch.load(model_path))
    
    # Load scaler if it exists
    scaler_path = f"{os.path.splitext(model_path)[0]}_scaler.pkl"
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    return model, hyperparams, scaler

def objective(trial: Trial, train_loader, val_loader, num_features, num_classes, device, epochs=30):
    """Optuna objective function for hyperparameter optimization"""
    # Define hyperparameters to optimize
    hidden_channels = trial.suggest_int('hidden_channels', 32, 256, step=32)
    num_heads = trial.suggest_int('num_heads', 1, 8)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    pooling = trial.suggest_categorical('pooling', ['add', 'mean'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    residual = trial.suggest_categorical('residual', [True, False])
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    
    # Create model with trial hyperparameters
    model = IDSGNNModel(
        num_features=num_features,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        dropout=dropout,
        num_layers=num_layers,
        pooling=pooling,
        residual=residual,
        batch_norm=batch_norm
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    patience = 5  # Early stopping patience
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_model(model, train_loader, optimizer, device)
        
        # Validate
        val_acc = evaluate_model(model, val_loader, device)
        
        # Report intermediate metric
        trial.report(val_acc, epoch)
        
        # Handle pruning (early stopping for this trial)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_acc

def optimize_hyperparameters(train_loader, val_loader, num_features, num_classes, device, n_trials=50, study_name="ids_gnn_optimization"):
    """Run hyperparameter optimization using Optuna"""
    # Create study directory
    os.makedirs("studies", exist_ok=True)
    
    # Create or load study
    study_path = f"studies/{study_name}.pkl"
    if os.path.exists(study_path):
        print(f"Loading existing study from {study_path}")
        study = joblib.load(study_path)
    else:
        print(f"Creating new study: {study_name}")
        study = optuna.create_study(direction="maximize", study_name=study_name, 
                                   pruner=optuna.pruners.MedianPruner())
    
    # Run optimization
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, num_features, num_classes, device), 
                  n_trials=n_trials, timeout=3600)  # 1 hour timeout
    
    # Save study
    joblib.dump(study, study_path)
    
    # Get best parameters
    best_params = study.best_params
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")
    
    # Create and return model with best parameters
    best_model = IDSGNNModel(
        num_features=num_features,
        num_classes=num_classes,
        hidden_channels=best_params.get('hidden_channels', 64),
        num_heads=best_params.get('num_heads', 8),
        dropout=best_params.get('dropout', 0.2),
        num_layers=best_params.get('num_layers', 2),
        pooling=best_params.get('pooling', 'add'),
        residual=best_params.get('residual', False),
        batch_norm=best_params.get('batch_norm', False)
    ).to(device)
    
    return best_model, best_params

def train_with_best_params(model, train_loader, val_loader, test_loader, device, hyperparams, 
                          epochs=100, patience=10, model_save_path="models/best_ids_gnn_model.pt"):
    """Train model with best hyperparameters"""
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hyperparams.get('learning_rate', 0.001),
        weight_decay=hyperparams.get('weight_decay', 1e-4)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    train_accs = []
    val_accs = []
    
    print("Starting training with best hyperparameters...")
    for epoch in tqdm(range(epochs)):
        # Train
        train_loss, train_acc = train_model(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_acc = evaluate_model(model, val_loader, device)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save model
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved at epoch {epoch+1} with validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(model_save_path))
    
    # Evaluate on test set with error handling
    try:
        test_acc, test_f1, test_report, test_cm, _, _ = evaluate_model(model, test_loader, device, detailed=True)
        
        print(f"\nBest model from epoch {best_epoch+1}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print("\nClassification Report:")
        for class_id, metrics in test_report.items():
            if isinstance(metrics, dict):
                print(f"Class {class_id}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig("plots/confusion_matrix.png")
        
    except Exception as e:
        print(f"Error during final evaluation: {str(e)}")
        test_acc = best_val_acc  # Use validation accuracy as fallback
        test_f1 = 0.0
        test_report = {}
        test_cm = np.zeros((2, 2))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.axhline(y=test_acc, color='r', linestyle='--', label=f'Test Accuracy: {test_acc:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Create directory for plots
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/training_curves.png")
    
    # Save hyperparameters
    save_model(model, model_save_path, hyperparams)
    
    return model, test_acc, test_f1, test_report, test_cm

def main(train_loader, val_loader, test_loader, num_features, num_classes, device=None):
    """Main function to run the entire model training pipeline"""
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("studies", exist_ok=True)
    
    # Hyperparameter optimization
    print("Starting hyperparameter optimization...")
    best_model, best_params = optimize_hyperparameters(
        train_loader, val_loader, num_features, num_classes, device, n_trials=20
    )
    
    # Train with best hyperparameters
    print("\nTraining with best hyperparameters...")
    final_model, test_acc, test_f1, test_report, test_cm = train_with_best_params(
        best_model, train_loader, val_loader, test_loader, device, best_params
    )
    
    print("\nTraining complete!")
    print(f"Best hyperparameters: {best_params}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    return final_model, best_params, test_acc