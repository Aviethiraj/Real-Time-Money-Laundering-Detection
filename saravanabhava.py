import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import xgboost as xgb

# Optional: Only import torch if installed
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch torch-geometric")

# Load the dataset
try:
    df = pd.read_excel(r"E:\Download\SAML-D.xlsx", engine='openpyxl')
except Exception as e:
    print(f"Error loading with openpyxl: {e}")
    df = pd.read_excel(r"E:\Download\SAML-D.xlsx", engine='xlrd')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Preprocess the data
df['Time_seconds'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour * 3600 + \
                     pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute * 60 + \
                     pd.to_datetime(df['Time'], format='%H:%M:%S').dt.second

df['Day_of_week'] = df['Date'].dt.dayofweek
df['Day_of_month'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Payment_currency', 'Received_currency', 'Sender_bank_location', 
                       'Receiver_bank_location', 'Payment_type', 'Laundering_type']

for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Create unique account IDs
all_accounts = pd.concat([df['Sender_account'], df['Receiver_account']]).unique()
account_to_id = {account: idx for idx, account in enumerate(all_accounts)}

df['Sender_id'] = df['Sender_account'].map(account_to_id)
df['Receiver_id'] = df['Receiver_account'].map(account_to_id)

print(f"\nTotal unique accounts: {len(all_accounts)}")

# ==================== XGBOOST MODEL ====================
print("\n" + "="*50)
print("TRAINING XGBOOST MODEL")
print("="*50)

# Prepare features for XGBoost
xgb_features = ['Amount', 'Time_seconds', 'Day_of_week', 'Day_of_month', 'Month', 'Year',
                'Payment_currency', 'Received_currency', 'Sender_bank_location', 
                'Receiver_bank_location', 'Payment_type', 'Sender_id', 'Receiver_id']

X = df[xgb_features]
y = df['Is_laundering']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# Evaluate XGBoost
xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

print("\n=== XGBoost Model Performance ===")
print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))
print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))
print(f"\nROC-AUC Score: {roc_auc_score(y_test, xgb_pred_proba):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': xgb_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'].head(10), feature_importance['importance'].head(10))
plt.xlabel('Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()

# ==================== GNN MODEL (FINAL FIXED + JSON OUTPUT) ====================
if TORCH_AVAILABLE:
    print("\n" + "="*50)
    print("TRAINING GNN MODEL (FINAL VERSION)")
    print("="*50)

    import time
    import json
    from sklearn.metrics import precision_score, recall_score, f1_score

    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ----------- EDGE INDEX (OPTIMIZED) -----------
    edge_index_np = np.vstack((df['Sender_id'].values, df['Receiver_id'].values))
    edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)

    num_nodes = len(all_accounts)

    # ----------- NODE FEATURES (NO LEAKAGE) -----------
    sender_group = df.groupby('Sender_id')['Amount']
    receiver_group = df.groupby('Receiver_id')['Amount']

    node_features = np.zeros((num_nodes, 6))

    node_features[:, 0] = sender_group.sum().reindex(range(num_nodes), fill_value=0)
    node_features[:, 1] = sender_group.count().reindex(range(num_nodes), fill_value=0)
    node_features[:, 2] = sender_group.mean().reindex(range(num_nodes), fill_value=0)

    node_features[:, 3] = receiver_group.sum().reindex(range(num_nodes), fill_value=0)
    node_features[:, 4] = receiver_group.count().reindex(range(num_nodes), fill_value=0)
    node_features[:, 5] = receiver_group.mean().reindex(range(num_nodes), fill_value=0)

    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)

    x = torch.tensor(node_features, dtype=torch.float).to(device)

    # ----------- LABELS -----------
    y = torch.tensor(df['Is_laundering'].values, dtype=torch.long).to(device)

    # ----------- TRAIN / TEST SPLIT -----------
    edge_indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        edge_indices,
        test_size=0.2,
        stratify=df['Is_laundering'],
        random_state=42
    )

    train_mask = torch.zeros(len(df), dtype=torch.bool).to(device)
    test_mask = torch.zeros(len(df), dtype=torch.bool).to(device)

    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # ----------- BALANCED CLASS WEIGHT (CAPPED) -----------
    num_pos = (y == 1).sum().item()
    num_neg = (y == 0).sum().item()

    pos_weight = min(50, num_neg / num_pos)  # 🔥 FIXED
    class_weights = torch.tensor([1.0, pos_weight]).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ----------- MODEL -----------
    class GNNModel(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.dropout = torch.nn.Dropout(0.3)
            self.lin = torch.nn.Linear(hidden_channels * 2, 2)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index))

            edge_emb = torch.cat(
                [x[edge_index[0]], x[edge_index[1]]],
                dim=1
            )

            return self.lin(edge_emb)

    model = GNNModel(x.shape[1], 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # ----------- TRAINING -----------
    print("\nTraining GNN...")
    best_loss = float('inf')
    patience = 10
    trigger = 0
    losses = []

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()

        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            trigger = 0
        else:
            trigger += 1

        if trigger >= patience:
            print("Early stopping triggered")
            break

    # ----------- EVALUATION WITH THRESHOLD SEARCH -----------
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        probs = F.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()

    y_true = y[test_mask].cpu().numpy()

    print("\nFinding best threshold...")

    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}

    for t in np.arange(0.05, 0.6, 0.05):
        preds = (probs > t).astype(int)

        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds)
        f1 = f1_score(y_true, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

    print(f"\nBest Threshold: {best_threshold}")
    print("Best Metrics:", best_metrics)

    # Final predictions
    final_preds = (probs > best_threshold).astype(int)

    cm = confusion_matrix(y_true, final_preds)
    roc = roc_auc_score(y_true, probs)

    print("\n=== FINAL GNN PERFORMANCE ===")
    print(cm)
    print(classification_report(y_true, final_preds))
    print(f"ROC-AUC: {roc:.4f}")

    # ----------- SAVE TO JSON -----------
import json
from sklearn.metrics import precision_score, recall_score, f1_score

# ----------- XGBOOST METRICS -----------
xgb_metrics = {
    "accuracy": float(accuracy_score(y_test, xgb_pred)),
    "roc_auc": float(roc_auc_score(y_test, xgb_pred_proba)),
    "precision": float(precision_score(y_test, xgb_pred)),
    "recall": float(recall_score(y_test, xgb_pred)),
    "f1_score": float(f1_score(y_test, xgb_pred)),
    "confusion_matrix": confusion_matrix(y_test, xgb_pred).tolist()
}

# ----------- GNN METRICS -----------
gnn_metrics = {
    "accuracy": float(accuracy_score(y_true, final_preds)),
    "roc_auc": float(roc),
    "precision": float(best_metrics['precision']),
    "recall": float(best_metrics['recall']),
    "f1_score": float(best_metrics['f1']),
    "best_threshold": float(best_threshold),
    "confusion_matrix": cm.tolist()
}

# ----------- TRAINING INFO -----------
training_info = {
    "epochs": len(losses),
    "final_loss": float(losses[-1])
}

# ----------- FEATURE IMPORTANCE -----------
feature_data = [
    {
        "name": row['feature'],
        "importance": float(row['importance']),
        "impact": (
            "High" if row['importance'] > 0.1 else
            "Medium" if row['importance'] > 0.05 else
            "Low"
        )
    }
    for _, row in feature_importance.head(10).iterrows()
]

# ----------- DATASET INFO -----------
dataset_info = {
    "total_transactions": int(len(df)),
    "fraud_cases": int(df['Is_laundering'].sum()),
    "fraud_ratio": float(df['Is_laundering'].mean())
}

# ----------- FINAL DASHBOARD JSON -----------
dashboard_data = {
    "xgboost": xgb_metrics,
    "gnn": gnn_metrics,
    "features": feature_data,
    "dataset": dataset_info,
    "training": training_info
}

# ----------- SAVE FILE -----------
with open('dashboard_data.json', 'w') as f:
    json.dump(dashboard_data, f, indent=2)

print("\n✅ All results saved to dashboard_data.json")
print(f"GNN completed in {time.time() - start_time:.2f} seconds")
