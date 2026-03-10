import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


class TabEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(4 * input_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(1)
        return self.head(x).squeeze(-1)


# -- Config --
SEED       = 42
EPOCHS     = 500
LR         = 1e-3
BATCH_SIZE = 16
DATA_PATH  = "/Users/manjilnepal/Downloads/Capstone-Final/new_ds.csv"
SAVE_DIR   = "./tab_results"

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- Load & prepare data --
df = pd.read_csv(DATA_PATH).drop(columns=['CS 7 day (Mpa)'])
feature_cols = df.columns.tolist()[:-1]

X = df[feature_cols].values.astype(np.float32)
y = df['CS 28 day (Mpa)'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

print(f'Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feature_cols)}')

X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).to(device)
X_test_t  = torch.tensor(X_test).to(device)
y_test_t  = torch.tensor(y_test).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

# -- Model, loss, optimizer --
model     = TabEncoder(input_dim=len(feature_cols)).to(device)
loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

history = {'train_mae': [], 'train_rmse': [], 'test_mae': [], 'test_rmse': []}

print(f'\nTraining for {EPOCHS} epochs...\n')
for epoch in range(1, EPOCHS + 1):

    # -- Train --
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss_fn(model(X_batch), y_batch).backward()
        optimizer.step()

    # -- Evaluate --
    model.eval()
    with torch.no_grad():
        tr_preds = model(X_train_t).cpu().numpy()
        te_preds = model(X_test_t).cpu().numpy()

    tr_mae  = mean_absolute_error(y_train, tr_preds)
    tr_rmse = np.sqrt(mean_squared_error(y_train, tr_preds))
    te_mae  = mean_absolute_error(y_test, te_preds)
    te_rmse = np.sqrt(mean_squared_error(y_test, te_preds))
    te_r2   = r2_score(y_test, te_preds)

    history['train_mae'].append(tr_mae)
    history['train_rmse'].append(tr_rmse)
    history['test_mae'].append(te_mae)
    history['test_rmse'].append(te_rmse)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:>3}/{EPOCHS}] "
              f"Train -> MAE: {tr_mae:.4f} | RMSE: {tr_rmse:.4f} | "
              f"Test  -> MAE: {te_mae:.4f} | RMSE: {te_rmse:.4f} | R²: {te_r2:.4f}")

print(f'\n{"═"*60}\nFINAL TEST RESULTS\n{"═"*60}')
print(f'MAE: {te_mae:.4f} | RMSE: {te_rmse:.4f} | R²: {te_r2:.4f}')
print(f'{"═"*60}\n')

# -- Plot --
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
for ax, key, title in zip(axes, ['mae', 'rmse'], ['MAE', 'RMSE']):
    ax.plot(history[f'train_{key}'], label='Train', linewidth=2)
    ax.plot(history[f'test_{key}'],  label='Test',  linewidth=2)
    ax.set(title=title, xlabel='Epoch', ylabel=title)
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'conv_plot.png'), dpi=300)
plt.close()