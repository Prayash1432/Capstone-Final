import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import os
import warnings
warnings.filterwarnings('ignore')


class TabEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -- Config --
SEED = 42
EPOCHS = 500
LR = 1e-3
BATCH_SIZE = 8
DATA_PATH = "../data/trial.engineered_ds.csv"
SAVE_DIR = "./results"

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -- Load data --
df = pd.read_csv(DATA_PATH).drop(columns=['CS 7 day (Mpa)'])

feature_cols = df.columns.tolist()[:-6]

X = df[feature_cols].values.astype(np.float32)
y = df['CS 28 day (Mpa)'].values.astype(np.float32)


# -- Train / Validation / Test split --
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED
)


# -- Scaling --
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

print(f'Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} | Features: {len(feature_cols)}')


# -- Convert to tensors --
X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).to(device)

X_val_t = torch.tensor(X_val).to(device)
y_val_t = torch.tensor(y_val).to(device)

X_test_t = torch.tensor(X_test).to(device)
y_test_t = torch.tensor(y_test).to(device)


train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True
)


# -- Model --
model = TabEncoder(input_dim=len(feature_cols)).to(device)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# -- Metrics history --
history = {
    'train_mae': [], 'train_rmse': [],
    'val_mae': [], 'val_rmse': [],
    'test_mae': [], 'test_rmse': []
}


print(f'\nTraining for {EPOCHS} epochs...\n')


# -- Training loop --
for epoch in range(1, EPOCHS + 1):

    model.train()

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        preds = model(X_batch)

        loss = loss_fn(preds, y_batch)

        loss.backward()

        optimizer.step()

    # -- Evaluation --
    model.eval()

    with torch.no_grad():

        train_preds = model(X_train_t).cpu().numpy()
        val_preds = model(X_val_t).cpu().numpy()
        test_preds = model(X_test_t).cpu().numpy()

    tr_mae = mean_absolute_error(y_train, train_preds)
    tr_rmse = np.sqrt(mean_squared_error(y_train, train_preds))

    val_mae = mean_absolute_error(y_val, val_preds)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    te_mae = mean_absolute_error(y_test, test_preds)
    te_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    te_r2 = r2_score(y_test, test_preds)

    history['train_mae'].append(tr_mae)
    history['train_rmse'].append(tr_rmse)

    history['val_mae'].append(val_mae)
    history['val_rmse'].append(val_rmse)

    history['test_mae'].append(te_mae)
    history['test_rmse'].append(te_rmse)

    if epoch % 10 == 0 or epoch == 1:

        print(
            f"Epoch [{epoch:>3}/{EPOCHS}] "
            f"Train MAE: {tr_mae:.4f} | Val MAE: {val_mae:.4f} | Test MAE: {te_mae:.4f} | "
            f"Train RMSE: {tr_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Test RMSE: {te_rmse:.4f} | "
            f"R²: {te_r2:.4f}"
        )


print(f'\n{"═"*60}')
print("FINAL TEST RESULTS")
print(f'{"═"*60}')
print(f'MAE: {te_mae:.4f}')
print(f'RMSE: {te_rmse:.4f}')
print(f'R²: {te_r2:.4f}')
print(f'{"═"*60}\n')


# -- Plot --
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

for ax, key, title in zip(axes, ['mae', 'rmse'], ['MAE', 'RMSE']):

    ax.plot(history[f'train_{key}'], label='Train', linewidth=2)

    ax.plot(history[f'val_{key}'], label='Validation', linewidth=2)

    ax.plot(history[f'test_{key}'], label='Test', linewidth=2)

    ax.set(title=title, xlabel='Epoch', ylabel=title)

    ax.legend()

    ax.grid(alpha=0.3)


plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_DIR, 'with_fe_trial_mlp_plot.png'),
    dpi=300
)

plt.close()
