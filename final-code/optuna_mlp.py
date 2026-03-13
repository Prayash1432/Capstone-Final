import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import optuna
import os
import warnings
warnings.filterwarnings('ignore')


class TabEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -- Config --
SEED = 42
DATA_PATH = "../data/final_engineered_ds.csv"
SAVE_DIR = "./results"
OPTUNA_TRIALS = 30

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- Load & split data --
df = pd.read_csv(DATA_PATH)
feature_cols = df.drop(columns=['CS 28 day (Mpa)']).keys()
X = df[feature_cols].values.astype(np.float32)
y = df['CS 28 day (Mpa)'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val   = sc.transform(X_val)
X_test  = sc.transform(X_test)

print(f'Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} | Features: {len(feature_cols)}')

# -- Convert to tensors --
X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).to(device)
X_val_t   = torch.tensor(X_val).to(device)
y_val_t   = torch.tensor(y_val).to(device)
X_test_t  = torch.tensor(X_test).to(device)
y_test_t  = torch.tensor(y_test).to(device)


# -- Optuna objective --
def objective(trial):
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    dropout    = trial.suggest_float("dropout", 0.0, 0.4)
    epochs     = trial.suggest_int("epochs", 100, 500, step=100)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    model  = TabEncoder(len(feature_cols), hidden_dim, dropout).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            opt.zero_grad()
            loss_fn(model(X_batch), y_batch).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).cpu().numpy()

    return mean_squared_error(y_val, val_preds)  # minimize val MSE


# -- Run Optuna --
print(f"\nRunning Optuna ({OPTUNA_TRIALS} trials)...\n")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=OPTUNA_TRIALS)

best = study.best_params
print(f"Best Params: {best}")


# -- Train final model with best params --
EPOCHS     = best["epochs"]
BATCH_SIZE = best["batch_size"]

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
model    = TabEncoder(len(feature_cols), best["hidden_dim"], best["dropout"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best["lr"])
loss_fn  = nn.MSELoss()

history = {'train_mae': [], 'train_rmse': [], 'val_mae': [], 'val_rmse': [], 'test_mae': [], 'test_rmse': []}

print(f'\nTraining final model for {EPOCHS} epochs...\n')

for epoch in range(1, EPOCHS + 1):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss_fn(model(X_batch), y_batch).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_t).cpu().numpy()
        val_preds   = model(X_val_t).cpu().numpy()
        test_preds  = model(X_test_t).cpu().numpy()

    history['train_mae'].append(mean_absolute_error(y_train, train_preds))
    history['train_rmse'].append(np.sqrt(mean_squared_error(y_train, train_preds)))
    history['val_mae'].append(mean_absolute_error(y_val, val_preds))
    history['val_rmse'].append(np.sqrt(mean_squared_error(y_val, val_preds)))
    history['test_mae'].append(mean_absolute_error(y_test, test_preds))
    history['test_rmse'].append(np.sqrt(mean_squared_error(y_test, test_preds)))

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:>3}/{EPOCHS}] "
              f"Train MAE: {history['train_mae'][-1]:.4f} | Val MAE: {history['val_mae'][-1]:.4f} | "
              f"Test MAE: {history['test_mae'][-1]:.4f}")

te_r2 = r2_score(y_test, test_preds)
print(f'\n{"═"*60}')
print("FINAL TEST RESULTS")
print(f'{"═"*60}')
print(f'MAE:  {history["test_mae"][-1]:.4f}')
print(f'RMSE: {history["test_rmse"][-1]:.4f}')
print(f'R²:   {te_r2:.4f}')
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
plt.savefig(os.path.join(SAVE_DIR, 'demo_with_fe_trial_mlp_plot.png'), dpi=300)
plt.close()