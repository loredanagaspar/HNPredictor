# models/regressor.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

class UpvoteRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def train_model(dataloader, base_embedding_dim, val_loader=None):
    model = UpvoteRegressor(base_embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        total_loss = 0.0
        model.train()
        for xb, yb in dataloader:
            pred = model(xb).squeeze(1)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")

    if val_loader:
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).squeeze(1)
                all_preds.extend(pred.tolist())
                all_targets.extend(yb.tolist())

        mae = mean_absolute_error(all_targets, all_preds)
        rmse = math.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)

        print(f"\nValidation Metrics:")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")

    return model
