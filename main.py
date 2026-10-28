"""Tutorial 11: one-step sine-wave forecasting with a simple PyTorch RNN."""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
POINTS = 1000
SEQUENCE_LENGTH = 30
TRAIN_FRACTION = 0.8
EPOCHS = 50
BATCH_SIZE = 64


def make_data():
    """Create a reproducible noisy sine wave and fixed-window samples."""
    rng = np.random.default_rng(SEED)
    time = np.linspace(0, 40 * np.pi, POINTS, dtype=np.float32)
    clean = np.sin(time)
    noisy = clean + rng.normal(0, 0.12, size=POINTS).astype(np.float32)

    split_at = int(POINTS * TRAIN_FRACTION)
    windows, targets = [], []
    for target_index in range(SEQUENCE_LENGTH, POINTS):
        windows.append(noisy[target_index - SEQUENCE_LENGTH:target_index])
        targets.append(noisy[target_index])

    x = np.asarray(windows, dtype=np.float32)[..., None]
    y = np.asarray(targets, dtype=np.float32)[:, None]
    # Training targets end before the chronological holdout boundary.
    train_count = split_at - SEQUENCE_LENGTH
    return time, noisy, split_at, x[:train_count], y[:train_count], x[train_count:], y[train_count:]


class SineRNN(nn.Module):
    """A single recurrent layer followed by one linear prediction layer."""
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, sequence):
        recurrent_output, _ = self.rnn(sequence)
        return self.output(recurrent_output[:, -1, :])


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    time, values, split_at, x_train, y_train, x_test, y_test = make_data()

    # Save the actual synthetic series used in the run.
    np.savetxt(out_dir / "sine_data.csv", np.column_stack((time, values)),
               delimiter=",", header="time,noisy_sine", comments="")

    plt.figure(figsize=(10, 3.8))
    plt.plot(time, values, color="#3568a8", linewidth=1, label="Noisy sine data")
    plt.axvline(time[split_at], color="#d27b28", linestyle="--", label="Train/test boundary")
    plt.title("Synthetic Sine Time Series with Noise")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sine_data.png", dpi=160)
    plt.close()

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    model = SineRNN()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    print(f"Device: CPU | Training samples: {len(train_dataset)} | Test samples: {len(x_test)}")
    print(f"Input window: {SEQUENCE_LENGTH} time steps | Epochs: {EPOCHS}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        average_loss = total_loss / len(train_dataset)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:02d}/{EPOCHS} - training MSE: {average_loss:.6f}")

    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(x_test)).squeeze(1).numpy()
    test_mse = float(np.mean((predictions - y_test.squeeze(1)) ** 2))
    print(f"Held-out one-step test MSE: {test_mse:.6f}")
    np.savetxt(out_dir / "predictions.csv",
               np.column_stack((time[split_at:], y_test.squeeze(1), predictions)),
               delimiter=",", header="time,actual,predicted", comments="")

    plt.figure(figsize=(10, 4.2))
    plt.plot(time, values, color="#b5b5b5", linewidth=0.8, label="Noisy sine data")
    plt.plot(time[split_at:], y_test.squeeze(1), color="#3568a8", linewidth=1.0, label="Actual holdout values")
    plt.plot(time[split_at:], predictions, color="#d27b28", linewidth=1.2, label="RNN one-step predictions")
    plt.axvline(time[split_at], color="#555555", linestyle="--", linewidth=0.8, label="Train/test boundary")
    plt.title("RNN One-Step Forecast on the Held-Out Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "predictions.png", dpi=160)
    plt.close()
    torch.save(model.state_dict(), out_dir / "sine_rnn.pth")
    print(f"Saved plots, data, predictions, and model to: {out_dir}")


if __name__ == "__main__":
    main()

