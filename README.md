# Tutorial 11: RNN Time-Series Forecasting

A beginner-friendly PyTorch program that learns one-step-ahead predictions from a synthetic sine wave with added noise. It follows the CSL75 Tutorial 11 exercise: create and plot noisy sine data, prepare fixed-length windows, train an `nn.RNN` followed by `nn.Linear` with MSE loss and Adam, print training loss, and plot predictions.

## Run in VS Code on Windows

1. Install Python 3.12 or newer and the Python extension in VS Code.
2. Open this folder in VS Code (**File → Open Folder**).
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation in this terminal, run:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   .\.venv\Scripts\Activate.ps1
   ```

5. Install the required libraries:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

6. In VS Code, press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, then select the interpreter inside this folder's `.venv`.
7. Run the program in the terminal:

   ```powershell
   python main.py
   ```

The run creates `results/` with the generated sine data, data and prediction plots, held-out predictions, and the trained model. A fixed seed makes runs reproducible.

## Publish as a separate GitHub repository

This folder is already initialized as its own local Git repository. To publish it after signing in to GitHub in VS Code:

1. Open the **Source Control** view (`Ctrl+Shift+G`).
2. Select **Publish to GitHub** (or **Publish Branch** if prompted).
3. Sign in to the GitHub account you want to own this tutorial repository.
4. Choose **Private** for a personal copy, or **Public** if your instructor needs to open the link.
5. Keep the repository name `Tutorial-11-RNN-Time-Series` and confirm publishing.

The report's GitHub link can be filled in with the URL VS Code shows after publishing.

## Model and data

- 1,000 points of a sine wave with Gaussian noise (standard deviation 0.12)
- Fixed input window: 30 time steps
- Chronological 80/20 training and holdout split
- One `nn.RNN` layer (32 hidden units) and one `nn.Linear` output layer
- Mean Squared Error loss and Adam optimizer
- One-step-ahead predictions: each holdout prediction uses the preceding observed 30 values

## Files

- `main.py` - data generation, model, training, evaluation, and plotting
- `requirements.txt` - Python dependencies
- `Tutorial-11-Report.pdf` - report in the numbered-section style used in the course examples
- `Tutorial-11-Report-Word.docx` - editable Word version with the terminal screenshot and plots
- `results/` - generated artifacts from the verified run
