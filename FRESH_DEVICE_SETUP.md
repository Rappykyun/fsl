# Fresh Device Setup (No Git, No Python)

This guide is for a brand-new machine.

## 1. Install Git

### Windows

```powershell
winget install --id Git.Git -e
```

### macOS

```bash
brew install git
```

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y git
```

Check:

```bash
git --version
```

## 2. Install UV (Python + package manager)

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS/Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Check:

```bash
uv --version
```

## 3. Install Python via UV

```bash
uv python install 3.10
```

Check:

```bash
uv python list
```

## 4. Clone the Project

```bash
git clone https://github.com/Rappykyun/fsl.git
cd fsl
```

## 5. Install Project Dependencies

```bash
uv sync --extra dev
```

Optional test check:

```bash
uv run pytest -q
```

## 6. Copy Datasets into Project Root

Place these folders in the project root:

- `Hand-Signs-A-Z/`
- `FSL-105 A dataset for recognizing 105 Filipino sign language videos/`
- `data_nsm/`

Expected structure:

```text
fsl/
  Hand-Signs-A-Z/
  FSL-105 A dataset for recognizing 105 Filipino sign language videos/
  data_nsm/
  src/
  configs/
```

## 7. Run Full Pipeline (Extraction -> Training -> Evaluation)

```bash
uv run python -m fsl.data.extract_word_frames --config configs/data.yaml
uv run python -m fsl.data.extract_nms_frames --config configs/nms_data.yaml

uv run python -m fsl.train_letters --config configs/train_letters.yaml
uv run python -m fsl.train_words --config configs/train_words.yaml
uv run python -m fsl.train_nms --config configs/train_nms.yaml

uv run python -m fsl.evaluate --model models/letters_model.joblib --split test
uv run python -m fsl.evaluate --model models/words_model.joblib --split test
uv run python -m fsl.evaluate --model models/nms_model.joblib --split test
```

## 8. Run Realtime App

```bash
uv run python -m fsl.app --config configs/app.yaml
```

Controls:

- `f` fullscreen toggle
- `m` mode toggle (LETTER/WORD)
- `n` NMS toggle
- `[` `]` confidence down/up
- `-` `+` stability down/up
- `space` commit token
- `c` clear
- `q` quit

## 9. Quick Start (If You Already Have Trained Models)

If `models/*.joblib` already exists on your machine, you can skip extraction/training and run:

```bash
uv run python -m fsl.app --config configs/app.yaml
```

## 10. Common Issues

### `uv: command not found`

Restart terminal after installation, then run:

```bash
uv --version
```

### Webcam does not open

Check OS camera permission for terminal/Python.

### MediaPipe warning messages

Warnings are common; continue unless there is a Python traceback/error.
