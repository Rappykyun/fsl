# Setup Guide: Camera-Based Holistic FSL (UV + Python)

This guide covers full setup from installation to training, evaluation, and realtime demo.

## 1. Prerequisites

- OS: Windows/Linux/macOS
- Python: 3.10+
- Webcam for realtime demo
- Project root: `fsl`

Required data folders in project root:
- `Hand-SIgns-A-Z/`
- `FSL-105 A dataset for recognizing 105 Filipino sign language videos/...`
- `data_nsm/` (with `neutral/` and `eyebrow_raise/`)

## 2. Install UV (if not installed)

Linux/macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:
```bash
uv --version
```

## 3. Install Project Dependencies

From project root:
```bash
uv sync --extra dev
```

Optional test check:
```bash
uv run pytest -q
```

## 4. Data Extraction Commands

### 4.1 Extract static word frames (manual word model)
```bash
uv run python -m fsl.data.extract_word_frames --config configs/data.yaml
```

Expected outputs:
- `artifacts/word_frames_train.csv`
- `artifacts/word_frames_test.csv`
- `data/word_frames/train/...`
- `data/word_frames/test/...`

### 4.2 Extract NMS frames (face/NMS model)
```bash
uv run python -m fsl.data.extract_nms_frames --config configs/nms_data.yaml
```

Expected outputs:
- `artifacts/nms_frames_train.csv`
- `artifacts/nms_frames_test.csv`
- `data/nms_frames/train/...`
- `data/nms_frames/test/...`

## 5. Training Commands

### 5.1 Train letters model (A-Z)
```bash
uv run python -m fsl.train_letters --config configs/train_letters.yaml
```

Expected artifacts:
- `models/letters_model.joblib`
- `models/letters_label_encoder.joblib`
- `reports/letters_confusion_matrix.png`

### 5.2 Train words model
```bash
uv run python -m fsl.train_words --config configs/train_words.yaml
```

Expected artifacts:
- `models/words_model.joblib`
- `models/words_label_encoder.joblib`
- `reports/words_confusion_matrix.png`

### 5.3 Train NMS model (neutral vs eyebrow_raise)
```bash
uv run python -m fsl.train_nms --config configs/train_nms.yaml
```

Expected artifacts:
- `models/nms_model.joblib`
- `models/nms_label_encoder.joblib`
- `reports/nms_confusion_matrix.png`

## 6. Evaluation Commands

```bash
uv run python -m fsl.evaluate --model models/letters_model.joblib --split test
uv run python -m fsl.evaluate --model models/words_model.joblib --split test
uv run python -m fsl.evaluate --model models/nms_model.joblib --split test
```

Metrics file:
- `reports/metrics.json`

## 7. Run Realtime Demo

```bash
uv run python -m fsl.app --config configs/app.yaml
```

Controls:
- `f` toggle fullscreen/windowed
- `m` toggle LETTER/WORD mode
- `n` toggle NMS on/off
- `[` lower confidence threshold
- `]` raise confidence threshold
- `-` reduce stability frames
- `+` / `=` increase stability frames
- `space` commit current token
- `backspace` delete last committed token
- `c` clear output text
- `q` quit

## 8. Full Pipeline (Copy-Paste)

```bash
uv sync --extra dev
uv run python -m fsl.data.extract_word_frames --config configs/data.yaml
uv run python -m fsl.data.extract_nms_frames --config configs/nms_data.yaml
uv run python -m fsl.train_letters --config configs/train_letters.yaml
uv run python -m fsl.train_words --config configs/train_words.yaml
uv run python -m fsl.train_nms --config configs/train_nms.yaml
uv run python -m fsl.evaluate --model models/letters_model.joblib --split test
uv run python -m fsl.evaluate --model models/words_model.joblib --split test
uv run python -m fsl.evaluate --model models/nms_model.joblib --split test
uv run python -m fsl.app --config configs/app.yaml
```

## 9. Notes

- MediaPipe/TFLite warnings during extraction/training are common; they are usually not fatal.
- NMS training can take longer due to face-landmark processing and hyperparameter search.
- If realtime is unstable, tune thresholds and stability using keyboard controls in the app.
