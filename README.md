# Camera-Based Holistic FSL System (MVP)

Realtime Filipino Sign Language (FSL) static-sign recognizer using a laptop webcam.

Current MVP scope:
- A-Z static fingerspelling recognition
- 5 static greeting words from FSL-105 videos:
  - GOOD MORNING
  - GOOD EVENING
  - HELLO
  - THANK YOU
  - YOURE WELCOME
- Mode toggle between LETTER and WORD inference
- Optional NMS (neutral vs eyebrow_raise) via face landmarks

## Tech Stack
- Python + uv workflow
- OpenCV webcam capture
- MediaPipe Hands for landmark extraction
- MediaPipe FaceMesh for NMS features
- RandomForest classifiers (letters model + words model)

## Project Structure
- `src/fsl/data/`: data loading and word-frame extraction
- `src/fsl/features/`: landmark feature engineering
- `src/fsl/models/`: training utilities
- `src/fsl/inference/`: realtime prediction/stability logic
- `configs/`: YAML configs
- `models/`: saved model artifacts
- `reports/`: metrics and confusion matrix outputs

## Setup
1. Install `uv` (if not installed): https://docs.astral.sh/uv/
2. Create environment and sync deps:
   - `uv sync`
3. Run tests:
   - `uv run pytest`

## Config Paths (default)
- Letters images: `Hand-SIgns-A-Z`
- FSL-105 root: `FSL-105 A dataset for recognizing 105 Filipino sign language videos/FSL-105 A dataset for recognizing 105 Filipino sign language videos`

## Commands
1. Extract static word frames:
   - `uv run python -m fsl.data.extract_word_frames --config configs/data.yaml`
2. Train letters model:
   - `uv run python -m fsl.train_letters --config configs/train_letters.yaml`
3. Train words model:
   - `uv run python -m fsl.train_words --config configs/train_words.yaml`
4. Extract NMS frames:
   - `uv run python -m fsl.data.extract_nms_frames --config configs/nms_data.yaml`
5. Train NMS model:
   - `uv run python -m fsl.train_nms --config configs/train_nms.yaml`
6. Evaluate model:
   - `uv run python -m fsl.evaluate --model models/letters_model.joblib --split test`
   - `uv run python -m fsl.evaluate --model models/words_model.joblib --split test`
   - `uv run python -m fsl.evaluate --model models/nms_model.joblib --split test`
7. Run realtime app:
   - `uv run python -m fsl.app --config configs/app.yaml`

## Realtime Controls
- `f`: toggle fullscreen/windowed
- `m`: toggle mode (LETTER/WORD)
- `n`: toggle NMS on/off
- `[`: lower confidence threshold by 0.05
- `]`: raise confidence threshold by 0.05
- `-`: reduce stability window
- `+` / `=`: increase stability window
- `space`: commit current stable token
- `backspace`: delete last committed token
- `c`: clear output text
- `q`: quit

## Logo Overlay
- Config file: `configs/app.yaml` -> `logo`
- Default path: `assets/project_kumpas_logo.jpg`
- Position: upper-right corner
- Tuning:
  - `max_width`: logo display width in pixels
  - `margin_top`, `margin_right`: offsets from top-right edge
  - `opacity`: blend value from `0.0` to `1.0`

## Scope and Limitations
- Static signs only (no dynamic temporal modeling)
- Word mode relies on static-frame extraction from greeting sign videos
- NMS currently supports 2 classes from `data_nsm` (`neutral`, `eyebrow_raise`)
