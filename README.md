# Wheelchair Gestures – Dataset and Training (MediaPipe Hands + Keras)

## Overview
- Collection of a hand gesture dataset from webcam via MediaPipe Hands (21 points).
- Training of a Keras model:
  - Default: MLP on landmarks (21×3 → 63 features)
  - Option: CNN on cropped images (IMG_SIZE×IMG_SIZE)
- No CLI: modify the constants at the top of the scripts according to your needs.

## Prerequisites
```bash
# Windows - Python 3.10 supported (version profile adapted)
pip install -r requirements.txt
```

## Directory Structure
- `collector/hand_dataset_collector.py`: captures webcam, saves `images/` and `landmarks/` by class.
- `training/train_hands_model.py`: trains MLP (landmarks) or CNN (images) depending on configuration.
- `data/HandGestures/`: dataset root (subdirectories = classes)
- `models/`: `.h5` models and `label_map.json`

## Dataset Collection
- Open `collector/hand_dataset_collector.py` and adjust the constants at the top of the file, e.g.:
  - `DATASET_ROOT`, `CLASS_NAMES`, `CAMERA_INDEX`, `IMG_SIZE`, `SAVE_FULL`, `SAVE_FORMAT`, etc.
- Run:
```bash
python collector/hand_dataset_collector.py
```
- Keyboard shortcuts:
  - `A` / `D` : change active class
  - `SPACE`   : capture a sample
  - `R`       : toggle continuous recording (at `CONTINUOUS_FPS`)
  - `Q` / `ESC`: quit

Each sample saves:
- `images/<uuid>.jpg` (cropped to `IMG_SIZE`)
- `landmarks/<uuid>.npz` with key `landmarks` (21×3 normalized)
- Optional: `images_full/<uuid>.jpg` if `SAVE_FULL=True`

## Training
- Open `training/train_hands_model.py` and adjust the constants at the top of the file:
  - `DATASET_ROOT`, `MODE`, `IMG_SIZE`, `EPOCHS`, `BATCH_SIZE`, `VAL_SPLIT`, etc.
- Run:
```bash
python training/train_hands_model.py
```
- Outputs:
  - `.h5` model in `models/`
  - `label_map.json` (id → class name)

## Notes
- Landmarks are normalized: centered at the wrist and scaled by a hand size measurement.
- Add/remove classes by creating/deleting subdirectories in `data/HandGestures`.

### Windows Tip (PowerShell)
```powershell
# Create a venv with Python 3.10
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```


