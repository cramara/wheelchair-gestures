# Wheelchair Gestures – Dataset et Entraînement (MediaPipe Hands + Keras)

## Vue d’ensemble
- Collecte d’un dataset de gestes de main à partir de la webcam via MediaPipe Hands (21 points).
- Entraînement d’un modèle Keras:
  - Par défaut: MLP sur landmarks (21×3 → 63 features)
  - Option: CNN sur images recadrées (IMG_SIZE×IMG_SIZE)
- Pas de CLI: modifiez les constantes en tête des scripts selon vos besoins.

## Pré-requis
```bash
pip install -r requirements.txt
```

## Arborescence
- `collector/hand_dataset_collector.py`: capture webcam, sauvegarde `images/` et `landmarks/` par classe.
- `training/train_hands_model.py`: entraîne MLP (landmarks) ou CNN (images) suivant la configuration.
- `data/HandGestures/`: racine du dataset (sous-dossiers = classes)
- `models/`: modèles `.h5` et `label_map.json`

## Collecte du dataset
- Ouvrez `collector/hand_dataset_collector.py` et ajustez les constantes en haut du fichier, par ex.:
  - `DATASET_ROOT`, `CLASS_NAMES`, `CAMERA_INDEX`, `IMG_SIZE`, `SAVE_FULL`, `SAVE_FORMAT`, etc.
- Lancez:
```bash
python collector/hand_dataset_collector.py
```
- Raccourcis clavier:
  - `[` / `]` : changer de classe active
  - `SPACE`   : capturer un échantillon
  - `R`       : basculer enregistrement continu (à `CONTINUOUS_FPS`)
  - `Q` / `ESC`: quitter

Chaque échantillon sauvegarde:
- `images/<uuid>.jpg` (recadré à `IMG_SIZE`)
- `landmarks/<uuid>.npz` avec la clé `landmarks` (21×3 normalisés)
- Optionnel: `images_full/<uuid>.jpg` si `SAVE_FULL=True`

## Entraînement
- Ouvrez `training/train_hands_model.py` et ajustez les constantes en haut du fichier:
  - `DATASET_ROOT`, `MODE`, `IMG_SIZE`, `EPOCHS`, `BATCH_SIZE`, `VAL_SPLIT`, etc.
- Lancez:
```bash
python training/train_hands_model.py
```
- Sorties:
  - Modèle `.h5` dans `models/`
  - `label_map.json` (id → nom de classe)

## Notes
- Les landmarks sont normalisés: centrés au poignet et mis à l’échelle par une mesure de taille de main.
- Ajoutez/retirez des classes en créant/supprimant des sous-dossiers dans `data/HandGestures`.
