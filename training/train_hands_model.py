#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
from typing import List, Tuple, Dict

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers


# Chemin dataset
DATASET_ROOT = "./data/HandGestures"

# Mode d'entrée
MODE = "landmarks"    # "landmarks" | "images"
IMG_SIZE = 224         # utilisé si MODE == "images"

# Entraînement
EPOCHS = 50
BATCH_SIZE = 16
VAL_SPLIT = 0.2
LEARNING_RATE = 1e-3
EARLY_STOPPING = True
PATIENCE = 10
CLASS_BALANCE = True  # pondération inverse à la fréquence
AUGMENT = True
LANDMARKS_AUGMENT = True  # augmentation pour les landmarks (bruit, rotation légère)
NORMALIZE_ORIENTATION = True  # normaliser l'orientation de la main (toujours vers la droite)

# Sorties
MODEL_OUT = "models/hands_landmarks.h5" if MODE == "landmarks" else "models/hands_images.h5"
LABEL_MAP_OUT = "models/label_map.json"
HISTORY_OUT = "models/training_history.json"

# Divers
SEED = 42


random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def list_classes(root: str) -> List[str]:
    classes = []
    if os.path.exists(root):
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isdir(p) and not name.startswith('.'):
                classes.append(name)
    return classes


def count_samples_per_class(root: str, classes: List[str], mode: str) -> Dict[str, int]:
    counts = {}
    for cls in classes:
        count = 0
        if mode == "landmarks":
            lm_dir = os.path.join(root, cls, "landmarks")
            if os.path.exists(lm_dir):
                count = len([f for f in os.listdir(lm_dir) if f.lower().endswith('.npz')])
        else:
            img_dir = os.path.join(root, cls, "images")
            if os.path.exists(img_dir):
                count = len([f for f in os.listdir(img_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        counts[cls] = count
    return counts


def normalize_landmark_orientation(lm: np.ndarray) -> np.ndarray:
    """
    Normalise l'orientation de la main pour qu'elle soit toujours orientée vers la droite.
    Utilise le vecteur du poignet vers l'index pour déterminer l'orientation.
    """
    if lm.shape != (21, 3):
        return lm
    
    wrist = lm[0, :2]  # (x, y) du poignet
    index_mcp = lm[5, :2]  # (x, y) de la base de l'index
    
    # Vecteur du poignet vers l'index
    direction = index_mcp - wrist
    
    # Si la main pointe vers la gauche (direction.x < 0), on la retourne
    if direction[0] < 0:
        lm_flipped = lm.copy()
        lm_flipped[:, 0] = -lm_flipped[:, 0]  # Inverser l'axe X
        return lm_flipped
    
    return lm


def augment_landmarks(lm: np.ndarray, noise_std: float = 0.02) -> np.ndarray:
    """Ajoute du bruit gaussien aux landmarks pour l'augmentation de données"""
    noise = np.random.normal(0, noise_std, lm.shape).astype('float32')
    lm_aug = lm + noise
    # Re-normaliser pour garder les valeurs dans une plage raisonnable
    lm_aug = np.clip(lm_aug, -1.5, 1.5)
    return lm_aug


def load_landmarks_dataset(root: str, classes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for idx, cls in enumerate(classes):
        lm_dir = os.path.join(root, cls, "landmarks")
        if not os.path.exists(lm_dir):
            print(f"AVERTISSEMENT: dossier landmarks manquant pour la classe '{cls}'")
            continue
        files = [f for f in os.listdir(lm_dir) if f.lower().endswith('.npz')]
        if not files:
            print(f"AVERTISSEMENT: aucun fichier .npz trouvé pour la classe '{cls}'")
            continue
        for f in files:
            p = os.path.join(lm_dir, f)
            try:
                data = np.load(p)
                if 'landmarks' not in data:
                    continue
                lm = data['landmarks'].astype('float32')
                if lm.shape == (21, 3):
                    # Normalisation de l'orientation si activée
                    if NORMALIZE_ORIENTATION:
                        lm = normalize_landmark_orientation(lm)
                    
                    X.append(lm.reshape(-1))
                    y.append(idx)
                    
                    # Augmentation de données si activée
                    if LANDMARKS_AUGMENT:
                        for _ in range(2):  # 2 variantes augmentées par échantillon
                            lm_aug = augment_landmarks(lm)
                            X.append(lm_aug.reshape(-1))
                            y.append(idx)
                else:
                    print(f"AVERTISSEMENT: format de landmarks incorrect dans {p}: {lm.shape}")
            except Exception as e:
                print(f"ERREUR lors du chargement de {p}: {e}")
    if len(X) == 0:
        return np.array([], dtype='float32').reshape(0, 63), np.array([], dtype='int64')
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int64')
    return X, y


def load_images_dataset(root: str, classes: List[str], img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for idx, cls in enumerate(classes):
        img_dir = os.path.join(root, cls, "images")
        if not os.path.exists(img_dir):
            print(f"AVERTISSEMENT: dossier images manquant pour la classe '{cls}'")
            continue
        files = [f for f in os.listdir(img_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            print(f"AVERTISSEMENT: aucune image trouvée pour la classe '{cls}'")
            continue
        for f in files:
            p = os.path.join(img_dir, f)
            try:
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                img = (img.astype('float32') / 255.0)
                X.append(img)
                y.append(idx)
            except Exception as e:
                print(f"ERREUR lors du chargement de {p}: {e}")
    if len(X) == 0:
        return np.array([], dtype='float32').reshape(0, img_size, img_size, 3), np.array([], dtype='int64')
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int64')
    return X, y


def build_mlp(num_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=(63,))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn(num_classes: int, img_size: int) -> tf.keras.Model:
    inp = layers.Input(shape=(img_size, img_size, 3))
    x = inp
    if AUGMENT:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.02)(x)
        x = layers.RandomZoom(0.1)(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> Dict[int, float]:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    inv = inv * (num_classes / inv.sum())
    return {i: float(inv[i]) for i in range(num_classes)}


def save_history(history, path: str) -> None:
    history_dict = {}
    for key in history.history.keys():
        history_dict[key] = [float(v) for v in history.history[key]]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history_dict, f, indent=2, ensure_ascii=False)


def main() -> None:
    print("=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE DE RECONNAISSANCE DE GESTES")
    print("=" * 60)
    print(f"Mode: {MODE}")
    print(f"Dataset: {DATASET_ROOT}")
    print()

    ensure_dir(os.path.dirname(MODEL_OUT))
    ensure_dir(os.path.dirname(LABEL_MAP_OUT))

    classes = list_classes(DATASET_ROOT)
    if not classes:
        print(f"ERREUR: aucune classe trouvée dans {DATASET_ROOT}")
        print("Assurez-vous que le dataset contient des sous-dossiers pour chaque classe.")
        return

    print(f"Classes trouvées ({len(classes)}): {', '.join(classes)}")
    print()

    counts = count_samples_per_class(DATASET_ROOT, classes, MODE)
    print("Échantillons par classe:")
    for cls in classes:
        print(f"  - {cls}: {counts[cls]} échantillons")
    print()

    empty_classes = [cls for cls, count in counts.items() if count == 0]
    if empty_classes:
        print(f"AVERTISSEMENT: classes sans échantillons: {', '.join(empty_classes)}")
        print("Ces classes seront ignorées lors de l'entraînement.")
        print()

    label_map = {i: c for i, c in enumerate(classes)}
    with open(LABEL_MAP_OUT, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Label map sauvegardé: {LABEL_MAP_OUT}")
    print()

    if MODE == "landmarks":
        print("Chargement du dataset de landmarks...")
        X, y = load_landmarks_dataset(DATASET_ROOT, classes)
        if X.shape[0] == 0:
            print("ERREUR: aucun échantillon landmarks trouvé.")
            return
        print(f"Dataset chargé: {X.shape[0]} échantillons, {X.shape[1]} features")
    else:
        print("Chargement du dataset d'images...")
        X, y = load_images_dataset(DATASET_ROOT, classes, IMG_SIZE)
        if X.shape[0] == 0:
            print("ERREUR: aucun échantillon images trouvé.")
            return
        print(f"Dataset chargé: {X.shape[0]} échantillons, taille image: {IMG_SIZE}x{IMG_SIZE}")
    
    print(f"Distribution des classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    print()

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("ERREUR: au moins 2 classes avec des échantillons sont nécessaires pour l'entraînement.")
        return

    stratify_param = y if len(unique_classes) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=SEED, stratify=stratify_param
    )
    print(f"Split train/validation: {len(X_train)} / {len(X_val)} échantillons")
    print()

    if MODE == "landmarks":
        print("Construction du modèle MLP...")
        model = build_mlp(num_classes=len(classes))
    else:
        print("Construction du modèle CNN...")
        model = build_cnn(num_classes=len(classes), img_size=IMG_SIZE)
    
    print(f"Modèle créé: {model.count_params()} paramètres")
    print()
    model.summary()
    print()

    cb = []
    if EARLY_STOPPING:
        cb.append(callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ))
    
    checkpoint_path = MODEL_OUT.replace('.h5', '_best.h5')
    cb.append(callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        mode='min'
    ))
    
    cb.append(callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    ))

    class_weight = None
    if CLASS_BALANCE:
        class_weight = compute_class_weights(y_train, num_classes=len(classes))
        print(f"Pondération des classes: {class_weight}")
        print()

    print("Début de l'entraînement...")
    print("=" * 60)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=cb,
        shuffle=True,
        verbose=1,
    )
    print("=" * 60)
    print()

    model.save(MODEL_OUT)
    print(f"Modèle sauvegardé: {MODEL_OUT}")
    
    save_history(history, HISTORY_OUT)
    print(f"Historique sauvegardé: {HISTORY_OUT}")
    print()

    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    
    best_epoch = np.argmax(history.history['val_accuracy'])
    best_val_acc = history.history['val_accuracy'][best_epoch]
    
    print("Résultats finaux:")
    print(f"  - Précision train: {train_acc:.4f}")
    print(f"  - Précision validation: {val_acc:.4f}")
    print(f"  - Perte train: {train_loss:.4f}")
    print(f"  - Perte validation: {val_loss:.4f}")
    print(f"  - Meilleure précision validation: {best_val_acc:.4f} (epoch {best_epoch + 1})")
    print()
    print("Entraînement terminé avec succès!")


if __name__ == "__main__":
    main()


