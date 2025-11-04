#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
from typing import List, Tuple

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
EPOCHS = 30
BATCH_SIZE = 32
VAL_SPLIT = 0.2
LEARNING_RATE = 1e-3
EARLY_STOPPING = True
PATIENCE = 8
CLASS_BALANCE = False  # pondération inverse à la fréquence
AUGMENT = True

# Sorties
MODEL_OUT = "models/thumbs_landmarks.h5"
LABEL_MAP_OUT = "models/label_map.json"

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
            if os.path.isdir(p):
                classes.append(name)
    return classes


def load_landmarks_dataset(root: str, classes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for idx, cls in enumerate(classes):
        lm_dir = os.path.join(root, cls, "landmarks")
        if not os.path.exists(lm_dir):
            continue
        for f in os.listdir(lm_dir):
            if f.lower().endswith('.npz'):
                p = os.path.join(lm_dir, f)
                try:
                    data = np.load(p)
                    lm = data['landmarks'].astype('float32')  # (21,3)
                    if lm.shape == (21, 3):
                        X.append(lm.reshape(-1))  # 63
                        y.append(idx)
                except Exception:
                    pass
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int64')
    return X, y


def load_images_dataset(root: str, classes: List[str], img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for idx, cls in enumerate(classes):
        img_dir = os.path.join(root, cls, "images")
        if not os.path.exists(img_dir):
            continue
        for f in os.listdir(img_dir):
            fl = f.lower()
            if fl.endswith('.jpg') or fl.endswith('.jpeg') or fl.endswith('.png'):
                p = os.path.join(img_dir, f)
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                img = (img.astype('float32') / 255.0)
                X.append(img)
                y.append(idx)
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int64')
    return X, y


def build_mlp(num_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=(63,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
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


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> dict:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    inv = inv * (num_classes / inv.sum())
    return {i: float(inv[i]) for i in range(num_classes)}


def main() -> None:
    ensure_dir(os.path.dirname(MODEL_OUT))
    ensure_dir(os.path.dirname(LABEL_MAP_OUT))

    classes = list_classes(DATASET_ROOT)
    if not classes:
        print("ERREUR: aucune classe trouvée dans", DATASET_ROOT)
        return

    label_map = {i: c for i, c in enumerate(classes)}
    with open(LABEL_MAP_OUT, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    if MODE == "landmarks":
        X, y = load_landmarks_dataset(DATASET_ROOT, classes)
        if X.shape[0] == 0:
            print("ERREUR: aucun sample landmarks trouvé.")
            return
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VAL_SPLIT, random_state=SEED, stratify=y if len(np.unique(y)) > 1 else None
        )
        model = build_mlp(num_classes=len(classes))
        cb = []
        if EARLY_STOPPING:
            cb.append(callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True))
        class_weight = None
        if CLASS_BALANCE:
            class_weight = compute_class_weights(y_train, num_classes=len(classes))
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=cb,
            shuffle=True,
            verbose=2,
        )
        model.save(MODEL_OUT)
    else:
        X, y = load_images_dataset(DATASET_ROOT, classes, IMG_SIZE)
        if X.shape[0] == 0:
            print("ERREUR: aucun sample images trouvé.")
            return
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VAL_SPLIT, random_state=SEED, stratify=y if len(np.unique(y)) > 1 else None
        )
        model = build_cnn(num_classes=len(classes), img_size=IMG_SIZE)
        cb = []
        if EARLY_STOPPING:
            cb.append(callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True))
        class_weight = None
        if CLASS_BALANCE:
            class_weight = compute_class_weights(y_train, num_classes=len(classes))
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=cb,
            shuffle=True,
            verbose=2,
        )
        model.save(MODEL_OUT)


if __name__ == "__main__":
    main()
