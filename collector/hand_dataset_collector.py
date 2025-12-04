#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import uuid
import math
import random
import copy
from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp


# Chemins et classes
DATASET_ROOT = "./data/HandGestures"
CLASS_NAMES = ["ThumbUp", "ThumbDown"]  # ou None pour auto-découverte

# Caméra et capture
CAMERA_INDEX = 0
MIRROR = False

# Sauvegarde (logique 3: landmarks uniquement par défaut)
IMG_SIZE = 224
SAVE_FULL = False          # sauvegarder aussi le crop original avant resize dans images_full/
SAVE_FORMAT = "both"  # "both" | "images" | "landmarks"

# MediaPipe Hands
MAX_HANDS = 1
HAND_SELECT = "auto"       # "auto" | "left" | "right"
MIN_DETECT = 0.5
MIN_TRACK = 0.5
BBOX_MARGIN = 0.2          # marge relative autour de la bbox

# Enregistrement continu
CONTINUOUS_FPS = 5.0

# Divers
SEED = 42


random.seed(SEED)
np.random.seed(SEED)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def discover_classes(root: str) -> List[str]:
    if not os.path.exists(root):
        return []
    classes = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            classes.append(name)
    return classes


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def compute_bbox_from_landmarks(lm_img_xy: np.ndarray, w: int, h: int, margin: float) -> Tuple[int, int, int, int]:
    x = lm_img_xy[:, 0]
    y = lm_img_xy[:, 1]
    xmin = int(np.floor(x.min()))
    xmax = int(np.ceil(x.max()))
    ymin = int(np.floor(y.min()))
    ymax = int(np.ceil(y.max()))
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    bw = (xmax - xmin)
    bh = (ymax - ymin)
    # marge relative
    bw = int(bw * (1.0 + margin))
    bh = int(bh * (1.0 + margin))
    xmin = int(cx - bw / 2)
    xmax = int(cx + bw / 2)
    ymin = int(cy - bh / 2)
    ymax = int(cy + bh / 2)
    # clamp
    xmin = clamp(xmin, 0, w - 1)
    xmax = clamp(xmax, 0, w - 1)
    ymin = clamp(ymin, 0, h - 1)
    ymax = clamp(ymax, 0, h - 1)
    if xmax <= xmin:
        xmax = min(w - 1, xmin + 1)
    if ymax <= ymin:
        ymax = min(h - 1, ymin + 1)
    return xmin, ymin, xmax, ymax


def normalize_landmarks_xy_z(lm_xyz: np.ndarray) -> np.ndarray:
    # lm_xyz: (21, 3), coordonnées image (x, y) et profondeur relative z fournie par MediaPipe
    # Centrage au poignet (index 0)
    wrist = lm_xyz[0, :].copy()
    lm_rel = lm_xyz - wrist
    # Échelle: distance poignet → moyenne MCP (5,9,13,17)
    mcp_indices = [5, 9, 13, 17]
    mcp_mean = lm_xyz[mcp_indices, :2].mean(axis=0)
    scale = math.dist((wrist[0], wrist[1]), (float(mcp_mean[0]), float(mcp_mean[1])))
    if scale < 1e-6:
        scale = 1.0
    lm_rel[:, 0] = lm_rel[:, 0] / scale
    lm_rel[:, 1] = lm_rel[:, 1] / scale
    lm_rel[:, 2] = lm_rel[:, 2] / scale
    # clamp
    lm_rel[:, 0] = np.clip(lm_rel[:, 0], -1.0, 1.0)
    lm_rel[:, 1] = np.clip(lm_rel[:, 1], -1.0, 1.0)
    lm_rel[:, 2] = np.clip(lm_rel[:, 2], -1.0, 1.0)
    return lm_rel


def handedness_match(h: str, wanted: str) -> bool:
    if wanted == "auto":
        return True
    return h.lower() == wanted.lower()


def select_hand_by_policy(results, width: int, height: int):
    # Retourne l’index de la main retenue selon HAND_SELECT et taille bbox
    if not results.multi_hand_landmarks:
        return -1
    candidate = -1
    best_area = -1
    for i, lm in enumerate(results.multi_hand_landmarks):
        # Filtrage par main gauche/droite si demandé
        if results.multi_handedness and i < len(results.multi_handedness):
            label = results.multi_handedness[i].classification[0].label  # "Left" ou "Right"
            if not handedness_match(label, HAND_SELECT):
                continue
        # Aire bbox
        pts = []
        for p in lm.landmark:
            x = int(p.x * width)
            y = int(p.y * height)
            pts.append((x, y))
        pts = np.array(pts)
        xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
        xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
        area = max(1, (xmax - xmin)) * max(1, (ymax - ymin))
        if area > best_area:
            best_area = area
            candidate = i
    return candidate


def get_filename() -> str:
    return str(uuid.uuid4()).replace("-", "")


def render_hand_skeleton_image(lm_norm_xyz: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Génère une image carrée (img_size x img_size) contenant uniquement
    le squelette de la main à partir des landmarks normalisés lm_norm_xyz (21, 3).
    Fond noir, traits/points blancs.
    """
    # Image de fond noir
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    if lm_norm_xyz is None or lm_norm_xyz.shape[0] != 21:
        return canvas

    # On recalcule un cadrage propre pour:
    # - centrer la main dans l'image
    # - utiliser au mieux la taille de l'image (zoom constant)
    pts_2d = lm_norm_xyz[:, :2].copy()  # (21, 2) -> (x, y)
    xs = pts_2d[:, 0]
    ys = pts_2d[:, 1]

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    # Centre du nuage de points
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)

    # Rayon max (pour garder le même facteur d'échelle en x et y)
    half_range_x = 0.5 * (max_x - min_x)
    half_range_y = 0.5 * (max_y - min_y)
    radius = max(half_range_x, half_range_y)
    if radius < 1e-6:
        radius = 1.0

    # Marge pour ne pas coller aux bords
    margin = 0.1  # 10% de bordure

    # Facteur d'échelle pour remplir l'image tout en gardant la marge
    usable = 1.0 - 2.0 * margin
    scale = usable / (2.0 * radius)

    # Transformation vers [margin, 1 - margin] centré en (0.5, 0.5)
    xs_n = 0.5 + (xs - cx) * scale
    ys_n = 0.5 + (ys - cy) * scale

    # Clamp au cas où
    xs_n = np.clip(xs_n, margin, 1.0 - margin)
    ys_n = np.clip(ys_n, margin, 1.0 - margin)

    # Passage en coordonnées pixel, sans inversion verticale (même orientation que les landmarks normalisés)
    xs_pix = xs_n * (img_size - 1)
    ys_pix = ys_n * (img_size - 1)

    pts_2d_int = np.stack([xs_pix.astype(np.int32), ys_pix.astype(np.int32)], axis=1)  # (21, 2)

    # Définition des connexions de la main (équivalent MediaPipe HAND_CONNECTIONS)
    connections = [
        # Pouce
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Majeur
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Annulaire
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Auriculaire
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    # Dessin des segments
    for i0, i1 in connections:
        p0 = tuple(pts_2d_int[i0])
        p1 = tuple(pts_2d_int[i1])
        cv2.line(canvas, p0, p1, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # Dessin des points
    for p in pts_2d_int:
        cv2.circle(canvas, tuple(p), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    return canvas


def main() -> None:
    # Prépare les classes
    classes = CLASS_NAMES if CLASS_NAMES else discover_classes(DATASET_ROOT)
    if not classes:
        classes = ["ClassA", "ClassB"]

    # Prépare les dossiers
    for c in classes:
        ensure_dir(os.path.join(DATASET_ROOT, c, "images"))
        ensure_dir(os.path.join(DATASET_ROOT, c, "landmarks"))
        if SAVE_FULL:
            ensure_dir(os.path.join(DATASET_ROOT, c, "images_full"))

    active_idx = 0

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERREUR: Impossible d'ouvrir la caméra", CAMERA_INDEX)
        return

    last_save_t = 0.0
    record_continuous = False

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=max(1, min(2, MAX_HANDS)),
                        min_detection_confidence=MIN_DETECT,
                        min_tracking_confidence=MIN_TRACK) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if MIRROR:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            chosen = select_hand_by_policy(results, w, h)
            lm_img = None
            lm_img_xyz = None
            bbox = None

            if chosen >= 0 and results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[chosen]
                pts_xy = []
                pts_xyz = []
                for p in hand_lm.landmark:
                    xi = float(p.x * w)
                    yi = float(p.y * h)
                    zi = float(p.z)
                    pts_xy.append([xi, yi])
                    pts_xyz.append([xi, yi, zi])
                lm_img = np.array(pts_xy, dtype=np.float32)  # (21,2)
                lm_img_xyz = np.array(pts_xyz, dtype=np.float32)  # (21,3)
                xmin, ymin, xmax, ymax = compute_bbox_from_landmarks(lm_img, w, h, BBOX_MARGIN)
                bbox = (xmin, ymin, xmax, ymax)

                # Dessin
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                )
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Aucune main detectee", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # UI infos
            cv2.putText(frame, f"Classe: {classes[active_idx]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "A/D changer classe  | SPACE capture  | R rec (landmarks)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
            cv2.putText(frame, "Q/ESC quitter", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

            # Enregistrement continu
            now = time.time()
            do_save_continuous = record_continuous and (now - last_save_t >= (1.0 / max(1e-6, CONTINUOUS_FPS)))

            # Affichage
            cv2.imshow("Hand Dataset Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            # Gestion clavier
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('a'), ord('A')):
                active_idx = (active_idx - 1) % len(classes)
            elif key in (ord('d'), ord('D')):
                active_idx = (active_idx + 1) % len(classes)
            elif key in (ord('r'), ord('R')):
                record_continuous = not record_continuous
            elif key == 32:  # SPACE
                do_save_continuous = True

            if do_save_continuous and lm_img is not None and bbox is not None:
                last_save_t = now
                cls = classes[active_idx]
                img_dir = os.path.join(DATASET_ROOT, cls, "images")
                lm_dir = os.path.join(DATASET_ROOT, cls, "landmarks")
                full_dir = os.path.join(DATASET_ROOT, cls, "images_full")
                ensure_dir(img_dir)
                ensure_dir(lm_dir)
                if SAVE_FULL:
                    ensure_dir(full_dir)

                xmin, ymin, xmax, ymax = bbox
                crop = frame[ymin:ymax, xmin:xmax, :]
                if crop.size > 0:
                    filename = get_filename()
                    # Sauvegarde éventuelle du crop original (optionnel)
                    if SAVE_FULL:
                        full_path = os.path.join(full_dir, f"{filename}.jpg")
                        cv2.imwrite(full_path, crop)

                    # Normalisation des landmarks (pour le .npz et pour l'image squelette)
                    if lm_img_xyz is not None:
                        lm_norm = normalize_landmarks_xy_z(lm_img_xyz)
                    else:
                        lm_norm = None

                    # Image squelette de la main sur fond uni
                    if SAVE_FORMAT in ("both", "images") and lm_norm is not None:
                        skeleton_img = render_hand_skeleton_image(lm_norm, IMG_SIZE)
                        img_path = os.path.join(img_dir, f"{filename}.jpg")
                        cv2.imwrite(img_path, skeleton_img)

                    # Sauvegarde des landmarks normalisés
                    if SAVE_FORMAT in ("both", "landmarks") and lm_norm is not None:
                        lm_path = os.path.join(lm_dir, f"{filename}.npz")
                        np.savez_compressed(lm_path, landmarks=lm_norm.astype(np.float32))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


