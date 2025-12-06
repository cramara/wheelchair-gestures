#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
from typing import Tuple, Dict, Optional

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model


# Configuration
MODEL_PATH = "models/hands_landmarks.h5"  # ou "models/hands_images.h5" - sera détecté automatiquement
LABEL_MAP_PATH = "models/label_map.json"
IMG_SIZE = 224  # utilisé si le mode détecté est "images"
USE_SKELETON_IMAGE = True  # True si les images d'entraînement sont des squelettes, False si ce sont des crops réels

# Caméra
CAMERA_INDEX = 0
MIRROR = True

# MediaPipe Hands
MAX_HANDS = 1
HAND_SELECT = "auto"  # "auto" | "left" | "right"
MIN_DETECT = 0.5
MIN_TRACK = 0.5
BBOX_MARGIN = 0.2  # marge relative autour de la bbox

# Prédiction
CONFIDENCE_THRESHOLD = 0.5  # seuil de confiance minimum pour afficher la prédiction
SMOOTHING = True  # lissage des prédictions avec moyenne mobile
SMOOTHING_WINDOW = 5  # nombre de prédictions pour la moyenne mobile

# Affichage
SHOW_FPS = True
SHOW_CONFIDENCE = True


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
    bw = int(bw * (1.0 + margin))
    bh = int(bh * (1.0 + margin))
    xmin = int(cx - bw / 2)
    xmax = int(cx + bw / 2)
    ymin = int(cy - bh / 2)
    ymax = int(cy + bh / 2)
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


def handedness_match(h: str, wanted: str) -> bool:
    if wanted == "auto":
        return True
    return h.lower() == wanted.lower()


def select_hand_by_policy(results, width: int, height: int):
    if not results.multi_hand_landmarks:
        return -1
    candidate = -1
    best_area = -1
    for i, lm in enumerate(results.multi_hand_landmarks):
        if results.multi_handedness and i < len(results.multi_handedness):
            label = results.multi_handedness[i].classification[0].label
            if not handedness_match(label, HAND_SELECT):
                continue
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


def prepare_landmarks_for_prediction(lm_img_xyz: np.ndarray) -> np.ndarray:
    """Prépare les landmarks pour la prédiction (même format que l'entraînement)"""
    lm_norm = normalize_landmarks_xy_z(lm_img_xyz)
    # Normaliser l'orientation (toujours vers la droite) - doit correspondre à l'entraînement
    lm_norm = normalize_landmark_orientation(lm_norm)
    return lm_norm.reshape(-1)  # (63,)


def render_hand_skeleton_from_bbox(
    lm_img_xy: np.ndarray,
    bbox: Tuple[int, int, int, int],
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """Génère une image squelette (points + segments) sur fond noir"""
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    if lm_img_xy is None or lm_img_xy.shape[0] != 21:
        return canvas
    
    xmin, ymin, xmax, ymax = bbox
    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)
    
    xs = (lm_img_xy[:, 0] - float(xmin)) / float(bw)
    ys = (lm_img_xy[:, 1] - float(ymin)) / float(bh)
    
    margin = 0.05
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)
    xs_n = margin + xs * (1.0 - 2.0 * margin)
    ys_n = margin + ys * (1.0 - 2.0 * margin)
    
    xs_pix = xs_n * (img_size - 1)
    ys_pix = ys_n * (img_size - 1)
    pts_2d_int = np.stack([xs_pix.astype(np.int32), ys_pix.astype(np.int32)], axis=1)
    
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    
    for i0, i1 in connections:
        p0 = tuple(pts_2d_int[i0])
        p1 = tuple(pts_2d_int[i1])
        cv2.line(canvas, p0, p1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    
    for p in pts_2d_int:
        cv2.circle(canvas, tuple(p), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    
    return canvas


def prepare_image_for_prediction(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    lm_img_xy: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """Prépare l'image pour la prédiction (même format que l'entraînement)"""
    xmin, ymin, xmax, ymax = bbox
    
    if USE_SKELETON_IMAGE and lm_img_xy is not None:
        img = render_hand_skeleton_from_bbox(lm_img_xy, bbox, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        crop = frame[ymin:ymax, xmin:xmax, :]
        if crop.size == 0:
            return None
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    img = (img.astype('float32') / 255.0)
    return img


def detect_mode_from_model_path(model_path: str) -> str:
    """Détecte le mode (landmarks ou images) à partir du nom du fichier modèle"""
    filename = os.path.basename(model_path).lower()
    if "landmarks" in filename:
        return "landmarks"
    elif "images" in filename:
        return "images"
    else:
        print("AVERTISSEMENT: impossible de détecter le mode depuis le nom du fichier.")
        print("Utilisation du mode 'landmarks' par défaut.")
        return "landmarks"


def load_model_and_labels() -> Tuple[tf.keras.Model, Dict[int, str], str]:
    """Charge le modèle et le label map, retourne aussi le mode détecté"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle non trouvé: {MODEL_PATH}")
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map non trouvé: {LABEL_MAP_PATH}")
    
    mode = detect_mode_from_model_path(MODEL_PATH)
    
    print(f"Chargement du modèle: {MODEL_PATH}")
    print(f"Mode détecté: {mode}")
    model = load_model(MODEL_PATH)
    
    print(f"Chargement du label map: {LABEL_MAP_PATH}")
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map_raw = json.load(f)
    
    label_map = {int(k): v for k, v in label_map_raw.items()}
    print(f"Classes chargées: {list(label_map.values())}")
    
    return model, label_map, mode


class PredictionSmoother:
    """Lisse les prédictions avec une moyenne mobile"""
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.predictions = []
    
    def add(self, pred_idx: int, confidence: float) -> Tuple[int, float]:
        self.predictions.append((pred_idx, confidence))
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
        
        if len(self.predictions) < self.window_size:
            return pred_idx, confidence
        
        # Calcul de la moyenne pondérée par la confiance
        class_scores = {}
        for p_idx, conf in self.predictions:
            if p_idx not in class_scores:
                class_scores[p_idx] = []
            class_scores[p_idx].append(conf)
        
        best_class = max(class_scores.keys(), key=lambda k: np.mean(class_scores[k]))
        avg_confidence = np.mean(class_scores[best_class])
        
        return best_class, avg_confidence


def main() -> None:
    print("=" * 60)
    print("PRÉDICTION DE GESTES EN TEMPS RÉEL")
    print("=" * 60)
    print(f"Modèle: {MODEL_PATH}")
    print()
    
    try:
        model, label_map, mode = load_model_and_labels()
    except Exception as e:
        print(f"ERREUR lors du chargement: {e}")
        return
    
    print(f"Mode de prédiction: {mode}")
    if mode == "images":
        print(f"Format d'image: {'squelette' if USE_SKELETON_IMAGE else 'crop réel'}")
    print()
    
    smoother = PredictionSmoother(SMOOTHING_WINDOW) if SMOOTHING else None
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERREUR: Impossible d'ouvrir la caméra {CAMERA_INDEX}")
        return
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Initialisation de MediaPipe Hands...")
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max(1, min(2, MAX_HANDS)),
        min_detection_confidence=MIN_DETECT,
        min_tracking_confidence=MIN_TRACK
    ) as hands:
        print("Démarrage de la capture vidéo...")
        print("Appuyez sur 'Q' ou 'ESC' pour quitter")
        print()
        
        fps_times = []
        current_prediction = None
        current_confidence = 0.0
        
        while True:
            t_start = time.time()
            ok, frame = cap.read()
            if not ok:
                continue
            
            if MIRROR:
                frame = cv2.flip(frame, 1)
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            chosen = select_hand_by_policy(results, w, h)
            prediction_made = False
            
            if chosen >= 0 and results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[chosen]
                
                # Dessin des landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                )
                
                # Extraction des landmarks
                pts_xyz = []
                pts_xy = []
                for p in hand_lm.landmark:
                    xi = float(p.x * w)
                    yi = float(p.y * h)
                    zi = float(p.z)
                    pts_xy.append([xi, yi])
                    pts_xyz.append([xi, yi, zi])
                
                lm_img_xyz = np.array(pts_xyz, dtype=np.float32)  # (21, 3)
                lm_img_xy = np.array(pts_xy, dtype=np.float32)  # (21, 2)
                
                xmin, ymin, xmax, ymax = compute_bbox_from_landmarks(lm_img_xy, w, h, BBOX_MARGIN)
                bbox = (xmin, ymin, xmax, ymax)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Prédiction
                try:
                    if mode == "landmarks":
                        features = prepare_landmarks_for_prediction(lm_img_xyz)
                        features = np.expand_dims(features, axis=0)  # (1, 63)
                    else:
                        img = prepare_image_for_prediction(frame, bbox, lm_img_xy)
                        if img is not None:
                            features = np.expand_dims(img, axis=0)  # (1, IMG_SIZE, IMG_SIZE, 3)
                        else:
                            features = None
                    
                    if features is not None:
                        probs = model.predict(features, verbose=0)[0]
                        pred_idx = int(np.argmax(probs))
                        confidence = float(probs[pred_idx])
                        
                        if smoother:
                            pred_idx, confidence = smoother.add(pred_idx, confidence)
                        
                        if confidence >= CONFIDENCE_THRESHOLD:
                            current_prediction = label_map[pred_idx]
                            current_confidence = confidence
                            prediction_made = True
                except Exception as e:
                    print(f"ERREUR lors de la prédiction: {e}")
            else:
                if not prediction_made:
                    current_prediction = None
                    current_confidence = 0.0
                    if smoother:
                        smoother.predictions.clear()
            
            # Affichage
            if current_prediction:
                text = f"Geste: {current_prediction.upper()}"
                color = (0, 255, 0)
                cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                if SHOW_CONFIDENCE:
                    conf_text = f"Confiance: {current_confidence:.2%}"
                    cv2.putText(frame, conf_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(frame, "Aucune main detectee", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if SHOW_FPS:
                fps = 1.0 / (time.time() - t_start)
                fps_times.append(fps)
                if len(fps_times) > 30:
                    fps_times.pop(0)
                avg_fps = np.mean(fps_times)
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Reconnaissance de Gestes", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Arrêt de la capture vidéo.")


if __name__ == "__main__":
    main()

