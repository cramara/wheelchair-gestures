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

import socket
import threading

# ============================================================
# ======================= DEBUG LOGGER ========================
# ============================================================

def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}][DEBUG] {msg}")


# ============================================================
# ======================== ESP32 SETUP ========================
# ============================================================

ESP_HOST = "192.168.4.1"
ESP_PORT = 80

# Masukkan perintah sesuai keinginan untuk setiap label
COMMAND_MAP = {
    "go":    "B\n",   # maju
    "left":  "E\n",   # kiri
    "right": "A\n",   # kanan
    "stop":  "C\n",   # stop
}

class SocketCommunicator:
    def __init__(self, host, port):
        log(f"Initializing ESP32 communicator at {host}:{port}")
        self.host = host
        self.port = port
        self.lock = threading.Lock()

    def send(self, data: bytes):
        with self.lock:
            try:
                log(f"Opening connection to ESP32...")
                with socket.create_connection((self.host, self.port), timeout=2) as s:
                    log(f"Sending data: {data.decode().strip()}")
                    s.sendall(data)

                    try:
                        s.settimeout(1)
                        resp = s.recv(1024)
                        if resp:
                            log(f"ESP32 Response: {resp.decode(errors='ignore').strip()}")
                    except socket.timeout:
                        log("ESP32 response timeout (no response)")

            except Exception as e:
                log(f"ESP32 ERROR: {e}")

    def close(self):
        log("Closing communicator (no persistent connection)")
        pass


# ============================================================
# ===================== MODEL CONFIG =========================
# ============================================================

MODEL_PATH = "D:\\wheelchair-gestures\\models\\hands_landmarks_best.h5"
LABEL_MAP_PATH = "D:\\wheelchair-gestures\\models\\label_map.json"
IMG_SIZE = 224
USE_SKELETON_IMAGE = True

CAMERA_INDEX = 0
MIRROR = True

MAX_HANDS = 1
HAND_SELECT = "auto"
MIN_DETECT = 0.5
MIN_TRACK = 0.5
BBOX_MARGIN = 0.2  

CONFIDENCE_THRESHOLD = 0.5
SMOOTHING = True
SMOOTHING_WINDOW = 5

SHOW_FPS = True
SHOW_CONFIDENCE = True


# ============================================================
# ====================== HELPER FUNCTIONS ====================
# ============================================================

def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def compute_bbox_from_landmarks(lm_img_xy: np.ndarray, w: int, h: int, margin: float):
    x = lm_img_xy[:, 0]
    y = lm_img_xy[:, 1]
    xmin = int(x.min())
    xmax = int(x.max())
    ymin = int(y.min())
    ymax = int(y.max())

    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    bw, bh = (xmax - xmin), (ymax - ymin)
    bw = int(bw * (1 + margin))
    bh = int(bh * (1 + margin))

    xmin = clamp(int(cx - bw / 2), 0, w - 1)
    xmax = clamp(int(cx + bw / 2), 0, w - 1)
    ymin = clamp(int(cy - bh / 2), 0, h - 1)
    ymax = clamp(int(cy + bh / 2), 0, h - 1)

    return xmin, ymin, xmax, ymax


def normalize_landmarks_xy_z(lm_xyz):
    wrist = lm_xyz[0].copy()
    lm_rel = lm_xyz - wrist
    mcp = lm_xyz[[5, 9, 13, 17], :2].mean(axis=0)
    scale = math.dist((wrist[0], wrist[1]), (float(mcp[0]), float(mcp[1])))
    if scale < 1e-6:
        scale = 1.0
    lm_rel /= scale
    lm_rel = np.clip(lm_rel, -1, 1)
    return lm_rel


def normalize_landmark_orientation(lm):
    wrist = lm[0, :2]
    index_mcp = lm[5, :2]
    direction = index_mcp - wrist
    if direction[0] < 0:
        lm[:, 0] = -lm[:, 0]
    return lm


def select_hand_by_policy(results, width, height):
    if not results.multi_hand_landmarks:
        return -1
    best = -1
    best_area = -1
    for i, lm in enumerate(results.multi_hand_landmarks):
        pts = np.array([[p.x * width, p.y * height] for p in lm.landmark])
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        area = (xmax - xmin) * (ymax - ymin)
        if area > best_area:
            best_area = area
            best = i
    return best


def prepare_landmarks_for_prediction(lm_xyz):
    lm_norm = normalize_landmarks_xy_z(lm_xyz)
    lm_norm = normalize_landmark_orientation(lm_norm)
    return lm_norm.reshape(-1)


def render_hand_skeleton_from_bbox(lm_xy, bbox, img_size=IMG_SIZE):
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    xmin, ymin, xmax, ymax = bbox
    bw, bh = xmax - xmin, ymax - ymin
    xs = (lm_xy[:, 0] - xmin) / max(1, bw)
    ys = (lm_xy[:, 1] - ymin) / max(1, bh)
    xs = np.clip(xs, 0, 1)
    ys = np.clip(ys, 0, 1)

    pts = np.stack([
        (xs * (img_size - 1)).astype(int),
        (ys * (img_size - 1)).astype(int)
    ], axis=-1)

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    for a, b in connections:
        cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 2)

    for p in pts:
        cv2.circle(canvas, tuple(p), 3, (255, 255, 255), -1)

    return canvas


def prepare_image_for_prediction(frame, bbox, lm_xy):
    xmin, ymin, xmax, ymax = bbox
    if USE_SKELETON_IMAGE:
        img = render_hand_skeleton_from_bbox(lm_xy, bbox)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        crop = frame[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            return None
        img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return img


def detect_mode_from_model_path(path):
    name = os.path.basename(path).lower()
    if "landmarks" in name:
        return "landmarks"
    if "images" in name:
        return "images"
    return "landmarks"


def load_model_and_labels():
    log("Loading model and label map...")
    model = load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    mode = detect_mode_from_model_path(MODEL_PATH)
    log(f"Model loaded ({mode}). Labels: {label_map}")
    return model, label_map, mode


# ============================================================
# ======================= PREDICTION SMOOTHING ================
# ============================================================

class PredictionSmoother:
    def __init__(self, w=5):
        self.w = w
        self.pred = []

    def add(self, idx, conf):
        self.pred.append((idx, conf))
        if len(self.pred) > self.w:
            self.pred.pop(0)
        if len(self.pred) < self.w:
            return idx, conf

        scores = {}
        for i, c in self.pred:
            scores.setdefault(i, []).append(c)

        best = max(scores, key=lambda k: np.mean(scores[k]))
        avg = np.mean(scores[best])
        return best, float(avg)


# ============================================================
# =========================== MAIN ===========================
# ============================================================

def main():
    log("Starting system...")

    model, label_map, mode = load_model_and_labels()
    smoother = PredictionSmoother(SMOOTHING_WINDOW) if SMOOTHING else None

    comm = SocketCommunicator(ESP_HOST, ESP_PORT)
    last_sent = None

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    log("Camera initialized.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DETECT,
        min_tracking_confidence=MIN_TRACK
    ) as hands:

        fps_buf = []

        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                log("Camera frame error")
                continue

            if MIRROR:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            chosen = select_hand_by_policy(results, w, h)
            detected = False
            current_prediction = None

            if chosen >= 0:
                lm = results.multi_hand_landmarks[chosen]
                mp_draw.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

                pts_xy = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
                pts_xyz = np.array([[p.x * w, p.y * h, p.z] for p in lm.landmark], dtype=np.float32)

                xmin, ymin, xmax, ymax = compute_bbox_from_landmarks(pts_xy, w, h, BBOX_MARGIN)
                bbox = (xmin, ymin, xmax, ymax)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                if mode == "landmarks":
                    feat = prepare_landmarks_for_prediction(pts_xyz)
                    feat = np.expand_dims(feat, 0)
                else:
                    img = prepare_image_for_prediction(frame, bbox, pts_xy)
                    if img is None:
                        log("Image preparation failed")
                        continue
                    feat = np.expand_dims(img, 0)

                probs = model.predict(feat, verbose=0)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])

                if smoother:
                    idx, conf = smoother.add(idx, conf)

                if conf >= CONFIDENCE_THRESHOLD:
                    current_prediction = label_map[idx]
                    detected = True
                    log(f"Geste Detected: {current_prediction} (conf={conf:.2f})")
                else:
                    log(f"Low confidence ({conf:.2f}), ignored")

            else:
                log("No hand detected")
                if smoother:
                    smoother.pred = []

            if detected:
                if current_prediction in COMMAND_MAP:
                    if last_sent != current_prediction:
                        log(f"Gesture changed: {last_sent} -> {current_prediction}")
                        cmd = COMMAND_MAP[current_prediction].encode()
                        log(f"Sending to ESP32: {cmd}")
                        comm.send(cmd)
                        last_sent = current_prediction
                    else:
                        log(f"Gesture same ({current_prediction}), not sending")
                else:
                    log(f"Unknown gesture '{current_prediction}' (not in COMMAND_MAP)")

            # FPS LOGGING
            dt = (time.time() - t0)
            fps = 1.0 / dt
            # log(f"Frame time: {dt*1000:.1f}ms | FPS={fps:.1f}")

            if SHOW_FPS:
                fps_buf.append(fps)
                if len(fps_buf) > 30:
                    fps_buf.pop(0)
                avg = np.mean(fps_buf)
                cv2.putText(frame, f"FPS: {avg:.1f}", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Hand Gesture + ESP32", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    cap.release()
    cv2.destroyAllWindows()
    comm.close()
    log("System stopped.")


if __name__ == "__main__":
    main()
