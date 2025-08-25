# -*- coding: utf-8 -*-
"""
Calisthenics AI — Enhanced with Smarter Posture Detection (Idle/Ready/Performing)
-----------------------------------------------------------------------------
Enhancements:
  • Added "ready" state: Detects starting posture before entering "performing" (where detection/judging happens).
  • State machine: IDLE → READY → PERFORMING → REST.
  • Auto-mode detection: If MODE=None, classifies skill based on pose.
  • Debounced gates: Average over frames for stability.
  • More feedback: e.g., "Get ready" messages.
  • Fixed eval issues: Replaced eval with direct function calls and argument mapping.

Other notes remain the same.
"""

import os
import cv2
import time
import math
import numpy as np
from collections import deque

# ========= EDIT THESE =========
SOURCE = 0  # 0 for webcam, or "/path/to/video.mp4", or "rtsp://user:pass@ip:554/..."
MODE = "Front Lever"  # One of: "Pull-up", "Front Lever", "Planche", "Handstand", "L-sit" (or None for auto-detect)
OUTPUT_FILE = "analyzed_output.mp4"
MIRROR_WEBCAM = True  # flip webcam input horizontally
MAX_BAD_SHOTS = 3     # auto screenshots on incorrect posture (per run)
DEBOUNCE_FRAMES = 5   # frames to average for stable state transitions
# ==============================

# --- States ---
STATES = {
    "IDLE": "Idle — waiting for \"%s\" shape…",
    "READY": "Ready — start the \"%s\" now!",
    "PERFORMING": "Performing…",
    "REST": "Rest — good rep, reset for next."
}

# --- UI strings (extended) ---
TXT = {
    "app_title": "Calisthenics AI (English UI + Smart Posture)",
    "mode": "Mode",
    "fps": "FPS",
    "reps": "Reps",
    "help_title": "Hotkeys",
    "help_lines": [
        "Q: Quit",
        "H: Toggle help",
        "G: Toggle grid/guides",
        "A: Toggle angle labels",
        "S: Save screenshot",
        "C: Calibrate (T-pose)",
    ],
    "calibrated": "Calibrated: shoulder width = %.1f px",
    "analyzing": "Analyzing…",
    "clean_rep": "Clean ✅",
    "pull_higher": "Pull higher (elbow ≤ 60°)",
    "reduce_kip": "Reduce swing — lock hips/knees",
    "keep_level": "Keep bodyline level — tuck pelvis, squeeze glutes",
    "lock_elbows": "Lock elbows",
    "insufficient_lean": "Planche: not enough lean (shift shoulders forward)",
    "banana": "Banana handstand — close ribs, posterior pelvic tilt",
    "badshot_saved": "Saved screenshot (%d/%d): ",
    "auto_mode": "Auto-detected: %s",
}

# --- Mode to function and argument mapping ---
MODE_TO_FUNC = {
    "Pull-up": {
        "name": "pullup",
        "ready_func": lambda elbow_avg, hip_avg, body_avg: gate_ready_pullup(elbow_avg, hip_avg, body_avg),
        "gate_func": lambda sL, sR, wL, wR, H: gate_pullup(sL, sR, wL, wR, H),
        "judge_func": lambda elbow_hist, hip_hist: judge_pullup(elbow_hist, hip_hist)
    },
    "Front Lever": {
        "name": "frontlever",
        "ready_func": lambda body_avg, elbow_avg: gate_ready_frontlever(body_avg, elbow_avg),
        "gate_func": lambda body: gate_frontlever(body),
        "judge_func": lambda body_hist, elbow_hist: judge_frontlever(body_hist, elbow_hist)
    },
    "Planche": {
        "name": "planche",
        "ready_func": lambda lean_avg: gate_ready_planche(lean_avg),
        "gate_func": lambda sL, wL, hC: gate_planche(sL, wL, hC),
        "judge_func": lambda sL, wL, hC: judge_planche(sL, wL, hC)
    },
    "Handstand": {
        "name": "handstand",
        "ready_func": lambda body_avg: gate_ready_handstand(body_avg),
        "gate_func": lambda body: gate_handstand(body),
        "judge_func": lambda body: judge_handstand(body)
    },
    "L-sit": {
        "name": "lsit",
        "ready_func": lambda hip_avg: gate_ready_lsit(hip_avg),
        "gate_func": lambda hL: gate_lsit(hL),
        "judge_func": lambda hL: TXT["clean_rep"] if abs(90 - hL) <= 12 else "Lift legs; depress shoulders"
    }
}

# --- Simple text draw (OpenCV) ---
def draw_text(img, text, pos, size=0.7, color=(255,255,255), thick=2):
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)
    return img

# --- MediaPipe Pose ---
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
except Exception as e:
    raise SystemExit("Install mediapipe first: pip install mediapipe==0.10.14" + str(e))

POSE_LMS = mp_pose.PoseLandmark

# --- Geometry helpers ---
def to_xy(lms, idx, shape):
    h, w = shape[:2]
    lm = lms[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32), lm.visibility

def angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = np.dot(ba, bc) / denom
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return math.degrees(math.acos(cosang))

# --- Drawing helpers ---
def line(img, p, q, col=(0, 255, 0), thick=3):
    p = tuple(p.astype(int)); q = tuple(q.astype(int))
    cv2.line(img, p, q, col, thick, cv2.LINE_AA)

# --- Panels ---
def panel(img, x, y, w, h, alpha=0.4, color=(0, 0, 0)):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# --- Angle labels ---
def put_angle_text(img, text, pos):
    return draw_text(img, text, (int(pos[0]), int(pos[1])), size=0.7, color=(255,255,255), thick=2)

# --- Kinematics helpers ---
def elbow_angle(lms, shape, side='LEFT'):
    S = POSE_LMS.LEFT_SHOULDER if side=='LEFT' else POSE_LMS.RIGHT_SHOULDER
    E = POSE_LMS.LEFT_ELBOW if side=='LEFT' else POSE_LMS.RIGHT_ELBOW
    W = POSE_LMS.LEFT_WRIST if side=='LEFT' else POSE_LMS.RIGHT_WRIST
    s,_ = to_xy(lms, S, shape); e,_ = to_xy(lms, E, shape); w,_ = to_xy(lms, W, shape)
    return angle(s,e,w), e

def hip_angle(lms, shape, side='LEFT'):
    S = POSE_LMS.LEFT_SHOULDER if side=='LEFT' else POSE_LMS.RIGHT_SHOULDER
    H = POSE_LMS.LEFT_HIP if side=='LEFT' else POSE_LMS.RIGHT_HIP
    K = POSE_LMS.LEFT_KNEE if side=='LEFT' else POSE_LMS.RIGHT_KNEE
    s,_ = to_xy(lms, S, shape); h,_ = to_xy(lms, H, shape); k,_ = to_xy(lms, K, shape)
    return angle(s,h,k), h

def bodyline_angle(lms, shape, side='LEFT'):
    S = POSE_LMS.LEFT_SHOULDER if side=='LEFT' else POSE_LMS.RIGHT_SHOULDER
    H = POSE_LMS.LEFT_HIP if side=='LEFT' else POSE_LMS.RIGHT_HIP
    A = POSE_LMS.LEFT_ANKLE if side=='LEFT' else POSE_LMS.RIGHT_ANKLE
    s,_ = to_xy(lms, S, shape); h,_ = to_xy(lms, H, shape); a,_ = to_xy(lms, A, shape)
    return angle(s,h,a), h

def shoulder_wrist(lms, shape, side='LEFT'):
    S = POSE_LMS.LEFT_SHOULDER if side=='LEFT' else POSE_LMS.RIGHT_SHOULDER
    W = POSE_LMS.LEFT_WRIST if side=='LEFT' else POSE_LMS.RIGHT_WRIST
    s,_ = to_xy(lms, S, shape); w,_ = to_xy(lms, W, shape)
    return s, w

# --- Helpers ---
def tail(seq, n):
    return list(seq)[-n:]

# --- Judges (simple, extensible) ---
def judge_pullup(elbow_deg_hist, hip_deg_hist):
    if len(elbow_deg_hist) < 10:
        return TXT["analyzing"]
    e = np.array(tail(elbow_deg_hist, 15), dtype=np.float32)
    h = np.array(tail(hip_deg_hist, 15), dtype=np.float32)
    depth_ok = (np.min(e) <= 60)
    kipping = (np.std(h) > 12)
    notes = []
    if not depth_ok:
        notes.append(TXT["pull_higher"])
    if kipping:
        notes.append(TXT["reduce_kip"])
    return TXT["clean_rep"] if not notes else "  • ".join(notes)

def judge_frontlever(bodyline_hist, elbow_hist):
    if len(bodyline_hist) < 10:
        return TXT["analyzing"]
    b = np.array(tail(bodyline_hist, 20), dtype=np.float32)
    e = np.array(tail(elbow_hist, 20), dtype=np.float32)
    flat = (np.mean(np.abs(180 - b)) <= 12)
    straight = (np.mean(e) >= 160)  # slightly friendlier lock
    notes = []
    if not flat:
        notes.append(TXT["keep_level"])
    if not straight:
        notes.append(TXT["lock_elbows"])
    return TXT["clean_rep"] if not notes else "  • ".join(notes)

def judge_planche(shoulder_xy, wrist_xy, hip_xy):
    sh = np.linalg.norm(shoulder_xy - hip_xy) + 1e-6
    sw = abs(wrist_xy[0] - shoulder_xy[0])
    ratio = sw / sh
    return TXT["clean_rep"] if ratio >= 0.9 else TXT["insufficient_lean"] + f" (ratio {ratio:.2f})"

def judge_handstand(trunk_deg):
    dev = abs(180 - trunk_deg)
    return TXT["clean_rep"] if dev <= 8 else TXT["banana"] + f" (dev {dev:.1f}°)"

# --- Idle gates per mode (basic geometric preconditions) ---
def gate_pullup(sL, sR, wL, wR, img_h):
    shoulder_y = (sL[1] + sR[1]) * 0.5
    wrist_y = (wL[1] + wR[1]) * 0.5
    above = (shoulder_y - wrist_y) > (0.03 * img_h)  # wrist clearly above shoulders
    return above

def gate_frontlever(body_deg):
    # near-horizontal bodyline within ~35° window
    return abs(180 - body_deg) < 35

def gate_planche(sL, wL, hC):
    # require noticeable lean before judging
    sh = np.linalg.norm(sL - hC) + 1e-6
    sw = abs(wL[0] - sL[0])
    return (sw / sh) > 0.5

def gate_handstand(body_deg):
    # near-vertical within ~35° window
    return abs(180 - body_deg) > 145  # i.e., body close to vertical

def gate_lsit(hip_deg):
    return abs(90 - hip_deg) < 35

# --- Enhanced Idle/Ready gates (with averaging) ---
def avg_over_hist(hist, n=DEBOUNCE_FRAMES):
    if len(hist) < n:
        return None
    return np.mean(tail(hist, n))

def gate_ready_pullup(elbow_avg, hip_avg, body_avg):
    # Ready: Hanging straight (elbows ~180°, hips straight ~180°, body vertical ~180°)
    return abs(180 - elbow_avg) < 15 and abs(180 - hip_avg) < 15 and abs(180 - body_avg) < 20

def gate_ready_frontlever(body_avg, elbow_avg):
    # Ready: Inverted hang or tuck position (body near horizontal, elbows locked)
    return abs(180 - body_avg) < 45 and elbow_avg > 150

def gate_ready_planche(lean_ratio_avg):
    # Ready: Basic support hold (some lean, but not full)
    return 0.3 < lean_ratio_avg < 0.6

def gate_ready_handstand(body_avg):
    # Ready: Approaching vertical (body near 180°)
    return abs(180 - body_avg) < 30

def gate_ready_lsit(hip_avg):
    # Ready: Seated with legs partially raised
    return abs(90 - hip_avg) < 45

# --- Auto-detect mode based on pose ---
def detect_mode(body_avg, elbow_avg, hip_avg, lean_ratio):
    if abs(180 - body_avg) < 20 and elbow_avg < 90:  # Vertical, bent arms → Pull-up
        return "Pull-up"
    elif abs(180 - body_avg) < 35 and elbow_avg > 150:  # Horizontal, straight arms → Front Lever
        return "Front Lever"
    elif lean_ratio > 0.4:  # Forward lean → Planche
        return "Planche"
    elif abs(180 - body_avg) > 145:  # Vertical inverted → Handstand
        return "Handstand"
    elif abs(90 - hip_avg) < 35:  # 90° hips → L-sit
        return "L-sit"
    return None  # Unknown, stay idle

# --- Capture helpers ---
def is_webcam_source(src):
    return isinstance(src, int) or (isinstance(src, str) and src.isdigit())

def open_capture(source):
    src = int(source) if isinstance(source, str) and source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 1 or native_fps > 240:
        native_fps = None
    return cap, native_fps

# --- Main (with enhancements) ---
def main():
    os.makedirs('screenshots', exist_ok=True)

    cap, native_fps = open_capture(SOURCE)
    wait_ms = int(1000 / (native_fps or 30))

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*"mp4v"), native_fps or 30, (W, H))

    show_help = True
    show_grid = True
    show_angles = True
    shoulder_width_px = None

    reps = 0
    current_state = "IDLE"
    current_mode = MODE  # Start with fixed, or None for auto

    bad_shot_count = 0
    last_reason = None
    last_shot_ms = 0
    COOLDOWN_MS = 1200

    elbow_hist = deque(maxlen=160)
    hip_hist = deque(maxlen=160)
    body_hist = deque(maxlen=160)
    lean_hist = deque(maxlen=160)  # New for planche lean ratio

    fps = 0.0
    t0 = time.time()
    n_frames = 0

    use_mirror = MIRROR_WEBCAM and is_webcam_source(SOURCE)

    state = 'top'  # For pull-up rep counting

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            img = cv2.flip(frame, 1) if use_mirror else frame
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            verdict = TXT["analyzing"]
            active = False  # Now tied to PERFORMING state

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark

                # skeleton overlay
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=res.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )

                # angles/lines
                eL, eLp = elbow_angle(lms, img.shape, 'LEFT')
                eR, eRp = elbow_angle(lms, img.shape, 'RIGHT')
                hL, hLp = hip_angle(lms, img.shape, 'LEFT')
                body, hipP = bodyline_angle(lms, img.shape, 'LEFT')
                sL, wL = shoulder_wrist(lms, img.shape, 'LEFT')
                sR, wR = shoulder_wrist(lms, img.shape, 'RIGHT')
                hC = to_xy(lms, POSE_LMS.LEFT_HIP, img.shape)[0]
                lean_ratio = abs(wL[0] - sL[0]) / (np.linalg.norm(sL - hC) + 1e-6)

                if shoulder_width_px is None:
                    shoulder_width_px = float(np.linalg.norm(sL - sR))

                # history
                elbow_hist.append((eL + eR) / 2.0)
                hip_hist.append(hL)
                body_hist.append(body)
                lean_hist.append(lean_ratio)

                # Averages for stability
                elbow_avg = avg_over_hist(elbow_hist)
                hip_avg = avg_over_hist(hip_hist)
                body_avg = avg_over_hist(body_hist)
                lean_avg = avg_over_hist(lean_hist)

                if elbow_avg is None:  # Not enough history
                    current_state = "IDLE"
                    continue

                # Auto-detect mode if not set
                if current_mode is None:
                    detected = detect_mode(body_avg, elbow_avg, hip_avg, lean_avg)
                    if detected:
                        current_mode = detected
                        verdict = TXT["auto_mode"] % current_mode

                if current_mode and current_mode in MODE_TO_FUNC:
                    # Get mode-specific functions
                    mode_info = MODE_TO_FUNC[current_mode]
                    ready_func = mode_info["ready_func"]
                    gate_func = mode_info["gate_func"]
                    judge_func = mode_info["judge_func"]

                    # Argument mapping
                    args_map = {
                        "elbow_avg": elbow_avg,
                        "hip_avg": hip_avg,
                        "body_avg": body_avg,
                        "lean_avg": lean_avg,
                        "sL": sL,
                        "sR": sR,
                        "wL": wL,
                        "wR": wR,
                        "H": H,
                        "body": body,
                        "hL": hL,
                        "hC": hC
                    }

                    # State transitions
                    if current_state == "IDLE":
                        if current_mode == "Pull-up":
                            if ready_func(elbow_avg, hip_avg, body_avg):
                                current_state = "READY"
                        elif current_mode == "Front Lever":
                            if ready_func(body_avg, elbow_avg):
                                current_state = "READY"
                        elif current_mode == "Planche":
                            if ready_func(lean_avg):
                                current_state = "READY"
                        elif current_mode == "Handstand":
                            if ready_func(body_avg):
                                current_state = "READY"
                        elif current_mode == "L-sit":
                            if ready_func(hip_avg):
                                current_state = "READY"
                    elif current_state == "READY":
                        performing = False
                        if current_mode == "Pull-up":
                            performing = gate_func(sL, sR, wL, wR, H)
                        elif current_mode in ["Front Lever", "Handstand"]:
                            performing = gate_func(body)
                        elif current_mode == "Planche":
                            performing = gate_func(sL, wL, hC)
                        elif current_mode == "L-sit":
                            performing = gate_func(hL)
                        if performing:
                            current_state = "PERFORMING"
                        else:
                            still_ready = False
                            if current_mode == "Pull-up":
                                still_ready = ready_func(elbow_avg, hip_avg, body_avg)
                            elif current_mode == "Front Lever":
                                still_ready = ready_func(body_avg, elbow_avg)
                            elif current_mode == "Planche":
                                still_ready = ready_func(lean_avg)
                            elif current_mode == "Handstand":
                                still_ready = ready_func(body_avg)
                            elif current_mode == "L-sit":
                                still_ready = ready_func(hip_avg)
                            if not still_ready:
                                current_state = "IDLE"
                    elif current_state == "PERFORMING":
                        active = True
                        if current_mode == "Pull-up":
                            verdict = judge_func(elbow_hist, hip_hist)
                            if elbow_avg < 80 and state == 'top':
                                state = 'bottom'
                            if elbow_avg > 150 and state == 'bottom':
                                state = 'top'
                                reps += 1
                                current_state = "REST"
                        elif current_mode == "Front Lever":
                            verdict = judge_func(body_hist, elbow_hist)
                        elif current_mode == "Planche":
                            verdict = judge_func(sL, wL, hC)
                        elif current_mode == "Handstand":
                            verdict = judge_func(body)
                        elif current_mode == "L-sit":
                            verdict = judge_func(hL)
                        if verdict == TXT["clean_rep"]:
                            current_state = "REST"
                    elif current_state == "REST":
                        is_performing = False
                        if current_mode == "Pull-up":
                            is_performing = gate_func(sL, sR, wL, wR, H)
                        elif current_mode in ["Front Lever", "Handstand"]:
                            is_performing = gate_func(body)
                        elif current_mode == "Planche":
                            is_performing = gate_func(sL, wL, hC)
                        elif current_mode == "L-sit":
                            is_performing = gate_func(hL)
                        if not is_performing:
                            current_state = "IDLE"

                # angle labels, guides, helper lines
                if show_angles:
                    put_angle_text(img, f"Elbow {eL:0.0f}°", eLp + np.array([10, -10]))
                    put_angle_text(img, f"Hip {hL:0.0f}°", hLp + np.array([10, -10]))
                    put_angle_text(img, f"Body {body:0.0f}°", hipP + np.array([10, -10]))
                if show_grid:
                    cv2.line(img, (0, H//2), (W, H//2), (50,50,50), 1, cv2.LINE_AA)
                    for k in [W//3, 2*W//3]:
                        cv2.line(img, (k, 0), (k, H), (50,50,50), 1, cv2.LINE_AA)
                aL = to_xy(lms, POSE_LMS.LEFT_ANKLE, img.shape)[0]
                line(img, sL, hLp, (0,255,0), 3)
                line(img, hLp, aL, (0,255,0), 3)
                line(img, sL, wL, (0,200,255), 2)

            # FPS meter
            n_frames += 1
            if n_frames >= 10:
                t1 = time.time()
                fps = 10.0 / (t1 - t0 + 1e-6)
                t0 = t1
                n_frames = 0

            # Top info panel (add state)
            img = panel(img, 10, 10, 500, 140, alpha=0.35)
            draw_text(img, f"{TXT['mode']}: {current_mode or 'Auto-detecting'}", (20, 30), 0.8)
            draw_text(img, f"State: {current_state}", (20, 50), 0.7)
            draw_text(img, f"{TXT['fps']}: {fps:0.1f}", (20, 80), 0.7)
            draw_text(img, f"{TXT['reps']}: {reps}", (20, 110), 0.7)
            draw_text(img, f"Shots: {bad_shot_count}/{MAX_BAD_SHOTS}", (210, 110), 0.7, (200,220,255), 2)
            if shoulder_width_px is not None:
                draw_text(img, TXT["calibrated"] % (shoulder_width_px,), (220, 80), 0.6, (200,200,200), 1)

            # Bottom verdict panel
            img = panel(img, 10, H-70, W-20, 60, alpha=0.35)
            status_txt = STATES[current_state] % current_mode if "%s" in STATES[current_state] else STATES[current_state]
            if res.pose_landmarks and current_state == "PERFORMING":
                draw_text(img, f"{status_txt} {verdict}", (25, H-35), 0.9, (255,255,255), 2)
            else:
                draw_text(img, status_txt, (25, H-35), 0.9, (220,220,220), 2)

            # Auto screenshot only when PERFORMING & incorrect
            now_ms = int(time.time()*1000)
            is_bad = (res.pose_landmarks is not None) and current_state == "PERFORMING" and (verdict != TXT["clean_rep"]) and (verdict != TXT["analyzing"])
            if is_bad and bad_shot_count < MAX_BAD_SHOTS:
                should_save = (verdict != last_reason) or (now_ms - last_shot_ms > COOLDOWN_MS)
                if should_save:
                    snap = img.copy()
                    snap = panel(snap, 10, 10, min(900, W-20), 50, alpha=0.5)
                    snap = draw_text(snap, "Incorrect posture detected: " + verdict, (20, 45), 0.8, (255,255,255), 2)
                    fn = os.path.join('screenshots', f"bad_{int(time.time())}.png")
                    cv2.imwrite(fn, snap)
                    bad_shot_count += 1
                    last_reason = verdict
                    last_shot_ms = now_ms
                    print(TXT["badshot_saved"] % (bad_shot_count, MAX_BAD_SHOTS), verdict, "->", fn)

            # Write & show
            out.write(img)
            cv2.imshow(TXT['app_title'], img)

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord('g'):
                show_grid = not show_grid
            elif key == ord('a'):
                show_angles = not show_angles
            elif key == ord('s'):
                fn = os.path.join('screenshots', f"snap_{int(time.time())}.png")
                cv2.imwrite(fn, img)
                print("Saved:", fn)
            elif key == ord('c'):
                if res and res.pose_landmarks:
                    lms = res.pose_landmarks.landmark
                    sL,_ = to_xy(lms, POSE_LMS.LEFT_SHOULDER, img.shape)
                    sR,_ = to_xy(lms, POSE_LMS.RIGHT_SHOULDER, img.shape)
                    shoulder_width_px = float(np.linalg.norm(sL - sR))
                    print("Calibration -> shoulder width:", shoulder_width_px, "px")

    cap.release(); out.release(); cv2.destroyAllWindows()
    print(f"Saved analyzed video -> {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
