"""
Walk Direction Detector — Frontal vs Sagittal
Requires: mediapipe>=0.10, opencv-python, numpy

    pip install mediapipe opencv-python numpy

Run:
    python walk_direction_detector.py            # webcam
    python walk_direction_detector.py --video x.mp4
    python walk_direction_detector.py --image x.jpg
    python walk_direction_detector.py --video x.mp4 --output out.mp4
"""

import cv2, numpy as np, argparse, sys, time, urllib.request, os, tempfile
from dataclasses import dataclass

# ── mediapipe 0.10+ (tasks API) ──────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as _mpy
from mediapipe.tasks.python import vision as _vis

print(f"[INFO] mediapipe {mp.__version__}")

# ── Landmark indices ──────────────────────────────────────────────────────────
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP,      R_HIP      = 23, 24
L_ELBOW,    R_ELBOW    = 13, 14
L_KNEE,     R_KNEE     = 25, 26
NOSE                   = 0
L_EAR,      R_EAR      = 7,  8

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

BONES = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
]

# ── Result ────────────────────────────────────────────────────────────────────
@dataclass
class Result:
    label: str        # Frontal | Sagittal | Oblique
    confidence: float
    score: float      # raw frontal score 0→1

COLORS = {"Frontal":(0,220,120), "Sagittal":(30,140,255), "Oblique":(255,180,50)}

# ── Classification ────────────────────────────────────────────────────────────
def classify(lms, W, H) -> Result:
    def xy(i):
        l = lms[i]
        v = l.visibility if l.visibility is not None else 1.0
        return l.x*W, l.y*H, v

    lsx,lsy,lsv = xy(L_SHOULDER); rsx,rsy,rsv = xy(R_SHOULDER)
    lhx,lhy,lhv = xy(L_HIP);      rhx,rhy,rhv = xy(R_HIP)
    lex,ley,lev = xy(L_ELBOW);    rex,rey,rev = xy(R_ELBOW)
    lkx,lky,lkv = xy(L_KNEE);     rkx,rky,rkv = xy(R_KNEE)
    nx, ny,  _  = xy(NOSE)
    _,  _,  lev2= xy(L_EAR);      _,  _,  rev2= xy(R_EAR)

    bw = max(lsx,rsx,lhx,rhx) - min(lsx,rsx,lhx,rhx) + 1e-6

    f1 = abs(rsx - lsx) / bw                                      # shoulder width
    f2 = 1.0 - abs(np.mean([lsv,lhv,lev,lkv]) - np.mean([rsv,rhv,rev,rkv]))  # symmetry
    f3 = float(lev2 < 0.35 and rev2 < 0.35)                       # both ears hidden
    f4 = abs(rhx - lhx) / bw                                      # hip width
    f5 = max(0.0, 1.0 - abs(nx - (lsx+rsx)/2) / bw * 2)          # nose centered

    score = 0.35*f1 + 0.30*f2 + 0.15*f3 + 0.10*f4 + 0.10*f5

    if   score >= 0.62: label, conf = "Frontal",  score
    elif score <= 0.38: label, conf = "Sagittal", 1.0 - score
    else:               label, conf = "Oblique",  1.0 - abs(score-0.5)*2

    return Result(label, round(conf,3), round(score,3))


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_skeleton(frame, lms, W, H):
    pts = [(int(l.x*W), int(l.y*H)) for l in lms]
    for a,b in BONES:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0,255,220), 2, cv2.LINE_AA)
    for x,y in pts:
        cv2.circle(frame, (x,y), 5, (255,255,255), -1)
        cv2.circle(frame, (x,y), 5, (0,180,255),   1)

def draw_hud(frame, r: Result, fps=0.0):
    H, W = frame.shape[:2]
    c = COLORS.get(r.label, (200,200,200))

    # dark bar
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (W,68), (8,8,8), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    # label
    cv2.putText(frame, r.label, (14,48),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, c, 2, cv2.LINE_AA)

    # confidence bar
    bx,by,bw,bh = W-205,16,180,18
    cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (50,50,50), -1)
    cv2.rectangle(frame, (bx,by), (bx+int(bw*r.confidence),by+bh), c, -1)
    cv2.putText(frame, f"{r.confidence*100:.0f}%", (bx,by+bh+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, c, 1, cv2.LINE_AA)

    if fps > 0:
        cv2.putText(frame, f"FPS {fps:.1f}", (W-88,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130,130,130), 1)

    cv2.putText(frame, f"frontal_score={r.score:.3f}", (14,H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (110,110,110), 1)
    return frame


def landmarks_payload(lms):
    points = {}
    for index, landmark in enumerate(lms):
        name = LANDMARK_NAMES[index] if index < len(LANDMARK_NAMES) else f"landmark_{index}"
        visibility = landmark.visibility if landmark.visibility is not None else 1.0
        points[name] = {
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(getattr(landmark, "z", 0.0)),
            "visibility": float(visibility),
        }
    return {
        "points": points,
        "bones": [
            [LANDMARK_NAMES[a], LANDMARK_NAMES[b]]
            for a, b in BONES
            if a < len(LANDMARK_NAMES) and b < len(LANDMARK_NAMES)
        ],
    }


# ── Model download ────────────────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

def get_model():
    # 1. current directory
    local = "pose_landmarker_lite.task"
    if os.path.exists(local):
        return os.path.abspath(local)
    # 2. temp dir
    tmp = os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task")
    if os.path.exists(tmp):
        return tmp
    # 3. download
    print("[INFO] Downloading pose model (~5 MB) …")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    print(f"[INFO] Saved to {tmp}")
    return tmp


# ── Detector ──────────────────────────────────────────────────────────────────
class Detector:
    def __init__(self, conf=0.5, video_mode=True):
        model = get_model()
        base  = _mpy.BaseOptions(model_asset_path=model)
        mode  = _vis.RunningMode.VIDEO if video_mode else _vis.RunningMode.IMAGE
        opts  = _vis.PoseLandmarkerOptions(
            base_options=base,
            running_mode=mode,
            min_pose_detection_confidence=conf,
            min_tracking_confidence=conf,
        )
        self._lm        = _vis.PoseLandmarker.create_from_options(opts)
        self._video     = video_mode
        self._ts        = 0

    def process(self, frame):
        H, W = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._video:
            self._ts += 33
            det = self._lm.detect_for_video(img, self._ts)
        else:
            det = self._lm.detect(img)

        if not det.pose_landmarks:
            return frame, None

        lms = det.pose_landmarks[0]
        draw_skeleton(frame, lms, W, H)
        r = classify(lms, W, H)
        return frame, r

    def process_with_landmarks(self, frame, draw=False):
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._video:
            self._ts += 33
            det = self._lm.detect_for_video(img, self._ts)
        else:
            det = self._lm.detect(img)

        if not det.pose_landmarks:
            return frame, None, None

        lms = det.pose_landmarks[0]
        if draw:
            draw_skeleton(frame, lms, W, H)
        r = classify(lms, W, H)
        return frame, r, landmarks_payload(lms)

    def close(self):
        self._lm.close()


# ── Video / webcam loop ───────────────────────────────────────────────────────
def run_video(source, conf, output_path):
    det = Detector(conf, video_mode=True)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open: {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = (cv2.VideoWriter(output_path,
              cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W,H))
              if output_path else None)

    prev = time.time()
    print("[INFO] Press Q or ESC to quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame, r = det.process(frame)
        now = time.time(); fps = 1/(now-prev+1e-9); prev = now

        if r:
            frame = draw_hud(frame, r, fps)
            print(f"  {r.label:<10}  conf={r.confidence:.2f}  score={r.score:.3f}")
        else:
            cv2.putText(frame, "No person detected", (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80,80,255), 2)

        if writer: writer.write(frame)
        cv2.imshow("Walk Direction Detector", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    det.close()


# ── Single image ──────────────────────────────────────────────────────────────
def run_image(path, conf):
    det   = Detector(conf, video_mode=False)
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"[ERROR] Cannot load: {path}")

    frame, r = det.process(frame)
    if r:
        frame = draw_hud(frame, r)
        print(f"\nLabel      : {r.label}")
        print(f"Confidence : {r.confidence:.2%}")
        print(f"Score      : {r.score}")
    else:
        print("[WARN] No person detected.")

    out = path.rsplit(".",1)[0] + "_detected.jpg"
    cv2.imwrite(out, frame)
    print(f"Saved → {out}")
    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    det.close()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    g  = ap.add_mutually_exclusive_group()
    g.add_argument("--video", help="Video file (omit = webcam)")
    g.add_argument("--image", help="Single image file")
    ap.add_argument("--output", default=None, help="Save annotated video here")
    ap.add_argument("--conf",   type=float, default=0.5)
    args = ap.parse_args()

    if args.image:
        run_image(args.image, args.conf)
    elif args.video:
        run_video(args.video, args.conf, args.output)
    else:
        print("[INFO] No source — opening webcam (device 0)")
        run_video(0, args.conf, args.output)

if __name__ == "__main__":
    main()
