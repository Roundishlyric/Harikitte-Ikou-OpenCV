import cv2
import mediapipe as mp
import numpy as np
import pygame

# ---------- CONFIG ----------
DEFAULT_IMG = "assets/Spec.jpg"
OPEN_IMG = "assets/kita.jpg"
MUSIC = "assets/ikou.mp3"

TARGET_POSE = "OPEN_PALM"  # change if you want
OPEN_THRESH_FRAMES = 3

# ---------- AUDIO ----------
pygame.mixer.init()
pygame.mixer.music.load(MUSIC)

def play_music():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)  # loop forever

def stop_music():
    pygame.mixer.music.stop()

# ---------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def thumb_up(lm, handed="Right"):
    return lm[TIP["thumb"]].x < lm[3].x if handed == "Right" else lm[TIP["thumb"]].x > lm[3].x

def classify_pose(lm, handed="Right"):
    index = finger_up(lm, TIP["index"], PIP["index"])
    middle = finger_up(lm, TIP["middle"], PIP["middle"])
    ring = finger_up(lm, TIP["ring"], PIP["ring"])
    pinky = finger_up(lm, TIP["pinky"], PIP["pinky"])
    thumb = thumb_up(lm, handed)

    if thumb and index and middle and ring and pinky:
        return "OPEN_PALM"
    if not any([thumb, index, middle, ring, pinky]):
        return "FIST"
    if index and middle and not ring and not pinky:
        return "PEACE"
    if index and not middle and not ring and not pinky:
        return "POINT"
    return "UNKNOWN"

# ---------- IMAGE HELPERS ----------
def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img

def fit(img, h, w):
    img = cv2.resize(img, (w, h))
    return img

# ---------- MAIN ----------
def main():
    default_img = load_img(DEFAULT_IMG)
    open_img = load_img(OPEN_IMG)

    cap = cv2.VideoCapture(0)

    pose_frames = 0
    active = False

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            pose = "None"

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                handed = "Right"
                if res.multi_handedness:
                    handed = res.multi_handedness[0].classification[0].label

                pose = classify_pose(lm.landmark, handed)

            # ----- pose trigger logic -----
            if pose == TARGET_POSE:
                pose_frames += 1
            else:
                pose_frames = 0

            active = pose_frames >= OPEN_THRESH_FRAMES

            # ----- music control -----
            if active:
                play_music()
            else:
                stop_music()

            # ----- right image -----
            right = open_img if active else default_img
            right = fit(right, h, w)

            combined = np.hstack([frame, right])

            cv2.putText(combined, f"POSE: {pose}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.imshow("Pose → Image → Music", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    stop_music()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
