import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

PUNCH_COOLDOWN    = 0.4
BOTH_PUNCH_WINDOW = 0.5   # 양손 펀치로 인정할 최대 시간 차이 (초)
MIN_VISIBILITY    = 0.6
MIN_SPEED         = 12
OVERLAP_DIST_PX   = 60

WRIST_COLOR = {
    "right": (0, 100, 255),  # 빨강 (BGR)
    "left":  (255, 150, 0),  # 파랑 (BGR)
}


def _is_in_frame(lm, margin=0.05) -> bool:
    return (margin < lm.x < 1 - margin) and (margin < lm.y < 1 - margin)


class PunchDetector:
    def __init__(self):
        self.wrist_in_zone   = {"right": False, "left": False}
        self.wrist_prev_pos  = {"right": None,  "left": None}
        self.last_punch_time = {"right": 0.0,   "left": 0.0}

    def detect(self, lm, w, h, now, box_x1, box_x2, box_y1, box_y2, center_x) -> str | None:
        # 두 손목 겹침 → 판정 중단
        rw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        lw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        if np.hypot((rw.x - lw.x) * w, (rw.y - lw.y) * h) < OVERLAP_DIST_PX:
            return None

        # flip 보정: MediaPipe LEFT = 화면상 오른손, RIGHT = 왼손
        joints = {
            "right": (lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                      lm[mp_pose.PoseLandmark.LEFT_ELBOW],
                      lm[mp_pose.PoseLandmark.LEFT_WRIST]),
            "left":  (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                      lm[mp_pose.PoseLandmark.RIGHT_ELBOW],
                      lm[mp_pose.PoseLandmark.RIGHT_WRIST]),
        }

        for side, (shoulder, elbow, wrist) in joints.items():
            if not all(_is_in_frame(pt) for pt in (shoulder, elbow, wrist)):
                continue
            if min(shoulder.visibility, elbow.visibility, wrist.visibility) < MIN_VISIBILITY:
                continue

            wx_px = int(wrist.x * w)
            wy_px = int(wrist.y * h)

            if side == "right":
                in_zone = center_x < wx_px < box_x2 and box_y1 < wy_px < box_y2
            else:
                in_zone = box_x1 < wx_px < center_x and box_y1 < wy_px < box_y2

            was_in_zone = self.wrist_in_zone[side]
            self.wrist_in_zone[side] = in_zone

            prev  = self.wrist_prev_pos[side]
            speed = np.hypot(wx_px - prev[0], wy_px - prev[1]) if prev else 0
            self.wrist_prev_pos[side] = (wx_px, wy_px)

            just_entered = in_zone and not was_in_zone
            in_cooldown  = (now - self.last_punch_time[side]) < PUNCH_COOLDOWN

            if just_entered and speed > MIN_SPEED and not in_cooldown:
                self.last_punch_time[side] = now
                other = "left" if side == "right" else "right"
                other_recent = (now - self.last_punch_time[other]) < BOTH_PUNCH_WINDOW
                if other_recent:
                    print("🔥🔥 양손 펀치!")
                    return "both punch"
                action = "right punch" if side == "right" else "left punch"
                print(f"🔥 {'오른손' if side == 'right' else '왼손'} 펀치! (속도: {speed:.0f}px/f)")
                return action

        return None

    def draw(self, image, lm, w, h):
        for side, idx in [("right", mp_pose.PoseLandmark.LEFT_WRIST),
                          ("left",  mp_pose.PoseLandmark.RIGHT_WRIST)]:
            wlm = lm[idx]
            cx, cy = int(wlm.x * w), int(wlm.y * h)
            cv2.circle(image, (cx, cy), 18, WRIST_COLOR[side], -1)
            cv2.circle(image, (cx, cy), 18, (255, 255, 255), 3)
