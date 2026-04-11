import mediapipe as mp

mp_pose = mp.solutions.pose

WALK_COOLDOWN     = 0.4    # 워크 이벤트 최소 간격 (초)
KNEE_RAISE_RATIO  = 0.05   # 무릎이 골반보다 이 값(정규화) 이상 위에 있어야 인식
MIN_VIS           = 0.4


class WalkDetector:
    def __init__(self):
        self.knee_state     = {"right": "grounded", "left": "grounded"}
        self.last_walk_time = {"right": 0.0, "left": 0.0}

    def detect(self, lm, w, h, now) -> str | None:
        # flip 보정: MediaPipe LEFT = 화면상 오른쪽 다리
        legs = {
            "right": (lm[mp_pose.PoseLandmark.LEFT_HIP],
                      lm[mp_pose.PoseLandmark.LEFT_KNEE]),
            "left":  (lm[mp_pose.PoseLandmark.RIGHT_HIP],
                      lm[mp_pose.PoseLandmark.RIGHT_KNEE]),
        }

        for side, (hip, knee) in legs.items():
            if min(hip.visibility, knee.visibility) < MIN_VIS:
                continue

            knee_raised = knee.y < hip.y - KNEE_RAISE_RATIO

            if knee_raised and self.knee_state[side] == "grounded":
                self.knee_state[side] = "raised"

            elif not knee_raised and self.knee_state[side] == "raised":
                self.knee_state[side] = "grounded"
                if (now - self.last_walk_time[side]) >= WALK_COOLDOWN:
                    self.last_walk_time[side] = now
                    label = "오른쪽" if side == "right" else "왼쪽"
                    print(f"🚶 {label}으로 이동!")
                    return "walk right" if side == "right" else "walk left"

        return None
