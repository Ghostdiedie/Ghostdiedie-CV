import mediapipe as mp

mp_pose = mp.solutions.pose

KICK_COOLDOWN    = 0.6
KNEE_VEL_THRESH  = 0.018  # 연속 프레임 무릎 상승 속도 threshold
KNEE_RAISE_RATIO = 0.05   # 골반보다 이 값 이상 위에 있으면 이미 찬 것으로 판정
MIN_VIS          = 0.3


class KickDetector:
    def __init__(self):
        self.last_kick_time = {"right": 0.0, "left": 0.0}
        self.prev_knee_y    = {"right": None, "left": None}

    def detect(self, lm, w, h, now) -> str | None:
        # flip 보정: MediaPipe LEFT = 화면상 오른쪽 다리
        legs = {
            "right": (lm[mp_pose.PoseLandmark.LEFT_KNEE],  lm[mp_pose.PoseLandmark.LEFT_HIP]),
            "left":  (lm[mp_pose.PoseLandmark.RIGHT_KNEE], lm[mp_pose.PoseLandmark.RIGHT_HIP]),
        }

        for side, (knee, hip) in legs.items():
            in_cooldown = (now - self.last_kick_time[side]) < KICK_COOLDOWN
            prev_y      = self.prev_knee_y[side]

            if knee.visibility >= MIN_VIS:
                curr_y = knee.y
                fired  = False

                if not in_cooldown:
                    # 1. 속도 기반: 연속 프레임에서 무릎이 빠르게 위로
                    if prev_y is not None and (prev_y - curr_y) > KNEE_VEL_THRESH:
                        fired = True
                    # 2. 등장 기반: 프레임 밖에서 들어왔는데 이미 골반 위
                    elif prev_y is None and hip.visibility >= MIN_VIS and curr_y < hip.y - KNEE_RAISE_RATIO:
                        fired = True

                self.prev_knee_y[side] = curr_y

                if fired:
                    self.last_kick_time[side] = now
                    label = "오른발" if side == "right" else "왼발"
                    print(f"🦵 {label} 발차기! ")
                    return "kick"
            else:
                self.prev_knee_y[side] = None

        return None
