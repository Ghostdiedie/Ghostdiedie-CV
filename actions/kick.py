import mediapipe as mp

mp_pose = mp.solutions.pose

KICK_COOLDOWN    = 0.6
KNEE_RAISE_RATIO = 0.04   # 무릎이 골반보다 이 값(정규화) 이상 위에 있어야 킥
MIN_VIS          = 0.4


class KickDetector:
    def __init__(self):
        self.kick_state     = {"right": "grounded", "left": "grounded"}
        self.last_kick_time = {"right": 0.0, "left": 0.0}

    def detect(self, lm, w, h, now) -> str | None:
        # flip 보정: MediaPipe LEFT = 화면상 오른쪽 다리
        legs = {
            "right": (lm[mp_pose.PoseLandmark.LEFT_HIP],
                      lm[mp_pose.PoseLandmark.LEFT_KNEE],
                      lm[mp_pose.PoseLandmark.LEFT_ANKLE]),
            "left":  (lm[mp_pose.PoseLandmark.RIGHT_HIP],
                      lm[mp_pose.PoseLandmark.RIGHT_KNEE],
                      lm[mp_pose.PoseLandmark.RIGHT_ANKLE]),
        }

        for side, (hip, knee, ankle) in legs.items():
            if min(hip.visibility, knee.visibility, ankle.visibility) < MIN_VIS:
                continue

            # 무릎이 골반보다 위로 올라오면 킥 모션
            knee_raised = knee.y < hip.y - KNEE_RAISE_RATIO

            if knee_raised and self.kick_state[side] == "grounded":
                self.kick_state[side] = "raised"

            elif not knee_raised and self.kick_state[side] == "raised":
                self.kick_state[side] = "grounded"
                if (now - self.last_kick_time[side]) >= KICK_COOLDOWN:
                    self.last_kick_time[side] = now
                    label = "오른발" if side == "right" else "왼발"
                    print(f"🦵 {label} 발차기!")
                    return "kick"

        return None
