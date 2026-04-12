import mediapipe as mp

mp_pose = mp.solutions.pose

KICK_COOLDOWN   = 0.6
HIP_DIFF_THRESH = 0.06  # 한쪽 골반이 반대쪽보다 이 값 이상 높으면 킥
MIN_VIS         = 0.4


class KickDetector:
    def __init__(self):
        self.last_kick_time = {"right": 0.0, "left": 0.0}
        self.kick_state     = {"right": "grounded", "left": "grounded"}

    def detect(self, lm, w, h, now) -> str | None:
        lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
        rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        if lh.visibility < MIN_VIS or rh.visibility < MIN_VIS:
            return None

        # flip 보정: LEFT_HIP = 화면상 오른쪽, RIGHT_HIP = 왼쪽
        # 오른발 차기 → LEFT_HIP(화면 오른쪽) 올라옴 → lh.y 작아짐
        # 왼발 차기 → RIGHT_HIP(화면 왼쪽) 올라옴 → rh.y 작아짐
        sides = {
            "right": rh.y - lh.y,   # 양수 = 왼쪽 골반이 오른쪽보다 높음 = 오른발 킥
            "left":  lh.y - rh.y,   # 양수 = 오른쪽 골반이 왼쪽보다 높음 = 왼발 킥
        }

        for side, diff in sides.items():
            if diff > HIP_DIFF_THRESH:
                if self.kick_state[side] == "grounded":
                    self.kick_state[side] = "raised"
                    if (now - self.last_kick_time[side]) >= KICK_COOLDOWN:
                        self.last_kick_time[side] = now
                        label = "오른발" if side == "right" else "왼발"
                        print(f"🦵 {label} 발차기! (골반 차이: {diff:.3f})")
                        return "kick"
            else:
                self.kick_state[side] = "grounded"

        return None
