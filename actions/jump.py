from collections import deque
import mediapipe as mp

mp_pose = mp.solutions.pose

JUMP_COOLDOWN        = 0.8
JUMP_THRESHOLD       = 0.04   # 힙이 기준선보다 이 값(정규화) 이상 위로 올라가야 점프
SHOULDER_WIDE_RATIO  = 1.15   # 어깨너비 15% 이상 넓으면 앞으로 기울기 → 점프 무시
BASELINE_FRAMES      = 20
MIN_VIS              = 0.4


class JumpDetector:
    def __init__(self):
        self.baseline_buf      = deque(maxlen=BASELINE_FRAMES)
        self.shoulder_buf      = deque(maxlen=BASELINE_FRAMES)
        self.in_air            = False
        self.last_jump_time    = 0.0

    def detect(self, lm, w, h, now) -> str | None:
        lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
        rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        if lh.visibility < MIN_VIS or rh.visibility < MIN_VIS:
            return None

        hip_y          = (lh.y + rh.y) / 2
        shoulder_width = abs(ls.x - rs.x) if (ls.visibility >= MIN_VIS and rs.visibility >= MIN_VIS) else None

        if not self.in_air:
            self.baseline_buf.append(hip_y)
            if shoulder_width is not None:
                self.shoulder_buf.append(shoulder_width)

        if len(self.baseline_buf) < BASELINE_FRAMES:
            return None

        baseline = sum(self.baseline_buf) / len(self.baseline_buf)

        # 어깨너비 기반 앞으로 기울기 체크
        leaning_forward = False
        if shoulder_width is not None and len(self.shoulder_buf) >= 5:
            shoulder_baseline = sum(self.shoulder_buf) / len(self.shoulder_buf)
            leaning_forward = shoulder_width > shoulder_baseline * SHOULDER_WIDE_RATIO

        # 공중 진입
        if not self.in_air and hip_y < baseline - JUMP_THRESHOLD and not leaning_forward:
            self.in_air = True

        # 착지 감지 → 점프 이벤트 발생
        elif self.in_air and hip_y >= baseline - JUMP_THRESHOLD * 0.3:
            self.in_air = False
            if (now - self.last_jump_time) >= JUMP_COOLDOWN:
                self.last_jump_time = now
                print("⬆️ 점프!")
                return "jump"

        return None
