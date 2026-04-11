from collections import deque
import mediapipe as mp

mp_pose = mp.solutions.pose

HEADBUTT_COOLDOWN    = 0.8
NOSE_VEL_THRESH      = 0.025  # 코 Y 하강 속도 threshold
STRAIGHT_RATIO       = 0.4    # 수평 이동이 수직 이동의 이 비율 이하여야 직선으로 판정
SHOULDER_WIDE_RATIO  = 1.12   # 어깨너비 12% 이상 넓으면 앞으로 기울기 → headbutt 무시
SHOULDER_BUF_FRAMES  = 20
MIN_VIS              = 0.4


class HeadbuttDetector:
    def __init__(self):
        self.last_headbutt_time = 0.0
        self.prev_nose_y        = None
        self.prev_nose_x        = None
        self.shoulder_buf       = deque(maxlen=SHOULDER_BUF_FRAMES)

    def detect(self, lm, w, h, now) -> str | None:
        nose = lm[mp_pose.PoseLandmark.NOSE]
        ls   = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs   = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # 어깨너비 baseline 업데이트
        if ls.visibility >= MIN_VIS and rs.visibility >= MIN_VIS:
            shoulder_width = abs(ls.x - rs.x)
            self.shoulder_buf.append(shoulder_width)

        if nose.visibility < MIN_VIS:
            self.prev_nose_y = None
            self.prev_nose_x = None
            return None

        curr_y, curr_x = nose.y, nose.x
        prev_y, prev_x = self.prev_nose_y, self.prev_nose_x
        self.prev_nose_y = curr_y
        self.prev_nose_x = curr_x

        if prev_y is None:
            return None

        delta_y = curr_y - prev_y   # 양수 = 아래로 내려감
        delta_x = abs(curr_x - prev_x)  # 수평 이동량

        # 아래로 빠르게 + 수평 이동이 수직의 STRAIGHT_RATIO 이하 = 일직선 하강
        if delta_y <= NOSE_VEL_THRESH:
            return None
        if delta_x > delta_y * STRAIGHT_RATIO:
            return None

        # 어깨가 넓어지고 있으면 (앞으로 기울기) → headbutt 아님
        if len(self.shoulder_buf) >= 5:
            shoulder_baseline = sum(self.shoulder_buf) / len(self.shoulder_buf)
            current_width = abs(ls.x - rs.x) if (ls.visibility >= MIN_VIS and rs.visibility >= MIN_VIS) else 0
            if current_width > shoulder_baseline * SHOULDER_WIDE_RATIO:
                return None

        if (now - self.last_headbutt_time) >= HEADBUTT_COOLDOWN:
            self.last_headbutt_time = now
            print(f"💥 박치기! (속도: {delta_y:.3f})")
            return "headbutt"

        return None
