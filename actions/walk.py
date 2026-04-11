from collections import deque
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose

WALK_COOLDOWN       = 0.5   # 좌우 이벤트 최소 간격 (초)
WALK_COOLDOWN_DEPTH = 0.15  # 앞뒤 이벤트 최소 간격 — continuous 방식
EDGE_ZONE_RATIO     = 0.18  # 화면 좌우 끝 몇 % 를 zone으로 쓸지
MIN_VIS             = 0.4

BASELINE_FRAMES  = 20    # 기준 어깨너비 산출에 쓸 프레임 수
FORWARD_RATIO    = 1.13  # 기준 대비 13% 이상 넓어지면 앞으로
BACKWARD_RATIO   = 0.87  # 기준 대비 13% 이상 좁아지면 뒤로
DEPTH_CONFIRM_FRAMES = 4 # 오인식 방지: N 프레임 연속으로 threshold 넘어야 이벤트 발생


class WalkDetector:
    def __init__(self):
        self.shoulder_in_zone  = {"right": False, "left": False}
        self.last_walk_time    = {"right": 0.0,   "left": 0.0}
        self.last_depth_time   = {"forward": 0.0, "backward": 0.0}
        self.depth_buf         = deque(maxlen=BASELINE_FRAMES)
        self.moving_depth      = None   # "forward" | "backward" | None
        self.depth_confirm     = {"forward": 0, "backward": 0}  # 연속 프레임 카운터

    def detect(self, lm, w, h, now) -> str | None:
        # ── 좌우 이동 감지 ───────────────────────────────────────────
        lateral = self._detect_lateral(lm, w, now)

        # 좌우 이동이 감지되면 앞뒤는 무시 (우선순위: 좌우 > 앞뒤)
        if lateral:
            return lateral

        # ── 앞뒤 이동 감지 ───────────────────────────────────────────
        return self._detect_depth(lm, w, now)

    def _detect_lateral(self, lm, w, now) -> str | None:
        # flip 보정: MediaPipe LEFT = 화면상 오른쪽 어깨
        shoulders = {
            "right": lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
            "left":  lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
        }
        zone_right_x = int(w * (1 - EDGE_ZONE_RATIO))
        zone_left_x  = int(w * EDGE_ZONE_RATIO)

        for side, shoulder in shoulders.items():
            if shoulder.visibility < MIN_VIS:
                continue

            sx_px   = int(shoulder.x * w)
            in_zone = sx_px > zone_right_x if side == "right" else sx_px < zone_left_x

            was_in_zone = self.shoulder_in_zone[side]
            self.shoulder_in_zone[side] = in_zone

            if in_zone and not was_in_zone:
                if (now - self.last_walk_time[side]) >= WALK_COOLDOWN:
                    self.last_walk_time[side] = now
                    action = "walk right" if side == "right" else "walk left"
                    print(f"🚶 {'오른쪽' if side == 'right' else '왼쪽'}으로 이동!")
                    return action

        return None

    def _detect_depth(self, lm, w, now) -> str | None:
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        if ls.visibility < MIN_VIS or rs.visibility < MIN_VIS:
            return None

        shoulder_width = abs(ls.x - rs.x) * w

        # baseline 충분히 쌓일 때까지 수집
        if len(self.depth_buf) < BASELINE_FRAMES:
            self.depth_buf.append(shoulder_width)
            return None

        baseline = sum(self.depth_buf) / len(self.depth_buf)

        if shoulder_width > baseline * FORWARD_RATIO:
            self.depth_confirm["forward"]  += 1
            self.depth_confirm["backward"]  = 0
            # N 프레임 연속으로 threshold 넘어야 실제 이동으로 판정
            if self.depth_confirm["forward"] >= DEPTH_CONFIRM_FRAMES:
                self.moving_depth = "forward"
                if (now - self.last_depth_time["forward"]) >= WALK_COOLDOWN_DEPTH:
                    self.last_depth_time["forward"] = now
                    print(f"🚶 앞으로 이동! (너비: {shoulder_width:.0f}px / 기준: {baseline:.0f}px)")
                    return "walk forward"

        elif shoulder_width < baseline * BACKWARD_RATIO:
            self.depth_confirm["backward"] += 1
            self.depth_confirm["forward"]   = 0
            if self.depth_confirm["backward"] >= DEPTH_CONFIRM_FRAMES:
                self.moving_depth = "backward"
                if (now - self.last_depth_time["backward"]) >= WALK_COOLDOWN_DEPTH:
                    self.last_depth_time["backward"] = now
                    print(f"🚶 뒤로 이동! (너비: {shoulder_width:.0f}px / 기준: {baseline:.0f}px)")
                    return "walk backward"

        else:
            # 중립 구간: 카운터 리셋 + baseline 갱신
            self.depth_confirm["forward"]  = 0
            self.depth_confirm["backward"] = 0
            self.moving_depth = None
            self.depth_buf.append(shoulder_width)

        return None

    def draw(self, image, lm, w, h):
        zone_right_x = int(w * (1 - EDGE_ZONE_RATIO))
        zone_left_x  = int(w * EDGE_ZONE_RATIO)

        for x1, x2 in [(0, zone_left_x), (zone_right_x, w)]:
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, 0), (x2, h), (200, 200, 200), -1)
            cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)
            cv2.rectangle(image, (x1, 0), (x2, h), (255, 255, 255), 2)
