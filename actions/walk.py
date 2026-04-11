from collections import deque
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose

WALK_COOLDOWN    = 0.5   # 워크 이벤트 최소 간격 (초)
EDGE_ZONE_RATIO  = 0.18  # anchor 기준 좌우 zone 크기 (어깨너비 대비 비율)
MIN_VIS          = 0.4

BASELINE_FRAMES  = 30    # 기준 어깨너비 산출에 쓸 프레임 수
FORWARD_RATIO    = 1.07  # 기준 대비 7% 이상 넓어지면 forward
BACKWARD_RATIO   = 0.87  # 기준 대비 13% 이상 좁아지면 backward


class WalkDetector:
    def __init__(self):
        self.anchor_cx        = None   # 좌우 기준 몸 중심 x (px)
        self.shoulder_in_zone = {"right": False, "left": False}
        self.last_walk_time   = {"right": 0.0,   "left": 0.0}
        self.last_depth_time  = {"forward": 0.0, "backward": 0.0}
        self.depth_buf        = deque(maxlen=BASELINE_FRAMES)
        self.moving_depth     = None  # "forward" | "backward" | None

    def detect(self, lm, w, h, now) -> str | None:
        # ── 좌우 이동 감지 ───────────────────────────────────────────
        lateral = self._detect_lateral(lm, w, now)

        # 좌우 이동이 감지되면 앞뒤는 무시 (우선순위: 좌우 > 앞뒤)
        if lateral:
            self._update_depth_baseline(lm, w)
            return lateral

        # ── 앞뒤 이동 감지 ───────────────────────────────────────────
        return self._detect_depth(lm, w, now)

    def _get_body_center(self, lm, w):
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if ls.visibility < MIN_VIS or rs.visibility < MIN_VIS:
            return None, None
        center_x = int((ls.x + rs.x) / 2 * w)
        shoulder_width = abs(ls.x - rs.x) * w
        return center_x, shoulder_width

    def _detect_lateral(self, lm, w, now) -> str | None:
        center_x, shoulder_width = self._get_body_center(lm, w)
        if center_x is None:
            return None

        # 첫 프레임에 anchor 초기화
        if self.anchor_cx is None:
            self.anchor_cx = center_x

        zone_offset  = int(shoulder_width * EDGE_ZONE_RATIO * (w / shoulder_width) * shoulder_width / w)
        # 어깨너비 절반을 zone offset으로 사용
        zone_offset  = int(shoulder_width * 0.5)
        zone_right_x = self.anchor_cx + zone_offset
        zone_left_x  = self.anchor_cx - zone_offset

        # flip 보정: MediaPipe LEFT = 화면상 오른쪽 어깨
        shoulder_lm = {
            "right": lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
            "left":  lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
        }

        for side, shoulder in shoulder_lm.items():
            if shoulder.visibility < MIN_VIS:
                continue

            sx_px   = int(shoulder.x * w)
            in_zone = sx_px > zone_right_x if side == "right" else sx_px < zone_left_x

            was_in_zone = self.shoulder_in_zone[side]
            self.shoulder_in_zone[side] = in_zone

            if in_zone and not was_in_zone:
                if (now - self.last_walk_time[side]) >= WALK_COOLDOWN:
                    self.last_walk_time[side] = now
                    # 감지된 위치를 새 anchor로 업데이트
                    self.anchor_cx = center_x
                    self.shoulder_in_zone = {"right": False, "left": False}
                    action = "walk right" if side == "right" else "walk left"
                    print(f"🚶 {'오른쪽' if side == 'right' else '왼쪽'}으로 이동! (anchor: {self.anchor_cx}px)")
                    return action

        return None

    def _update_depth_baseline(self, lm, w):
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if ls.visibility >= MIN_VIS and rs.visibility >= MIN_VIS and self.moving_depth is None:
            self.depth_buf.append(abs(ls.x - rs.x) * w)

    def _detect_depth(self, lm, w, now) -> str | None:
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        if ls.visibility < MIN_VIS or rs.visibility < MIN_VIS:
            return None

        shoulder_width = abs(ls.x - rs.x) * w

        if self.moving_depth is None:
            self.depth_buf.append(shoulder_width)

        if len(self.depth_buf) < BASELINE_FRAMES:
            return None

        baseline = sum(self.depth_buf) / len(self.depth_buf)

        if shoulder_width > baseline * FORWARD_RATIO:
            if self.moving_depth != "forward":
                self.moving_depth = "forward"
                if (now - self.last_depth_time["forward"]) >= WALK_COOLDOWN:
                    self.last_depth_time["forward"] = now
                    print(f"🚶 앞으로 이동! (너비: {shoulder_width:.0f}px / 기준: {baseline:.0f}px)")
                    return "walk forward"

        elif shoulder_width < baseline * BACKWARD_RATIO:
            if self.moving_depth != "backward":
                self.moving_depth = "backward"
                if (now - self.last_depth_time["backward"]) >= WALK_COOLDOWN:
                    self.last_depth_time["backward"] = now
                    # 뒤로 이동한 위치를 새 baseline으로 리셋
                    self.depth_buf.clear()
                    self.depth_buf.append(shoulder_width)
                    print(f"🚶 뒤로 이동! (너비: {shoulder_width:.0f}px / 기준: {baseline:.0f}px)")
                    return "walk backward"

        else:
            self.moving_depth = None

        return None

    def draw(self, image, lm, w, h):
        center_x, shoulder_width = self._get_body_center(lm, w)
        if center_x is None or self.anchor_cx is None:
            return

        zone_offset  = int(shoulder_width * 0.5)
        zone_right_x = self.anchor_cx + zone_offset
        zone_left_x  = self.anchor_cx - zone_offset

        for x1, x2 in [(0, zone_left_x), (zone_right_x, w)]:
            x1, x2 = max(0, x1), min(w, x2)
            if x1 >= x2:
                continue
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, 0), (x2, h), (200, 200, 200), -1)
            cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)
            cv2.rectangle(image, (x1, 0), (x2, h), (255, 255, 255), 2)
