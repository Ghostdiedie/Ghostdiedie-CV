from collections import deque
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose

WALK_COOLDOWN       = 0.5   # 좌우 이벤트 최소 간격 (초)
WALK_COOLDOWN_DEPTH = 1.0   # 앞뒤 이벤트 최소 간격
MIN_VIS             = 0.4

BASELINE_FRAMES  = 30    # 기준 어깨너비 산출에 쓸 프레임 수
FORWARD_RATIO    = 1.25  # 기준 대비 25% 이상 넓어지면 forward
BACKWARD_RATIO   = 0.75  # 기준 대비 25% 이상 좁아지면 backward
SHOULDER_SMOOTH  = 25    # 어깨 경계선 스무딩 프레임 수 (클수록 딜레이 증가)
HIP_JUMP_THRESH  = 0.03  # 힙이 이 값 이상 올라가 있으면 점프 중 → forward 스킵


class WalkDetector:
    def __init__(self):
        self.last_walk_time   = {"right": 0.0,   "left": 0.0}
        self.last_depth_time  = {"forward": 0.0, "backward": 0.0}
        self.depth_buf        = deque(maxlen=BASELINE_FRAMES)
        self.moving_depth     = None   # "forward" | "backward" | None
        self.moving_lateral   = None   # "right" | "left" | None
        self.left_edge_buf    = deque(maxlen=SHOULDER_SMOOTH)  # 왼쪽 경계 스무딩
        self.right_edge_buf   = deque(maxlen=SHOULDER_SMOOTH)  # 오른쪽 경계 스무딩
        self.hip_baseline_buf = deque(maxlen=BASELINE_FRAMES)  # 힙 Y baseline

    def detect(self, lm, w, h, now) -> str | None:
        # ── 좌우 이동 감지 ───────────────────────────────────────────
        lateral = self._detect_lateral(lm, w, now)
        if lateral:
            return lateral

        # ── 앞뒤 이동 감지 ───────────────────────────────────────────
        return self._detect_depth(lm, w, now)

    def _detect_lateral(self, lm, w, now) -> str | None:
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if ls.visibility < MIN_VIS or rs.visibility < MIN_VIS:
            return None

        nose = lm[mp_pose.PoseLandmark.NOSE]
        if nose.visibility < MIN_VIS:
            return None

        nose_x = nose.x * w

        # flip 보정: LEFT_SHOULDER = 화면상 오른쪽, RIGHT_SHOULDER = 왼쪽
        # 스무딩 버퍼에 현재 어깨 위치 추가
        self.right_edge_buf.append(ls.x * w)
        self.left_edge_buf.append(rs.x * w)

        if len(self.right_edge_buf) < 5:
            return None

        # 경계선 = 과거 어깨 위치 평균 (딜레이 효과)
        right_edge = sum(self.right_edge_buf) / len(self.right_edge_buf)
        left_edge  = sum(self.left_edge_buf)  / len(self.left_edge_buf)

        if nose_x > right_edge:
            if self.moving_lateral != "right":
                self.moving_lateral = "right"
                if (now - self.last_walk_time["right"]) >= WALK_COOLDOWN:
                    self.last_walk_time["right"] = now
                    print(f"🚶 오른쪽으로 이동! (nose: {nose_x:.0f}px / edge: {right_edge:.0f}px)")
                    return "walk right"
        elif nose_x < left_edge:
            if self.moving_lateral != "left":
                self.moving_lateral = "left"
                if (now - self.last_walk_time["left"]) >= WALK_COOLDOWN:
                    self.last_walk_time["left"] = now
                    print(f"🚶 왼쪽으로 이동! (nose: {nose_x:.0f}px / edge: {left_edge:.0f}px)")
                    return "walk left"
        else:
            self.moving_lateral = None

        return None

    def _detect_depth(self, lm, w, now) -> str | None:
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        if ls.visibility < MIN_VIS or rs.visibility < MIN_VIS:
            return None

        shoulder_width = abs(ls.x - rs.x) * w

        # 힙 Y baseline 업데이트
        lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
        rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        hip_y = (lh.y + rh.y) / 2
        self.hip_baseline_buf.append(hip_y)
        hip_baseline = sum(self.hip_baseline_buf) / len(self.hip_baseline_buf)
        is_jumping = hip_y < hip_baseline - HIP_JUMP_THRESH

        # baseline 충분히 쌓일 때까지 수집
        if len(self.depth_buf) < BASELINE_FRAMES:
            self.depth_buf.append(shoulder_width)
            return None

        baseline = sum(self.depth_buf) / len(self.depth_buf)

        if shoulder_width > baseline * FORWARD_RATIO and not is_jumping:
            self.moving_depth = "forward"
            if (now - self.last_depth_time["forward"]) >= WALK_COOLDOWN_DEPTH:
                self.last_depth_time["forward"] = now
                print(f"🚶 앞으로 이동! (너비: {shoulder_width:.0f}px / 기준: {baseline:.0f}px)")
                return "walk forward"

        elif shoulder_width < baseline * BACKWARD_RATIO:
            self.moving_depth = "backward"
            if (now - self.last_depth_time["backward"]) >= WALK_COOLDOWN_DEPTH:
                self.last_depth_time["backward"] = now
                print(f"🚶 뒤로 이동! (너비: {shoulder_width:.0f}px / 기준: {baseline:.0f}px)")
                return "walk backward"

        else:
            self.moving_depth = None
            self.depth_buf.append(shoulder_width)

        return None

    def draw(self, image, lm, w, h):
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if ls.visibility < MIN_VIS or rs.visibility < MIN_VIS:
            return

        # 스무딩된 경계선 사용 (버퍼 없으면 현재 값으로 fallback)
        right_edge = int(sum(self.right_edge_buf) / len(self.right_edge_buf)) if self.right_edge_buf else int(ls.x * w)
        left_edge  = int(sum(self.left_edge_buf)  / len(self.left_edge_buf))  if self.left_edge_buf  else int(rs.x * w)

        for (x1, x2), color in [
            ((0, left_edge),    (255, 100, 0)),   # 왼쪽 zone: 파랑
            ((right_edge, w),   (0, 100, 255)),   # 오른쪽 zone: 빨강
        ]:
            x1, x2 = max(0, x1), min(w, x2)
            if x1 >= x2:
                continue
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)
            cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
            cv2.rectangle(image, (x1, 0), (x2, h), color, 2)
