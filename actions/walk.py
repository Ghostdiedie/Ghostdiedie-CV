from collections import deque
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose

WALK_COOLDOWN       = 1.0   # 좌우 이벤트 최소 간격 (초)
WALK_COOLDOWN_DEPTH = 0.4   # 앞뒤 이벤트 최소 간격 (초) — zone에 있는 동안 계속 발사
MIN_VIS             = 0.4

# 캘리브레이션
CAL_FRAMES       = 120   # 수집 프레임 수 (약 4초)

# 격자 경계 (캘리브레이션 기준 상대값)
GRID_COL_RATIO   = 0.6   # 좌우 threshold = 어깨너비 × 이 값
FORWARD_RATIO    = 1.40  # 어깨너비 30% 이상 넓으면 forward zone
BACKWARD_RATIO   = 0.80  # 어깨너비 20% 이상 좁으면 backward zone

SHOULDER_SMOOTH  = 25
HIP_JUMP_THRESH  = 0.03


class WalkDetector:
    def __init__(self):
        self.last_walk_time  = {"right": 0.0, "left": 0.0}
        self.last_depth_time = {"forward": 0.0, "backward": 0.0}
        self.moving_lateral  = None
        self.moving_depth    = None

        # 캘리브레이션
        self.cal_buf         = deque(maxlen=CAL_FRAMES)
        self.calibrated      = False
        self.cal_nose_x      = None   # 중립 코 x (px)
        self.cal_shoulder_w  = None   # 중립 어깨너비 (px)

        # 경계선 스무딩 (draw용)
        self.left_edge_buf   = deque(maxlen=SHOULDER_SMOOTH)
        self.right_edge_buf  = deque(maxlen=SHOULDER_SMOOTH)

        # 힙 baseline
        self.hip_baseline_buf = deque(maxlen=CAL_FRAMES)
        self.shoulder_w_buf   = deque(maxlen=14)          # 어깨너비 스무딩

    def recalibrate(self):
        """외부에서 호출 가능한 재캘리브레이션"""
        self.cal_buf.clear()
        self.calibrated = False
        self.cal_nose_x = None
        self.cal_shoulder_w = None
        print("🔄 재캘리브레이션 시작...")

    def _try_calibrate(self, nose_x, shoulder_w):
        self.cal_buf.append((nose_x, shoulder_w))
        if len(self.cal_buf) >= CAL_FRAMES:
            xs = [v[0] for v in self.cal_buf]
            ws = [v[1] for v in self.cal_buf]
            self.cal_nose_x     = sum(xs) / len(xs)
            self.cal_shoulder_w = sum(ws) / len(ws)
            self.calibrated     = True
            # 게이지가 즉시 중앙에 오도록 버퍼를 캘리브레이션 값으로 채움
            self.shoulder_w_buf.clear()
            for _ in range(self.shoulder_w_buf.maxlen):
                self.shoulder_w_buf.append(self.cal_shoulder_w)
            print(f"✅ 캘리브레이션 완료! (코 중심: {self.cal_nose_x:.0f}px / 어깨너비: {self.cal_shoulder_w:.0f}px)")

    def detect(self, lm, w, h, now) -> str | None:
        lateral = self._detect_lateral(lm, w, now)
        if lateral:
            return lateral
        return self._detect_depth(lm, w, now)

    def _detect_lateral(self, lm, w, now) -> str | None:
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = lm[mp_pose.PoseLandmark.NOSE]
        if ls.visibility < MIN_VIS or rs.visibility < MIN_VIS or nose.visibility < MIN_VIS:
            return None

        nose_x      = nose.x * w
        shoulder_w  = abs(ls.x - rs.x) * w

        # 경계선 스무딩 버퍼 업데이트 (draw용)
        self.right_edge_buf.append(ls.x * w)
        self.left_edge_buf.append(rs.x * w)

        if not self.calibrated:
            self._try_calibrate(nose_x, shoulder_w)
            return None

        threshold = self.cal_shoulder_w * GRID_COL_RATIO
        delta     = nose_x - self.cal_nose_x

        if delta > threshold:
            self.moving_lateral = "right"
            if (now - self.last_walk_time["right"]) >= WALK_COOLDOWN:
                self.last_walk_time["right"] = now
                print(f"🚶 오른쪽 이동! (delta: {delta:.0f}px / threshold: {threshold:.0f}px)")
                return "walk right"
        elif delta < -threshold:
            self.moving_lateral = "left"
            if (now - self.last_walk_time["left"]) >= WALK_COOLDOWN:
                self.last_walk_time["left"] = now
                print(f"🚶 왼쪽 이동! (delta: {delta:.0f}px / threshold: {threshold:.0f}px)")
                return "walk left"
        else:
            self.moving_lateral = None

        return None

    def _detect_depth(self, lm, w, now) -> str | None:
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if ls.visibility < MIN_VIS or rs.visibility < MIN_VIS:
            return None

        if not self.calibrated:
            return None

        self.shoulder_w_buf.append(abs(ls.x - rs.x) * w)
        shoulder_w = sum(self.shoulder_w_buf) / len(self.shoulder_w_buf)

        # 힙 점프 체크
        lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
        rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        hip_y = (lh.y + rh.y) / 2
        self.hip_baseline_buf.append(hip_y)
        hip_baseline = sum(self.hip_baseline_buf) / len(self.hip_baseline_buf)
        is_jumping   = hip_y < hip_baseline - HIP_JUMP_THRESH

        if shoulder_w > self.cal_shoulder_w * FORWARD_RATIO and not is_jumping:
            self.moving_depth = "forward"
            if (now - self.last_depth_time["forward"]) >= WALK_COOLDOWN_DEPTH:
                self.last_depth_time["forward"] = now
                print(f"🚶 앞으로 이동! (너비: {shoulder_w:.0f}px / 기준: {self.cal_shoulder_w:.0f}px)")
                return "walk forward"

        elif shoulder_w < self.cal_shoulder_w * BACKWARD_RATIO:
            self.moving_depth = "backward"
            if (now - self.last_depth_time["backward"]) >= WALK_COOLDOWN_DEPTH:
                self.last_depth_time["backward"] = now
                print(f"🚶 뒤로 이동! (너비: {shoulder_w:.0f}px / 기준: {self.cal_shoulder_w:.0f}px)")
                return "walk backward"

        else:
            self.moving_depth = None   # 중립 → 멈춤

        return None

    def draw(self, image, lm, w, h):
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if ls.visibility < MIN_VIS or rs.visibility < MIN_VIS:
            return

        if not self.calibrated:
            # 캘리브레이션 진행 중 표시
            progress = int(len(self.cal_buf) / CAL_FRAMES * w)
            cv2.rectangle(image, (0, h - 10), (progress, h), (0, 255, 255), -1)
            cv2.putText(image, "Calibrating...", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return

        # ── 좌우 zone ────────────────────────────────────────────────
        threshold  = int(self.cal_shoulder_w * GRID_COL_RATIO)
        center_x   = int(self.cal_nose_x)
        left_line  = center_x - threshold
        right_line = center_x + threshold

        for (x1, x2), color in [
            ((0, left_line),   (255, 100, 0)),   # 왼쪽 zone: 파랑
            ((right_line, w),  (0, 100, 255)),   # 오른쪽 zone: 빨강
        ]:
            x1, x2 = max(0, x1), min(w, x2)
            if x1 >= x2:
                continue
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)
            cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
            cv2.rectangle(image, (x1, 0), (x2, h), color, 2)

        # ── 앞뒤 depth 게이지 바 (하단) ─────────────────────────────
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if ls.visibility < MIN_VIS or rs.visibility < MIN_VIS:
            return

        shoulder_w_now = sum(self.shoulder_w_buf) / len(self.shoulder_w_buf) if self.shoulder_w_buf else abs(ls.x - rs.x) * w
        bar_h          = 18
        bar_y          = h - bar_h - 2
        bar_w          = w

        # 전체 범위: calibration × 0.5 ~ × 1.7
        min_w = self.cal_shoulder_w * 0.5
        max_w = self.cal_shoulder_w * 1.7
        total = max_w - min_w

        # 구역 경계 x좌표
        bwd_x = int((self.cal_shoulder_w * BACKWARD_RATIO - min_w) / total * bar_w)
        fwd_x = int((self.cal_shoulder_w * FORWARD_RATIO  - min_w) / total * bar_w)

        # 현재 위치 x
        cur_x = int(max(0, min(bar_w - 1, (shoulder_w_now - min_w) / total * bar_w)))

        # 배경 구역 색칠
        cv2.rectangle(image, (0,       bar_y), (bwd_x,  bar_y + bar_h), (255, 100,  0), -1)  # backward: 파랑
        cv2.rectangle(image, (bwd_x,   bar_y), (fwd_x,  bar_y + bar_h), ( 50, 180, 50), -1)  # neutral:  초록
        cv2.rectangle(image, (fwd_x,   bar_y), (bar_w,  bar_y + bar_h), (  0, 100, 255), -1) # forward:  빨강

        # 현재 위치 마커
        cv2.rectangle(image, (cur_x - 3, bar_y - 2), (cur_x + 3, bar_y + bar_h + 2), (255, 255, 255), -1)

        # 라벨
        cv2.putText(image, "BACK", (4, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, "NEUTRAL", (bwd_x + 4, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, "FORWARD", (fwd_x + 4, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
