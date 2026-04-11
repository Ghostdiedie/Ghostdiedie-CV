# import cv2
# import mediapipe as mp
# import queue
# import time
# import numpy as np

# # ── 조정 가능한 설정값 ────────────────────────────────────────────────────────

# PUNCH_ANGLE_THRESHOLD = 155    # 펀치로 인식할 팔꿈치 최소 각도 (높일수록 더 뻗어야 인식)
# PUNCH_ANGLE_VELOCITY  = 8.0    # 프레임당 최소 각도 증가 속도 (높일수록 둔감)
# PUNCH_COOLDOWN        = 0.4    # 같은 손 연속 펀치 최소 간격 (초)

# MIN_DETECTION_CONF    = 0.5
# MIN_TRACKING_CONF     = 0.5
# CAMERA_INDEX          = 0
# WINDOW_TITLE          = "Ghostdiedie - Motion Tracking"

# # ─────────────────────────────────────────────────────────────────────────────

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils


# def calc_angle(a, b, c) -> float:
#     """세 랜드마크로 b를 꼭짓점으로 하는 각도(도) 계산"""
#     a = np.array([a.x, a.y, a.z])
#     b = np.array([b.x, b.y, b.z])
#     c = np.array([c.x, c.y, c.z])
#     ba = a - b
#     bc = c - b
#     cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
#     return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


# def run_detection(action_queue: queue.Queue):
#     """OpenCV + MediaPipe 동작 감지 → action_queue에 액션 넣기"""
#     pose = mp_pose.Pose(
#         min_detection_confidence=MIN_DETECTION_CONF,
#         min_tracking_confidence=MIN_TRACKING_CONF,
#     )
#     cap = cv2.VideoCapture(CAMERA_INDEX)

#     prev_angle     = {"right": None, "left": None}
#     last_punch_time = {"right": 0.0, "left": 0.0}

#     print("카메라를 켭니다. 정면을 보고 주먹을 뻗어보세요!")

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.flip(frame, 1)

#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#         results = pose.process(image)
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(
#                 image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                 mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#             )

#             lm = results.pose_landmarks.landmark

#             joints = {
#                 "right": (
#                     lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
#                     lm[mp_pose.PoseLandmark.RIGHT_ELBOW],
#                     lm[mp_pose.PoseLandmark.RIGHT_WRIST],
#                 ),
#                 "left": (
#                     lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
#                     lm[mp_pose.PoseLandmark.LEFT_ELBOW],
#                     lm[mp_pose.PoseLandmark.LEFT_WRIST],
#                 ),
#             }

#             now = time.time()

#             for side, (shoulder, elbow, wrist) in joints.items():
#                 angle = calc_angle(shoulder, elbow, wrist)
#                 velocity = 0.0
#                 if prev_angle[side] is not None:
#                     velocity = angle - prev_angle[side]  # 양수 = 팔이 펴지는 중

#                 in_cooldown = (now - last_punch_time[side]) < PUNCH_COOLDOWN

#                 if angle > PUNCH_ANGLE_THRESHOLD and velocity > PUNCH_ANGLE_VELOCITY and not in_cooldown:
#                     action = "right punch" if side == "right" else "left punch"
#                     action_queue.put(action)
#                     last_punch_time[side] = now
#                     label = "오른손" if side == "right" else "왼손"
#                     print(f"{label} 펀치!  (각도: {angle:.1f}°, 속도: {velocity:.1f}°/f)")

#                 prev_angle[side] = angle

#                 # 디버그: 팔꿈치 각도 화면에 표시
#                 elbow_px = (int(elbow.x * image.shape[1]), int(elbow.y * image.shape[0]))
#                 cv2.putText(image, f"{angle:.0f}", elbow_px,
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#         cv2.imshow(WINDOW_TITLE, image)

#         if cv2.waitKey(10) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import queue
import time
import numpy as np
from collections import deque

# ── 안정화된 설정값 ────────────────────────────────────────────────────────
PUNCH_ANGLE_EXTENDED  = 140    # 팔을 뻗었을 때 (펀치 인식 기준)
PUNCH_ANGLE_RETRACTED = 110    # 팔을 굽혔을 때 (펀치 장전 기준)
PUNCH_COOLDOWN        = 0.4    # 펀치 쿨다운 (초)

MIN_DETECTION_CONF    = 0.5
MIN_TRACKING_CONF     = 0.5
MIN_VISIBILITY        = 0.6    # 랜드마크 최소 신뢰도 (이 이하면 판정 스킵)
ANGLE_SMOOTH_FRAMES   = 1     # 각도 스무딩에 사용할 프레임 수

CAMERA_INDEX          = 0
WINDOW_TITLE          = "Ghostdiedie - Motion Tracking"

WRIST_COLOR = {
    "right": (0, 100, 255),   # 빨강
    "left":  (255, 150, 0),   # 파랑
}
# ─────────────────────────────────────────────────────────────────────────────

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def is_in_frame(lm, margin=0.05) -> bool:
    """랜드마크가 화면 안에 있는지 확인 (카메라 밖 추정값 방지)"""
    return (margin < lm.x < 1 - margin) and (margin < lm.y < 1 - margin)

def calc_angle(a, b, c) -> float:
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

def run_detection(action_queue: queue.Queue):
    pose = mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )
    cap = cv2.VideoCapture(CAMERA_INDEX)

    arm_state = {"right": "retracted", "left": "retracted"}
    last_punch_time = {"right": 0.0, "left": 0.0}
    angle_buffer = {"right": deque(maxlen=ANGLE_SMOOTH_FRAMES),
                    "left":  deque(maxlen=ANGLE_SMOOTH_FRAMES)}
    wrist_in_zone  = {"right": False, "left": False}
    wrist_prev_pos = {"right": None,  "left": None}   # 이전 프레임 손목 위치 (픽셀)

    print("카메라를 켭니다. 팔을 굽혔다가 정면으로 뻗어보세요!")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w = image.shape[:2]

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=6),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3),
            )

            # 손목 점: 크고 뚜렷하게
            for side, lm_idx in [("right", mp_pose.PoseLandmark.LEFT_WRIST),
                                  ("left",  mp_pose.PoseLandmark.RIGHT_WRIST)]:
                wlm = lm[lm_idx]
                wx, wy = int(wlm.x * w), int(wlm.y * h)
                cv2.circle(image, (wx, wy), 18, WRIST_COLOR[side], -1)
                cv2.circle(image, (wx, wy), 18, (255, 255, 255), 3)

            # ── 몸통 박스 계산 ─────────────────────────────────────
            rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            lh = lm[mp_pose.PoseLandmark.LEFT_HIP]

            shoulder_y = int(min(rs.y, ls.y) * h)
            hip_y      = int(max(rh.y, lh.y) * h)

            box_x1  = int(min(rs.x, ls.x, rh.x, lh.x) * w)
            box_x2  = int(max(rs.x, ls.x, rh.x, lh.x) * w)
            box_y1  = shoulder_y
            box_y2  = shoulder_y + int((hip_y - shoulder_y) * 0.55)  # 어깨~골반의 55%까지만
            center_x = (box_x1 + box_x2) // 2

            # 박스 전체 흰색 반투명으로 그리기
            overlay = image.copy()
            cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (220, 220, 220), -1)
            cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
            cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 2)
            cv2.line(image, (center_x, box_y1), (center_x, box_y2), (255, 255, 255), 2)

            # flip된 화면에서 MediaPipe LEFT/RIGHT가 반전되므로 교체
            joints = {
                "right": (
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                    lm[mp_pose.PoseLandmark.LEFT_ELBOW],
                    lm[mp_pose.PoseLandmark.LEFT_WRIST],
                ),
                "left": (
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                    lm[mp_pose.PoseLandmark.RIGHT_ELBOW],
                    lm[mp_pose.PoseLandmark.RIGHT_WRIST],
                ),
            }

            now = time.time()

            for side, (shoulder, elbow, wrist) in joints.items():
                # visibility 필터 + 프레임 범위 체크
                out_of_frame = not all(is_in_frame(pt) for pt in (shoulder, elbow, wrist))
                min_vis = min(shoulder.visibility, elbow.visibility, wrist.visibility)
                low_conf = min_vis < MIN_VISIBILITY
                if out_of_frame or low_conf:
                    elbow_px = (int(elbow.x * w), int(elbow.y * h))
                    if out_of_frame:
                        label = f"OUT OF FRAME"
                    else:
                        label = f"LOW CONF {min_vis:.2f}"
                    cv2.putText(image, label, elbow_px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    continue

                raw_angle = calc_angle(shoulder, elbow, wrist)
                angle_buffer[side].append(raw_angle)
                angle = float(np.mean(angle_buffer[side]))

                if angle < PUNCH_ANGLE_RETRACTED:
                    arm_state[side] = "retracted"

                in_cooldown = (now - last_punch_time[side]) < PUNCH_COOLDOWN

                # ── 펀치 판정: 박스 진입 + 속도 필터 ────────────────
                wx_px = int(wrist.x * w)
                wy_px = int(wrist.y * h)
                if side == "right":
                    in_zone = center_x < wx_px < box_x2 and box_y1 < wy_px < box_y2
                else:
                    in_zone = box_x1 < wx_px < center_x and box_y1 < wy_px < box_y2

                was_in_zone = wrist_in_zone[side]
                wrist_in_zone[side] = in_zone

                prev = wrist_prev_pos[side]
                speed = np.hypot(wx_px - prev[0], wy_px - prev[1]) if prev else 0
                wrist_prev_pos[side] = (wx_px, wy_px)

                just_entered = in_zone and not was_in_zone
                fast_enough = speed > 12
                if just_entered and fast_enough and not in_cooldown:
                    action = "right punch" if side == "right" else "left punch"
                    action_queue.put(action)
                    last_punch_time[side] = now
                    print(f"🔥 {'오른손' if side == 'right' else '왼손'} 펀치! (속도: {speed:.0f}px/f)")

                # 디버그: 각도 표시
                elbow_px = (int(elbow.x * w), int(elbow.y * h))
                status_color = (0, 255, 0) if arm_state[side] == "retracted" else (0, 0, 255)
                cv2.putText(image, f"{angle:.0f} [{arm_state[side]}]", elbow_px,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow(WINDOW_TITLE, image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
