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
PUNCH_ANGLE_EXTENDED  = 150    # 팔을 뻗었을 때 (펀치 인식 기준)
PUNCH_ANGLE_RETRACTED = 100    # 팔을 굽혔을 때 (펀치 장전 기준)
PUNCH_COOLDOWN        = 0.4    # 펀치 쿨다운 (초)

MIN_DETECTION_CONF    = 0.5
MIN_TRACKING_CONF     = 0.5
MIN_VISIBILITY        = 0.6    # 랜드마크 최소 신뢰도 (이 이하면 판정 스킵)
ANGLE_SMOOTH_FRAMES   = 1     # 각도 스무딩에 사용할 프레임 수

CAMERA_INDEX          = 0
WINDOW_TITLE          = "Ghostdiedie - Motion Tracking"
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
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )
    cap = cv2.VideoCapture(CAMERA_INDEX)

    arm_state = {"right": "retracted", "left": "retracted"}
    last_punch_time = {"right": 0.0, "left": 0.0}
    angle_buffer = {"right": deque(maxlen=ANGLE_SMOOTH_FRAMES),
                    "left":  deque(maxlen=ANGLE_SMOOTH_FRAMES)}

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

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            joints = {
                "right": (
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                    lm[mp_pose.PoseLandmark.RIGHT_ELBOW],
                    lm[mp_pose.PoseLandmark.RIGHT_WRIST],
                    lm[mp_pose.PoseLandmark.RIGHT_HIP],
                ),
                "left": (
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                    lm[mp_pose.PoseLandmark.LEFT_ELBOW],
                    lm[mp_pose.PoseLandmark.LEFT_WRIST],
                    lm[mp_pose.PoseLandmark.LEFT_HIP],
                ),
            }

            now = time.time()

            for side, (shoulder, elbow, wrist, hip) in joints.items():
                # ── 개선 1: visibility 필터 + 프레임 범위 체크 ──────────
                out_of_frame = not all(is_in_frame(lm) for lm in (shoulder, elbow, wrist))
                low_conf = min(shoulder.visibility, elbow.visibility, wrist.visibility) < MIN_VISIBILITY
                if out_of_frame or low_conf:
                    elbow_px = (int(elbow.x * image.shape[1]), int(elbow.y * image.shape[0]))
                    label = "OUT OF FRAME" if out_of_frame else "LOW CONF"
                    cv2.putText(image, label, elbow_px,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                    continue

                raw_angle = calc_angle(shoulder, elbow, wrist)

                # ── 개선 2: 각도 스무딩 ──────────────────────────────────
                angle_buffer[side].append(raw_angle)
                angle = float(np.mean(angle_buffer[side]))

                # 팔을 굽히면 '장전(retracted)' 상태
                if angle < PUNCH_ANGLE_RETRACTED:
                    arm_state[side] = "retracted"

                # 펀치 판정: 뻗음 + 장전된 상태였음 + 손목이 허리보다 위
                is_arm_up = wrist.y < hip.y
                in_cooldown = (now - last_punch_time[side]) < PUNCH_COOLDOWN

                if angle > PUNCH_ANGLE_EXTENDED and arm_state[side] == "retracted" and is_arm_up and not in_cooldown:
                    action = "right punch" if side == "right" else "left punch"
                    action_queue.put(action)

                    last_punch_time[side] = now
                    arm_state[side] = "extended"

                    label = "오른손" if side == "right" else "왼손"
                    print(f"🔥 {label} 펀치 적중! (각도: {angle:.1f}°)")

                # 디버그: 팔꿈치 각도와 현재 상태 표시
                elbow_px = (int(elbow.x * image.shape[1]), int(elbow.y * image.shape[0]))
                status_color = (0, 255, 0) if arm_state[side] == "retracted" else (0, 0, 255)
                cv2.putText(image, f"{angle:.0f} [{arm_state[side]}]", elbow_px,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow(WINDOW_TITLE, image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
