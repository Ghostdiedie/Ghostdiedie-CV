import cv2
import mediapipe as mp
import queue

# ── 조정 가능한 설정값 ────────────────────────────────────────────────────────

Z_THRESHOLD         = 0.15   # 펀치 감지 민감도 (낮을수록 민감, 높을수록 둔감)
MIN_DETECTION_CONF  = 0.5    # MediaPipe 최초 감지 신뢰도 (0.0 ~ 1.0)
MIN_TRACKING_CONF   = 0.5    # MediaPipe 추적 신뢰도 (0.0 ~ 1.0)
CAMERA_INDEX        = 0      # 카메라 번호 (보통 0이 기본 웹캠)
WINDOW_TITLE        = "Ghostdiedie - Motion Tracking"

# ─────────────────────────────────────────────────────────────────────────────

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def run_detection(action_queue: queue.Queue):
    """OpenCV + MediaPipe 동작 감지 → action_queue에 액션 넣기"""
    pose = mp_pose.Pose(
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )
    cap = cv2.VideoCapture(CAMERA_INDEX)

    state = {"r_punching": False, "l_punching": False}

    print("카메라를 켭니다. 정면을 보고 주먹을 뻗어보세요!")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            landmarks = results.pose_landmarks.landmark
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            r_wrist    = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            l_wrist    = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            if r_shoulder.z - r_wrist.z > Z_THRESHOLD:
                if not state["r_punching"]:
                    state["r_punching"] = True
                    action_queue.put("right punch")
                    print("오른손 펀치!")
            else:
                state["r_punching"] = False

            if l_shoulder.z - l_wrist.z > Z_THRESHOLD:
                if not state["l_punching"]:
                    state["l_punching"] = True
                    action_queue.put("left punch")
                    print("왼손 펀치!")
            else:
                state["l_punching"] = False

        cv2.imshow(WINDOW_TITLE, image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
