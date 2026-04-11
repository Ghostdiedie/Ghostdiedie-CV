import cv2
import mediapipe as mp
import queue

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def run_detection(action_queue: queue.Queue):
    """OpenCV + MediaPipe 동작 감지 → action_queue에 액션 넣기"""
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

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

            z_threshold = 0.15

            if r_shoulder.z - r_wrist.z > z_threshold:
                if not state["r_punching"]:
                    state["r_punching"] = True
                    action_queue.put("right punch")
                    print("오른손 펀치!")
            else:
                state["r_punching"] = False

            if l_shoulder.z - l_wrist.z > z_threshold:
                if not state["l_punching"]:
                    state["l_punching"] = True
                    action_queue.put("left punch")
                    print("왼손 펀치!")
            else:
                state["l_punching"] = False

        cv2.imshow("Ghostdiedie - Motion Tracking", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
